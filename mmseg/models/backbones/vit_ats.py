# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import (
    build_norm_layer,
    constant_init,
    kaiming_init,
    normal_init,
    trunc_normal_init,
)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ...ops.ats import AdaptiveTokenSampler
from ..builder import BACKBONES
from ..utils import PatchEmbed

from torch.utils.checkpoint import checkpoint

from .helpers.vit_helper import load_weights_from_npz, load_weights_from_HRT_Cls_format
from .vit import VisionTransformer
from mmseg.models.utils import get_aspect_ratio, reshape_as_aspect_ratio
from ...ops.reconstruction import TokenReconstructionBlock

from mmcv import deprecated_api_warning
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.drop import build_dropout


class ATSMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 use_ats=True,
                 drop_tokens=False,
                 **kwargs):
        super(ATSMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        if use_ats:
            self.attn = AdaptiveTokenSampler(
                embed_dims,
                num_heads,
                attn_drop,
                drop_tokens=drop_tokens,)
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        if self.batch_first:

            def _bnc_to_nbc(forward):
                """Because the dataflow('key', 'query', 'value') of
                ``torch.nn.MultiheadAttention`` is (num_query, batch,
                embed_dims), We should adjust the shape of dataflow from
                batch_first (batch, num_query, embed_dims) to num_query_first
                (num_query ,batch, embed_dims), and recover ``attn_output``
                from num_query_first to batch_first."""

                def forward_wrapper(**kwargs):
                    convert_keys = ('key', 'query', 'value')
                    for key in kwargs.keys():
                        if key in convert_keys:
                            kwargs[key] = kwargs[key].transpose(0, 1)
                    out, out_weights, selected_x, policy, sampler = forward(**kwargs)
                    return out.transpose(0, 1), out_weights, selected_x.transpose(0, 1), policy.transpose(0, 1), sampler

                return forward_wrapper

            self.attn.forward = _bnc_to_nbc(self.attn.forward)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                policy=None,
                n_tokens=None,
                raw_x=None,
                n_ref_tokens=None,
                reconstructer=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out, out_weights, selected_x, policy, sampler = self.attn(
            query=query,
            key=key,
            value=value,
            policy=policy,
            n_tokens=n_tokens,
            raw_x=raw_x,
            n_ref_tokens=n_ref_tokens,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)
        reconstructer.update_state(feat_after_pooling=selected_x[:, 1:])

        return selected_x + self.dropout_layer(self.proj_drop(out)), policy, sampler


class ATSTransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        ATS_cfg (dict): The ATSivation config for FFNs.
            Defalut: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        batch_first=True,
        use_ats=True,
        drop_tokens=False,
    ):
        super(ATSTransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = ATSMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias,
            use_ats=use_ats,
            drop_tokens=drop_tokens,
        )

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            act_cfg=act_cfg,
        )

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(
        self, 
        x,
        n_tokens,
        policy = None,
        sampler = None,
        n_ref_tokens = 197,
        reconstructer=None,
    ):
        x = self.norm1(x)
        reconstructer.update_state(feat_before_pooling=x[:, 1:])
        x, policy, sampler = self.attn(
            query=x, 
            key=x, 
            value=x, 
            need_weights=False,
            policy=policy,
            sampler=sampler,
            n_tokens=n_tokens,
            raw_x=x,
            n_ref_tokens=n_ref_tokens,
            reconstructer=reconstructer,
        )

        x = x * policy
        x = self.ffn(self.norm2(x), identity=x)
        x = x * policy
        return x


@BACKBONES.register_module()
class VisionTransformerATS(VisionTransformer):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        embed_dims=768, 
        num_layers=12, 
        num_heads=12, 
        mlp_ratio=4, 
        out_indices=-1, 
        qkv_bias=True, 
        drop_rate=0, 
        attn_drop_rate=0, 
        drop_path_rate=0, 
        with_cls_token=True, 
        output_cls_token=False, 
        norm_cfg=..., 
        act_cfg=..., 
        patch_norm=False, 
        final_norm=False, 
        interpolate_mode="bicubic", 
        num_fcs=2, 
        norm_eval=False, 
        with_cp=False, 
        pretrained=None, 
        init_cfg=None, 
        use_checkpoint=False, 
        resize_shape=None, 
        ats_block_index=None,
        num_tokens=197,    # num of tokens to be sampled
        drop_tokens=False,
        reconsctruct_cfg=None,
        **kwargs
    ):
        super().__init__(
            img_size, 
            patch_size, 
            in_channels, 
            embed_dims, 
            num_layers, 
            num_heads, 
            mlp_ratio, 
            out_indices, 
            qkv_bias, 
            drop_rate, 
            attn_drop_rate, 
            drop_path_rate, 
            with_cls_token, 
            output_cls_token, 
            norm_cfg, 
            act_cfg, 
            patch_norm, 
            final_norm, 
            interpolate_mode, 
            num_fcs, 
            norm_eval, 
            with_cp, 
            pretrained, 
            init_cfg, 
            use_checkpoint, 
            resize_shape, 
            **kwargs
        )

        self.num_tokens = num_tokens
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.ats_block_index = ats_block_index
        if ats_block_index is not None:
            self.layers[ats_block_index] = ATSTransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                attn_drop_rate=attn_drop_rate,
                drop_rate=drop_rate,
                drop_path_rate=dpr[ats_block_index],
                num_fcs=num_fcs,
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                batch_first=True,
                drop_tokens=drop_tokens,
            )

        self.reconstruction_layer = TokenReconstructionBlock(embed_dim=embed_dims, cfg=reconsctruct_cfg)

    def reconstruct(self, x, reconstructer):
        if reconstructer.org_num_features == x.shape[1]:
            x = reshape_as_aspect_ratio(x, reconstructer.aspect_ratio)
            return x
        reconstructer.update_state(used=True)
        x, reshaped = reconstructer.call(x)
        if not reshaped:
            x = reshape_as_aspect_ratio(x, reconstructer.aspect_ratio)
        return x

    def forward(self, inputs):
        B, _, H, W = inputs.shape

        x, hw_shape = (
            self.patch_embed(inputs),
            (self.patch_embed.DH, self.patch_embed.DW),
        )
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        # prepare for passing transformer
        aspect_ratio = get_aspect_ratio(*hw_shape)
        reconstructer = self.reconstruction_layer.derive_unpooler()
        reconstructer.aspect_ratio = aspect_ratio
        reconstructer.org_num_features = x.shape[1]

        outs = []
        init_n = x.shape[1]
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        sampler = torch.nonzero(policy)
        for i, layer in enumerate(self.layers):
            if i == self.ats_block_index:
                layer = partial(
                    layer, 
                    n_tokens=init_n,
                    policy=policy,
                    sampler=sampler,
                    n_ref_tokens=init_n,
                    reconstructer=reconstructer,
                )
            if self.use_checkpoint:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = self.reconstruct(out, reconstructer)
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

