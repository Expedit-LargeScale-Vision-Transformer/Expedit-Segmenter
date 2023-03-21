# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

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
from ..builder import BACKBONES
from ..utils import PatchEmbed

from torch.utils.checkpoint import checkpoint

from .helpers.vit_helper import load_weights_from_npz, load_weights_from_HRT_Cls_format

from .vit import TransformerEncoderLayer, VisionTransformer

def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out

# make MultiheadAttention output with attention
class MHA(MultiheadAttention):
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
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

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, attn = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out)), attn

class EvoTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
        self, 
        embed_dims, 
        num_heads, 
        feedforward_channels, 
        drop_rate=0, 
        attn_drop_rate=0, 
        drop_path_rate=0, 
        num_fcs=2, 
        qkv_bias=True, 
        act_cfg=..., 
        norm_cfg=..., 
        batch_first=True, 
        prune_ratio=1.0, 
        tradeoff=0.5,
    ):
        super().__init__(
            embed_dims, 
            num_heads, 
            feedforward_channels, 
            drop_rate, 
            attn_drop_rate, 
            drop_path_rate, 
            num_fcs, 
            qkv_bias, 
            act_cfg, 
            norm_cfg, 
            batch_first,
        )

        self.attn = MHA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias,
        )

        self.prune_ratio = prune_ratio
        self.tradeoff = tradeoff

    def forward(self, x, cls_attn=None):
        if self.prune_ratio != 1:
            x_patch = x[:, 1:, :]

            B, N, C = x_patch.shape
            N_ = int(N * self.prune_ratio + 0.5)
            indices = torch.argsort(cls_attn, dim=1, descending=True)
            x_patch = torch.cat((x_patch, cls_attn.unsqueeze(-1)), dim=-1)
            x_sorted = easy_gather(x_patch, indices)
            x_patch, cls_attn = x_sorted[:, :, :-1], x_sorted[:, :, -1]

            if self.training:
                x_ = torch.cat((x[:, :1, :], x_patch), dim=1)
            else:
                x[:, 1:, :] = x_patch
                x_ = x
            x = x_[:, :N_ + 1]

            # slow updating
            x, attn = self.attn(self.norm1(x), identity=x)

            # with torch.no_grad():
            if self.training:
                temp_cls_attn = (1 - self.tradeoff) * cls_attn[:, :N_] + self.tradeoff * attn[:, 0, 1:]
                cls_attn = torch.cat((temp_cls_attn, cls_attn[:, N_:]), dim=1)

            else:
                cls_attn[:, :N_] = (1 - self.tradeoff) * cls_attn[:, :N_] + self.tradeoff * attn[:, 0, 1:]

            x = self.ffn(self.norm2(x), identity=x)

            if self.training:
                x = torch.cat((x, x_[N_ + 1:]), dim=1)
            else:
                x_[:, :N_ + 1] = x
                x = x_
            
            indices_inverse = torch.argsort(indices, dim=1)
            x_patch = x[:, 1:, :]
            x_patch = torch.cat((x_patch, cls_attn.unsqueeze(-1)), dim=-1)
            x_inverse = easy_gather(x_patch, indices_inverse)
            x_patch, cls_attn = x_inverse[:, :, :-1], x_inverse[:, :, -1]
            if self.training:
                x = torch.cat((x[:, :1, :], x_patch), dim=1)
            else:
                x[:, 1:, :] = x_patch
        else:
            x, attn = self.attn(self.norm1(x), identity=x)
            if cls_attn == None:
                cls_attn = attn[:, 0, 1:]
            else:
                cls_attn = (1 - self.tradeoff) * cls_attn + self.tradeoff * attn[:, 0, 1:]
            x = self.ffn(self.norm2(x), identity=x)
        return x, cls_attn


@BACKBONES.register_module()
class EvoVisionTransformer(VisionTransformer):
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
        prune_ratio=1.,
        tradeoff=0.5,
        prune_location=3, 
        **kwargs):
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
            **kwargs)
        
        if prune_location >= 0 and prune_location < num_layers:
            if not isinstance(prune_ratio, (list, tuple)):
                prune_ratio = [1.0] * prune_location + [prune_ratio] * (num_layers - prune_location)
            else:
                prune_ratio = [1.0] * num_layers
        if not isinstance(tradeoff, (list, tuple)):
            tradeoff = [tradeoff] * num_layers

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EvoTransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                    prune_ratio=prune_ratio[i], 
                    tradeoff=tradeoff[i],
                )
            )

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

        outs = []
        cls_attn = None
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x, cls_attn = checkpoint(layer, x, cls_attn)
            else:
                x, cls_attn = layer(x, cls_attn)
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
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

