import torch
from torch.utils.checkpoint import checkpoint

from mmcv.runner import ModuleList
from mmseg.models.utils import get_aspect_ratio, reshape_as_aspect_ratio
from ..builder import BACKBONES
from ...ops.reconstruction import TokenReconstructionBlock

from ...ops.cluster import TokenClusteringBlock

from .vit import TransformerEncoderLayer, VisionTransformer


@BACKBONES.register_module()
class HourglassVisionTransformer(VisionTransformer):
    """
    Just for test, fixed cluster layer and cluster number
    """

    def __init__(
        self,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        norm_cfg=dict(type="LN"),
        act_cfg=dict(type="GELU"),
        cluster_cfg=None,
        reconstruction_cfg=None,
        embed_dims=768,
        **kwds,
    ):
        super().__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwds,
        )

        self.embed_dims = embed_dims

        self.build_token_clustering_block(cluster_cfg)
        self.build_token_reconstruction_block(reconstruction_cfg)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
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
                )
            )


    def build_token_clustering_block(self, cfg: dict):
        assert cfg is not None
        self.token_clustering_layer = TokenClusteringBlock(**cfg.get("stage"))
        self.clustering_location = cfg.get("location")

    def build_token_reconstruction_block(self, cfg: dict):
        assert cfg is not None
        self.reconstruction_location = cfg.get("location", -1)
        self.token_reconstruction_layer = TokenReconstructionBlock(cfg=cfg)

    def cluster(self, x, reconstructer):
        reconstructer.update_state(feat_before_pooling=x[:, 1:])
        cls_tokens = x[:, 0:1]
        x = reshape_as_aspect_ratio(x[:, 1:], reconstructer.aspect_ratio)
        x_cluster, hard_labels = self.token_clustering_layer(x)
        self.hard_labels = hard_labels
        reconstructer.update_state(feat_after_pooling=x_cluster)
        x_cluster = torch.cat((cls_tokens, x_cluster), dim=1)
        return x_cluster

    def reconstruct(self, x, reconstructer, disabled=True):
        if reconstructer.org_num_features == x.shape[1] or disabled:
            x = reshape_as_aspect_ratio(x, reconstructer.aspect_ratio)
            return x
        reconstructer.update_state(used=True)
        x, reshaped = reconstructer.call(x)
        if not reshaped:
            x = reshape_as_aspect_ratio(x, reconstructer.aspect_ratio)
        return x

    def forward(self, inputs):
        # embed input
        batch_size = inputs.shape[0]
        x = self.patch_embed(inputs)
        hw_shape = (self.patch_embed.DH, self.patch_embed.DW)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        if not self.with_cls_token:
            x = x[:, 1:]

        # prepare for passing transformer
        aspect_ratio = get_aspect_ratio(*hw_shape)
        reconstructer = self.token_reconstruction_layer.derive_unpooler()
        reconstructer.aspect_ratio = aspect_ratio
        reconstructer.hw_shape = hw_shape
        reconstructer.org_num_features = x.shape[1] - 1
        reconstructer.update_state(used=False)
        self.reconstructer = reconstructer

        outs = []
        for i, layer in enumerate(self.layers):
            x = checkpoint(layer, x) if self.use_checkpoint else layer(x)

            if i == self.clustering_location:
                x = self.cluster(x, reconstructer)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            if i == self.reconstruction_location and reconstructer.org_num_features != x.shape[1]:
                out = x[:, 1:] if self.with_cls_token else x
                out, reshaped = reconstructer.call(out)
                reconstructer.update_state(used=True)
                x = torch.cat([x[:, :1], out], dim=1)

            if i in self.out_indices:
                out = x[:, 1:] if self.with_cls_token else x
                out = self.reconstruct(out, reconstructer, disabled=True)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)
                
        return tuple(outs)

