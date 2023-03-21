import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmseg.models.utils import get_aspect_ratio, reshape_as_aspect_ratio
from ..builder import BACKBONES
from ...ops.reconstruction import TokenReconstructionBlock
from .vit import VisionTransformer

class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        # args:
        #       x: torch.Tensor, shape of [B, L, C]
        # return:
        #       outputs: torch.Tensor, shape of [B, num_tokens_out, C]
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return torch.matmul(attn, x)

@BACKBONES.register_module()
class ViTWithPatchMerger(VisionTransformer):
    def __init__(self, 
        embed_dims=768, 
        num_tokens_patch_merger=16, 
        patch_merger_loc=11,
        unpool_cfg=None,
        frozen_stages=-1, 
        **kwargs):
        super(ViTWithPatchMerger, self).__init__(embed_dims=embed_dims, **kwargs)

        self.patch_merger_loc = patch_merger_loc
        self.patch_merger = PatchMerger(dim = embed_dims, num_tokens_out = num_tokens_patch_merger)
        self.token_reconstruction_layer = TokenReconstructionBlock(embed_dim=embed_dims, cfg=unpool_cfg)

        self._freeze_stages(frozen_stages)

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
        token_reconstruction_layer = self.token_reconstruction_layer.derive_unpooler()
        outs = []
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            if i == self.patch_merger_loc:
                if self.with_cls_token:
                    cls_tokens = x[:, 0:1]
                    x = x[:, 1:]
                token_reconstruction_layer.update_state(feat_before_pooling=x)
                x = self.patch_merger(x)
                token_reconstruction_layer.update_state(feat_after_pooling=x)
                if self.with_cls_token:
                    x = torch.cat([cls_tokens, x], dim=1)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                out, reshaped = token_reconstruction_layer.call(out)
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)
        

        return tuple(outs)

    def _freeze_stages(self, frozen_stages=-1):
        """Freeze stages param and norm stats."""
        if frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.norm1, self.patch_embed]:
                for param in m.parameters():
                    param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(len(self.layers)):
            if i + 1 <= frozen_stages:
                self.layers[i].eval()
                for param in self.layers[i].parameters():
                    param.requires_grad = False
            else:
                break
        
        if frozen_stages == len(self.layers) + 1:
            self.patch_merger.eval()
            for param in self.patch_merger.parameters():
                    param.requires_grad = False
            