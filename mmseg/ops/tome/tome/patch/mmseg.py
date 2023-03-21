from typing import Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from mmseg.models.backbones.vit import (
    TransformerEncoderLayer,
    VisionTransformer,
)

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

class BipartiteSoftMatching(nn.Module):
    def forward(self, *args, **kwargs):
        return bipartite_soft_matching(*args, **kwargs)

class ToMeTransformerEncoderLayer(TransformerEncoderLayer):
    def apply_tome(self, module, x, **kwarg):
        r = self._tome_info["r"].pop(0)
        if r > 0:
            r = int(x.shape[1] * r)
            # Apply ToMe here
            merge, unmerge = self.bipartite_soft_matching(
                x,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, _ = merge_wavg(merge, x)

        x = module(x, **kwarg)
        if r > 0:
            x = unmerge(x)
        return x

    def forward(self, x):
        x = x + self.apply_tome(self.attn, self.norm1(x), identity=0)
        x = x + self.apply_tome(self.ffn, self.norm2(x), identity=0)
        return x

class ToMeVisionTransformer(VisionTransformer):
    """
    Modifications:
     - Initialize r, token size, and token sources.
    """

    def forward(self, *args, **kwdargs) -> torch.Tensor:
        self._tome_info["r"] = [self.r] * (len(self.layers) * 2) # parse_r(len(self.blocks) * 2, self.r)
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        return super().forward(*args, **kwdargs)

def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, TransformerEncoderLayer):
            module.__class__ = ToMeTransformerEncoderLayer
            module._tome_info = model._tome_info
            module.bipartite_soft_matching = BipartiteSoftMatching()
            module.add_module("bipartite_soft_matching", module.bipartite_soft_matching)

