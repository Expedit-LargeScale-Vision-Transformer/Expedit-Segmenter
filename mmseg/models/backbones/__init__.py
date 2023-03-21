# Copyright (c) OpenMMLab. All rights reserved.
from .swin import SwinTransformer
from .vit import VisionTransformer
from .hourglass_vit import HourglassVisionTransformer
from .vit_token_learner import TokenLearnerVisionTransformer
from .evit import EViT
from .tome_vit import ToMeViT
from .dynamic_vit import DynamicViT
from .vit_act import VisionTransformerACT
from .vit_smyrf import VisionTransformerSMYRF
from .vit_with_patch_merger import ViTWithPatchMerger
from .vit_ats import VisionTransformerATS
from .evo_vit import EvoVisionTransformer

__all__ = [
    "VisionTransformer",
    "SwinTransformer",
    "HourglassVisionTransformer",
    "TokenLearnerVisionTransformer",
    "EViT",
    "ToMeViT",
    "DynamicViT",
    "VisionTransformerACT",
    "VisionTransformerSMYRF",
    "ViTWithPatchMerger",
    "VisionTransformerATS",
    "EvoVisionTransformer",
]
