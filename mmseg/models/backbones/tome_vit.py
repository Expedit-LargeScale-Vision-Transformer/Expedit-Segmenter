from .vit import VisionTransformer
from tome.patch.mmseg import apply_patch as apply_tome
from ..builder import BACKBONES

@BACKBONES.register_module()
class ToMeViT(VisionTransformer):
    def __init__(self, tome_r, **kwargs):
        super().__init__(**kwargs)
        apply_tome(self)
        self.r = tome_r