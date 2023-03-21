# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class BCELoss(nn.Module):

    def __init__(self,
                 class_weighted=False):
        super(BCELoss, self).__init__()
        self.mul_label_loss = nn.BCEWithLogitsLoss()

    def forward(self,
                img_pred,
                label,
                **kwargs):
        """Forward function."""
        return self.mul_label_loss(img_pred, label)
