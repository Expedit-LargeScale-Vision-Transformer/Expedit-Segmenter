# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32
import math
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..losses.accuracy import accuracy, img_recall, pixel_recall


@HEADS.register_module()
class UPerNNCEHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, topk_cls, img_loss_weight, pool_scales=(1, 2, 3, 6), use_separable=False, loss_img=dict(type='BCELoss'), **kwargs):
        super(UPerNNCEHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.topk_cls = topk_cls
        self.loss_img = build_loss(loss_img)
        self.img_loss_weight = img_loss_weight

        self.img_cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=self.in_channels[-1],
                out_channels=self.in_channels[-1],
                kernel_size=1,
                stride=1),
            nn.Conv2d(self.in_channels[-1], self.num_classes, kernel_size=1),
        )

        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        conv_builder = DepthwiseSeparableConvModule if use_separable else ConvModule
        self.bottleneck = conv_builder(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = conv_builder(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = conv_builder(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    @staticmethod
    def _get_batch_hist_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch, H, W = target.shape
        tvect = target.new_zeros((batch, nclass))
        for i in range(batch):
            hist = torch.histc(
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1
            )
            tvect[i] = hist
        return tvect

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        img_pred = self.img_cls_head(inputs[-1])
        topk_index = torch.argsort(img_pred, dim=1, descending=True)[
            :, : self.topk_cls
        ].squeeze(-1).squeeze(-1)  # [B, topk]

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)

        full_output = self.cls_seg(output)
        B, C, H, W = full_output.shape

        topk_index_tmp = (
                    topk_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
                )

        output = torch.gather(full_output, 1, topk_index_tmp)

        if self.training:
            return img_pred, topk_index, output

        else:
            full_output = output.new_ones((B, self.num_classes, H, W)) * -100
            full_output.scatter_(1, topk_index_tmp, output)
            return full_output

    @force_fp32(apply_to=('outputs', ))
    def losses(self, outputs, seg_label):
        """Compute segmentation loss."""
        hist = self._get_batch_hist_vector(seg_label.squeeze(1), self.num_classes)
        img_pred, topk_index, pixel_pred = outputs
        loss = dict()
        loss['loss_img'] = self.loss_img(
            img_pred.squeeze(-1).squeeze(-1),
            (hist > 0).float()) * self.img_loss_weight
        seg_logit = resize(
            input=pixel_pred,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        topk_vector = (
            topk_index.unsqueeze(-1).unsqueeze(-1)
            == seg_label.repeat(1, self.topk_cls, 1, 1)
        ).long()
        topk_max_prob, topk_label = torch.max(topk_vector, dim=1)
        topk_label[topk_max_prob == 0] = self.ignore_index
        topk_label[seg_label.squeeze(1) == self.ignore_index] = self.ignore_index

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, topk_label)
        else:
            seg_weight = None
        topk_label = topk_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            topk_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, topk_label)
        loss['acc_pix'] = pixel_recall(hist, topk_index)
        loss['acc_img'] = img_recall(hist, topk_index)
        return loss
