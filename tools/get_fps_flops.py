# Copyright (c) OpenMMLab. All rights reserved.
import sys
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import argparse

from mmcv import Config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from flops_counter import get_model_complexity_info

from mmseg.models import build_segmentor

from tome.patch.mmseg import apply_patch as apply_tome

default_shapes = {
    'ade': [640, 640],
    'city': [768, 768],
    'pascal': [480, 480],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--load-pretrain', action='store_true')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=None,
        help='input image size')
    parser.add_argument('--tome_r', type=float, default=0.)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.shape is None:
        for k in ['ade', 'city', 'pascal']:
            if k in args.config:
                args.shape = default_shapes[k]
                break

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if not args.load_pretrain:
        cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # apply_tome(model.backbone)
    # model.backbone.r = args.tome_r
    if args.load_pretrain:
        model.init_weights()
        
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # parameters_to_prune = []
    # for i in range(24):
    #     parameters_to_prune.append((model.backbone.layers[i].ffn.layers[0][0], 'weight'))
    #     parameters_to_prune.append((model.backbone.layers[i].ffn.layers[1], 'weight'))
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.3,
    # )
    model = model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    num_warmup = 50
    pure_inf_time = 0
    total_iters = 200
    batch_size = args.batch_size
    for i in range(total_iters):
        sample = torch.ones(()).new_empty(
            (batch_size, *input_shape),
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) * batch_size / pure_inf_time
                print('Done image [{:3}/ {}], '.format(i+1, total_iters) + 
                      'fps: {:.2f} img / s'.format(fps))

        if (i + 1) == total_iters:
            fps = (total_iters - num_warmup) * batch_size / pure_inf_time
            print('Overall fps: {:.2f} img / s'.format(fps))
            break

    with torch.no_grad():
        flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=True)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
