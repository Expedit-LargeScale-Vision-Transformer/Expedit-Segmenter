# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

# from torchprofile.torchprofile import profile_macs
import flops_counter as m_flops_counter


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    # parser.add_argument(
    #     '-r', '--resolution', type=int, default=-1, help='fixed resolution if set')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.test_cfg.mode = 'whole'
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])
    # import pdb; pdb.set_trace()

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 100
    pure_inf_time = 0
    total_iters = 500

    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):
        data['img'][0] = torch.randn(1, 3, *cfg['crop_size'])
        # data['img'][0] = torch.randn(1, 3, 512, 512)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        # start_time = time.time()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        # elapsed = time.time() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print('Done image [{:3}/ {}], '.format(i+1, total_iters) + 
                      'fps: {:.2f} img / s'.format(fps))

        if (i + 1) == total_iters:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print('Overall fps: {:.2f} img / s'.format(fps))
            # macs = profile_macs(model, (data['img'], data['img_metas']))
            # print('macs : {:.2f} G'.format(macs / 1e9))
            break

    model_counter = m_flops_counter.add_flops_counting_methods(model.module.backbone)
    model_counter.start_flops_count()
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)
    flops_count, params_count = model_counter.compute_average_flops_cost()
    # if print_per_layer_stat:
    # m_flops_counter.print_model_with_flops(
    #     model_counter, flops_count, params_count, ost=sys.stdout, flush=False)
    model_counter.stop_flops_count()
    flops = m_flops_counter.flops_to_string(flops_count)
    params = m_flops_counter.params_to_string(params_count)
    split_line = '=' * 30
    print('{0}\nFlops: {1}\nParams: {2}\n{0}'.format(
        split_line, flops, params))


if __name__ == '__main__':
    main()
