<!-- <div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>
<br /> -->

<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

Documentation: https://mmsegmentation.readthedocs.io/ -->

# Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning

<!-- English | [简体中文](README_zh-CN.md) -->

<!-- [[Paper]](https://arxiv.org/abs/2210.01035) -->

## Introduction

This is the official implementation of the paper "[Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning](https://arxiv.org/abs/2210.01035)" on [Segmenter](https://arxiv.org/abs/2105.05633). The codebase is [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). 

We will also implement several token-reduction methods on Segmenter.

+ [EViT](https://github.com/youweiliang/evit)
+ [ACT](https://github.com/gaopengcuhk/SMCA-DETR/tree/main/Adaptive_Cluster_Transformer)
+ [ToMe](https://github.com/facebookresearch/ToMe)


## Results 

### ADE20K

The results is evaluated using the [official weights](https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_large_mask_640/checkpoint.pth) of Segmenter.

#### Ours

| Method           | Backbone | $\alpha$ | h $\times$ w   | GFLOPs | FPS   | Throughput (im/s) | mIoU  |
| ---------------- | -------- | -------- | -------------- | ------ | ----- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -        | 40 $\times$ 40 | 658.98 | 6.55  | 6.93              | 51.82 |
| Segmenter + Ours | ViT-L/16 | 16       | 28 $\times$ 28 | 529.81 | 7.92  | 8.38              | 51.93 |
| Segmenter + Ours | ViT-L/16 | 10       | 28 $\times$ 28 | 443.84 | 9.51  | 10.13             | 51.56 |
| Segmenter + Ours | ViT-L/16 | 8        | 20 $\times$ 20 | 309.4  | 13.51 | 15.34             | 47.96 |

#### EViT

| Method           | Backbone | keep rate | GFLOPs | FPS   | Throughput (im/s) | mIoU  |
| ---------------- | -------- | --------- | ------ | ----- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -         | 658.98 | 6.55  | 6.93              | 51.82 |
| Segmenter + EViT | ViT-L/16 | 0.9       | 572.01 | 7.58  | 8.22              | 51.52 |
| Segmenter + EViT | ViT-L/16 | 0.8       | 500.19 | 8.50  | 9.22              | 50.37 |
| Segmenter + EViT | ViT-L/16 | 0.5       | 351.83 | 12.03 | 12.84             | 38.89 |

#### ToMe

| Method           | Backbone | r%   | GFLOPs | FPS   | Throughput (im/s) | mIoU  |
| ---------------- | -------- | ---- | ------ | ----- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -    | 658.98 | 6.55  | 6.93              | 51.82 |
| Segmenter + ToMe | ViT-L/16 | 20%  | 516.23 | 6.97  | 8.02              | 51.66 |
| Segmenter + ToMe | ViT-L/16 | 30%  | 448.70 | 8.31  | 9.29              | 50.96 |
| Segmenter + ToMe | ViT-L/16 | 50%  | 321.29 | 10.75 | 12.08             | 47.12 |

#### ACT

| Method          | Backbone | #query-hashes | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| --------------- | -------- | ------------- | ------ | ---- | ----------------- | ----- |
| Segmenter       | ViT-L/16 | -             | 658.98 | 6.55 | 6.93              | 51.82 |
| Segmenter + ACT | ViT-L/16 | 32            | 611.06 | 6.01 | 8.39              | 51.69 |
| Segmenter + ACT | ViT-L/16 | 24            | 545.18 | 6.16 | 8.71              | 51.24 |
| Segmenter + ACT | ViT-L/16 | 16            | 533.52 | 6.33 | 8.42              | 48.03 |

### Cityscapes

The results is evaluated using the [official weights](https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/cityscapes/seg_large_mask/checkpoint.pth) of Segmenter.

#### Ours

| Method           | Backbone | $\alpha$ | h $\times$ w   | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| ---------------- | -------- | -------- | -------------- | ------ | ---- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -        | 48 $\times$ 48 | 995.6  | 4.20 | 4.29              | 79.14 |
| Segmenter + Ours | ViT-L/16 | 18       | 32 $\times$ 32 | 840.9  | 4.82 | 5.13              | 78.82 |
| Segmenter + Ours | ViT-L/16 | 12       | 32 $\times$ 32 | 691.0  | 5.89 | 6.28              | 78.38 |
| Segmenter + Ours | ViT-L/16 | 10       | 24 $\times$ 24 | 529.64 | 8.02 | 8.58              | 76.20 |

#### EViT

| Method           | Backbone | keep rate | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| ---------------- | -------- | --------- | ------ | ---- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -         | 995.6  | 4.20 | 4.29              | 79.14 |
| Segmenter + EViT | ViT-L/16 | 0.9       | 822.7  | 5.27 | 5.40              | 79.03 |
| Segmenter + EViT | ViT-L/16 | 0.8       | 707.2  | 5.96 | 6.31              | 78.49 |
| Segmenter + EViT | ViT-L/16 | 0.5       | 506.2  | 8.68 | 8.87              | 68.14 |

#### ToMe

| Method           | Backbone | r%   | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| ---------------- | -------- | ---- | ------ | ---- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -    | 995.6  | 4.20 | 4.29              | 79.14 |
| Segmenter + ToMe | ViT-L/16 | 20%  | 760.8  | 5.20 | 5.41              | 78.37 |
| Segmenter + ToMe | ViT-L/16 | 30%  | 651.5  | 5.50 | 5.93              | 77.81 |
| Segmenter + ToMe | ViT-L/16 | 50%  | 448.5  | 7.84 | 8.50              | 71.23 |

#### ACT

| Method          | Backbone | #query-hashes | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| --------------- | -------- | ------------- | ------ | ---- | ----------------- | ----- |
| Segmenter       | ViT-L/16 | -             | 995.6  | 4.20 | 4.29              | 79.14 |
| Segmenter + ACT | ViT-L/16 | 32            | 906.26 | 4.76 | 6.09              | 79.00 |
| Segmenter + ACT | ViT-L/16 | 24            | 742.7  | 4.49 | 6.03              | 78.71 |
| Segmenter + ACT | ViT-L/16 | 16            | 730.44 | 5.32 | 6.21              | 75.42 |

### Pascal Context

The results is evaluated using the [official weights](https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/pascal_context/seg_large_mask/checkpoint.pth) of Segmenter.

#### Ours

| Method           | Backbone | $\alpha$ | h $\times$ w   | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| ---------------- | -------- | -------- | -------------- | ------ | ---- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -        | 30 $\times$ 30 | 338.7  | 14.7 | 15.1              | 58.07 |
| Segmenter + Ours | ViT-L/16 | 14       | 20 $\times$ 20 | 251.2  | 18.2 | 19.7              | 58.27 |
| Segmenter + Ours | ViT-L/16 | 12       | 15 $\times$ 15 | 201.3  | 21.6 | 26.2              | 57.85 |
| Segmenter + Ours | ViT-L/16 | 8        | 15 $\times$ 15 | 161.0  | 25.0 | 31.3              | 55.08 |

#### EViT

| Method           | Backbone | keep rate | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| ---------------- | -------- | --------- | ------ | ---- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -         | 338.7  | 14.7 | 15.07             | 58.07 |
| Segmenter + EViT | ViT-L/16 | 0.9       | 271.7  | 16.0 | 17.59             | 57.94 |
| Segmenter + EViT | ViT-L/16 | 0.8       | 261.0  | 17.7 | 19.88             | 56.99 |
| Segmenter + EViT | ViT-L/16 | 0.5       | 184.4  | 23.5 | 28.02             | 48.57 |

#### ToMe

| Method           | Backbone | r%   | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| ---------------- | -------- | ---- | ------ | ---- | ----------------- | ----- |
| Segmenter        | ViT-L/16 | -    | 338.7  | 14.7 | 15.07             | 58.07 |
| Segmenter + ToMe | ViT-L/16 | 20%  | 269.8  | 13.2 | 17.20             | 57.67 |
| Segmenter + ToMe | ViT-L/16 | 30%  | 236.5  | 12.8 | 20.20             | 57.24 |
| Segmenter + ToMe | ViT-L/16 | 50%  | 172.4  | 14.2 | 25.80             | 54.25 |

#### ACT

| Method          | Backbone | #query-hashes | GFLOPs | FPS  | Throughput (im/s) | mIoU  |
| --------------- | -------- | ------------- | ------ | ---- | ----------------- | ----- |
| Segmenter       | ViT-L/16 | -             | 338.7  | 14.7 | 15.07             | 58.07 |
| Segmenter + ACT | ViT-L/16 | 32            | 306.7  | 11.1 | 16.32             | 58.04 |
| Segmenter + ACT | ViT-L/16 | 24            | 299.0  | 11.7 | 16.36             | 57.88 |
| Segmenter + ACT | ViT-L/16 | 16            | 298.3  | 11.9 | 15.98             | 56.08 |

## Installation

The code was tested with Python 3.8, PyTorch 1.10.1. Please use the following command set up dependencies.

```
pip install -r requirements.txt
```

Please refer to [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Download the official checkpoint, and use [prepocess_ckpt.py](tools/prepocess_ckpt.py) to prepocess checkpoints. 

```bash
python tools/prepocess_ckpt.py /path/to/checkpoints
```

If you want to evaluate our methods, use the command as follows.

```bash
bash tools/dist_test.sh /path/to/configs /path/to/checkpoints --eval "mIoU" 
```

If you want to compute the latency and cost of the model, use the command as follows.

```bash
python tools/get_fps_flops.py /path/to/configs
```

## Citation

If you find this project useful in your research, please consider cite:

```latex
@article{liang2022expediting,
	author    = {Liang, Weicong and Yuan, Yuhui and Ding, Henghui and Luo, Xiao and Lin, Weihong and Jia, Ding and Zhang, Zheng and Zhang, Chao and Hu, Han},
	title     = {Expediting large-scale vision transformer for dense prediction without fine-tuning},
	journal   = {arXiv preprint arXiv:2210.01035},
	year      = {2022},
}
```

```
@article{strudel2021,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).