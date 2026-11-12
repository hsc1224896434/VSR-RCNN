# VSR R-CNN

[English](README.md) | [简体中文](README_zh-CN.md)

**Visual Semantic Reasoning R-CNN** detects transmission-line fittings in UAV images. This repository contains the model components, reference configuration, and training and evaluation tools for MMDetection 2.24.1.

## What is included

- **GA-RPN** generates guided proposals and location predictions.
- **CRM and GRM** use class co-occurrence and geometric priors, respectively, to guide proposal-level reasoning. The priors are built from training annotations.
- **RSM** refines bolt and associated fitting scores using their box relationships and GA-RPN location predictions, then applies Gaussian Soft-NMS. See [RSM implementation](RSM_STATUS.md) for the exact inference rules.

The model code is under [`mmdet_overlay/`](mmdet_overlay/); [`configs/vsr_rcnn_r50_fpn_10class.py`](configs/vsr_rcnn_r50_fpn_10class.py) defines the ten-class reference experiment. The overlay keeps the method-specific code separate from the MMDetection source tree.

## Installation

This overlay targets MMDetection 2.24.1. Install PyTorch, torchvision, and an MMCV-Full 1.x wheel compatible with your CUDA environment first. The accompanying paper reports Python 3.7.13, PyTorch 1.8.1, and CUDA 10.1. From this repository's root, install an editable MMDetection checkout and copy the model modules into it:

```bash
git clone --branch v2.24.1 https://github.com/open-mmlab/mmdetection.git ../mmdetection
python -m pip install -v -e ../mmdetection
python tools/apply_overlay.py --mmdet-root ../mmdetection
python -m pip install -r requirements.txt
```

The overlay installer checks the MMDetection version before copying files.

## Prepare the dataset

The reference configuration expects COCO-format annotations and images in this layout:

```text
data/line_fittings/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Category IDs must be 1–10 in the order defined by the [configuration](configs/vsr_rcnn_r50_fpn_10class.py). The normalizer accepts the reference dataset's source category names or their English equivalents and writes separate copies for use by the configuration:

```bash
python tools/normalize_coco_categories.py \
  --input data/line_fittings/annotations/instances_train2017.json \
  --output data/line_fittings/annotations/instances_train2017_en.json

python tools/normalize_coco_categories.py \
  --input data/line_fittings/annotations/instances_val2017.json \
  --output data/line_fittings/annotations/instances_val2017_en.json
```

Generate the CRM/GRM priors and RSM statistics from **training annotations**:

```bash
python tools/build_priors.py \
  --annotations data/line_fittings/annotations/instances_train2017_en.json \
  --output-dir data/line_fittings

python tools/build_rsm_stats.py \
  --annotations data/line_fittings/annotations/instances_train2017_en.json \
  --output data/line_fittings/rsm_stats.json
```

These commands create `adjacency_info.pkl`, `spatial_adj.pkl`, and `rsm_stats.json`. The reference configuration uses `val2017` as its held-out evaluation split. The inspection images, annotations, and trained checkpoints are not included in this repository.

## Train and evaluate

Run the commands from this repository's root after installation and data preparation:

```bash
python tools/train.py configs/vsr_rcnn_r50_fpn_10class.py \
  --work-dir work_dirs/vsr_rcnn_r50_fpn_10class

python tools/test.py configs/vsr_rcnn_r50_fpn_10class.py \
  path/to/checkpoint.pth --eval bbox
```

Replace `path/to/checkpoint.pth` with a checkpoint from your training run. RSM is enabled in the reference configuration. For an inference comparison using the **same** checkpoint with RSM disabled, add `--cfg-options model.test_cfg.rcnn.rsm.enabled=False` to the test command. Use RSM statistics generated from the training annotations corresponding to that checkpoint.

## Repository layout

| Path | Contents |
| --- | --- |
| [`configs/`](configs/) | Ten-class model, data, and training configuration |
| [`mmdet_overlay/`](mmdet_overlay/) | GA-RPN, reasoning heads, detector, and RSM |
| [`tools/`](tools/) | Installation, annotation normalization, prior generation, training, and evaluation |
| [`RSM_STATUS.md`](RSM_STATUS.md) | RSM pair selection, score update, and inference settings |

## License

The research contributions are released under [Apache License 2.0](LICENSE). See [RIGHTS.md](RIGHTS.md) and [THIRD_PARTY_OPENMMLAB_LICENSE.txt](THIRD_PARTY_OPENMMLAB_LICENSE.txt) for source attribution.
