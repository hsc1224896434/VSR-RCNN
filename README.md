# VSR R-CNN: Transmission Line Fitting Detection

Research code for **Visual Semantic Reasoning R-CNN (VSR R-CNN)**, a detector for transmission-line fittings in UAV imagery. It combines a guided-anchoring proposal network (GA-RPN), a connecting relationship module (CRM), a geometric relationship module (GRM), and a re-scoring module (RSM). The repository provides the detector modules, a ten-class reference configuration, tools for constructing training priors, and training and evaluation entry points for MMDetection 2.24.1.

## Method and paper correspondence

The CRM prior follows the conditional co-occurrence relationship in manuscript Eq. (5); the GRM prior follows the geometric descriptor in Eq. (6). The RSM pairs predicted bolts with fitting components using the overlap and relative-area conditions in Eq. (9), updates their scores by Eq. (10) using GA-RPN location probabilities, and applies Gaussian Soft-NMS. [RSM_STATUS.md](RSM_STATUS.md) gives the complete executable rule, including how the code handles multiple qualifying pairs and feature-map sampling.

The reference config uses the manuscript's ten classes, 1333 × 800 input size, 30 epochs, batch size 3, SGD learning rate 0.016, and its stated graph-attention settings. The learning-rate step milestones `[9, 15]` follow the surviving experiment config. The source is provided as an overlay for MMDetection 2.24.1 so that the method-specific files remain easy to inspect.

The inspection dataset and trained weights are not distributed. Numerical reproduction of the paper therefore requires the corresponding data split and checkpoint; this repository does not claim a reproduced benchmark score.

## Software environment

The manuscript reports Python 3.7.13, PyTorch 1.8.1, CUDA 10.1, cuDNN 7.6.5, and Ubuntu 18.04.4. This overlay targets MMDetection v2.24.1. Install a compatible PyTorch, torchvision, and MMCV-Full 1.x combination for your CUDA environment.

From the root of this repository, create the Python environment and install
PyTorch, torchvision, and MMCV-Full wheels matched to the machine's CUDA
runtime. Then install the matching MMDetection source checkout:

```bash
git clone --branch v2.24.1 https://github.com/open-mmlab/mmdetection.git ../mmdetection
python -m pip install -v -e ../mmdetection
```

Install the method modules and remaining dependencies from this repository:

```bash
python tools/apply_overlay.py --mmdet-root ../mmdetection
python -m pip install -r requirements.txt
```

The overlay refuses other MMDetection versions. It copies only the paper-specific modules into the editable checkout.

## Dataset and priors

Place the COCO-format fitting dataset under `data/line_fittings/`:

```text
data/line_fittings/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

The reference annotations use the original category IDs 1–10 in the class order used by the config. `normalize_coco_categories.py` checks the original class labels and ID order, then writes English-named copies without overwriting the input files:

```bash
python tools/normalize_coco_categories.py \
  --input data/line_fittings/annotations/instances_train2017.json \
  --output data/line_fittings/annotations/instances_train2017_en.json

python tools/normalize_coco_categories.py \
  --input data/line_fittings/annotations/instances_val2017.json \
  --output data/line_fittings/annotations/instances_val2017_en.json
```

Generate the CRM and GRM priors and the RSM thresholds using training annotations only:

```bash
python tools/build_priors.py \
  --annotations data/line_fittings/annotations/instances_train2017_en.json \
  --output-dir data/line_fittings

python tools/build_rsm_stats.py \
  --annotations data/line_fittings/annotations/instances_train2017_en.json \
  --output data/line_fittings/rsm_stats.json
```

This writes `adjacency_info.pkl`, `spatial_adj.pkl`, and `rsm_stats.json`. The last file records the annotation SHA-256, threshold means, pair count, and eligible component labels, without copying annotations. The validation/test split is not used to build priors or RSM thresholds. In this reference config, `val2017` is the held-out test split and is evaluated once after training; do not lower the evaluation interval unless a separate validation split is configured.

## Training and evaluation

Run from this repository's root after installing the environment and overlay:

```bash
python tools/train.py configs/vsr_rcnn_r50_fpn_10class.py \
  --work-dir work_dirs/vsr_rcnn_r50_fpn_10class

python tools/test.py configs/vsr_rcnn_r50_fpn_10class.py \
  path/to/checkpoint.pth \
  --eval bbox
```

Replace `path/to/checkpoint.pth` with a checkpoint produced by training.

RSM is enabled by default in the reference config. To compare the same
checkpoint without RSM, use `--cfg-options model.test_cfg.rcnn.rsm.enabled=False`
with `tools/test.py`. Build `rsm_stats.json` from the same training annotation
version used for the evaluated checkpoint. The released implementation keeps
the detector weights unchanged when RSM is toggled.

## Included files and rights

- `mmdet_overlay/`: GA-RPN, CRM/GRM, RoI, detector, and RSM post-processing modules.
- `configs/`: 10-class reference configuration.
- `tools/build_priors.py`: builds class co-occurrence and geometric priors.
- `tools/normalize_coco_categories.py`: creates separate English-named COCO annotation files.
- `tools/build_rsm_stats.py`: computes the SIVP RSM global thresholds from training boxes.
- `tools/train.py`, `tools/test.py`: MMDetection v2.24.1 training and evaluation entry points.
- `RSM_STATUS.md`: exact RSM implementation rules and source correspondence.
- `CODE_AVAILABILITY.md`: concise code availability wording for the manuscript.

The research contributions are released under Apache License 2.0. OpenMMLab-derived files retain their attribution; see [LICENSE](LICENSE), [RIGHTS.md](RIGHTS.md), and [THIRD_PARTY_OPENMMLAB_LICENSE.txt](THIRD_PARTY_OPENMMLAB_LICENSE.txt).
