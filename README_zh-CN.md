# VSR R-CNN

[English](README.md) | [简体中文](README_zh-CN.md)

**Visual Semantic Reasoning R-CNN** 用于无人机图像中的输电线路金具检测。本仓库提供基于 MMDetection 2.24.1 的模型模块、参考配置，以及训练和评估工具。

## 代码内容

- **GA-RPN** 生成引导式候选框和位置预测。
- **CRM 与 GRM** 分别利用类别共现先验和几何先验引导候选框之间的关系推理；先验由训练集标注生成。
- **RSM** 根据螺栓与相关金具的框关系及 GA-RPN 位置预测调整得分，再执行高斯 Soft-NMS。配对、得分更新和推理参数见 [RSM 实现说明](RSM_STATUS.md)。

模型代码位于 [`mmdet_overlay/`](mmdet_overlay/)，十类检测任务的参考配置为 [`configs/vsr_rcnn_r50_fpn_10class.py`](configs/vsr_rcnn_r50_fpn_10class.py)。覆盖层将本方法的代码与 MMDetection 原始代码分开保存。

## 复现状态

本仓库是论文方法的参考实现，尚未验证能够重现论文报告的 62.8% mAP。原始的 RSM 自定义后处理算子没有留存；公开的 RSM 按论文式（9）—（10）及 [RSM_STATUS.md](RSM_STATUS.md) 所列执行约定重建。产生论文结果的检查点和私有数据集均未发布。使用者应报告自己训练与评估所得的指标，不应把论文数值当作本仓库运行的预期输出。

## 安装

此覆盖层面向 MMDetection 2.24.1，要求 MMCV-Full 版本为 1.3.17–1.6.0。先安装与本机 CUDA 环境匹配的 PyTorch、torchvision 和 MMCV-Full。论文报告的环境为 Python 3.7.13、PyTorch 1.8.1 和 CUDA 10.1。从本仓库根目录执行：

```bash
git clone --branch v2.24.1 https://github.com/open-mmlab/mmdetection.git ../mmdetection
python -m pip install -v -e ../mmdetection
python tools/apply_overlay.py --mmdet-root ../mmdetection
python -m pip install -r requirements.txt
```

覆盖层安装脚本会检查 MMDetection 版本，再复制模型文件。

## 准备数据

参考配置要求使用以下目录结构的 COCO 格式数据：

```text
data/line_fittings/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

类别 ID 应为 1–10，顺序与[配置文件](configs/vsr_rcnn_r50_fpn_10class.py)一致。类别名称转换脚本接受参考数据集的原始类别名或对应的英文类别名，并生成独立的英文标注副本：

```bash
python tools/normalize_coco_categories.py \
  --input data/line_fittings/annotations/instances_train2017.json \
  --output data/line_fittings/annotations/instances_train2017_en.json

python tools/normalize_coco_categories.py \
  --input data/line_fittings/annotations/instances_val2017.json \
  --output data/line_fittings/annotations/instances_val2017_en.json
```

仅使用**训练集标注**生成 CRM/GRM 先验和 RSM 统计量：

```bash
python tools/build_priors.py \
  --annotations data/line_fittings/annotations/instances_train2017_en.json \
  --output-dir data/line_fittings

python tools/build_rsm_stats.py \
  --annotations data/line_fittings/annotations/instances_train2017_en.json \
  --output data/line_fittings/rsm_stats.json
```

生成的文件为 `adjacency_info.pkl`、`spatial_adj.pkl` 和 `rsm_stats.json`。参考配置将 `val2017` 用作留出的评估集。本仓库不包含巡检图像、标注和训练好的检查点。

## 训练与评估

完成安装和数据准备后，在本仓库根目录运行：

```bash
python tools/train.py configs/vsr_rcnn_r50_fpn_10class.py \
  --work-dir work_dirs/vsr_rcnn_r50_fpn_10class

python tools/test.py configs/vsr_rcnn_r50_fpn_10class.py \
  path/to/checkpoint.pth --eval bbox
```

将 `path/to/checkpoint.pth` 替换为训练得到的检查点。参考配置默认启用 RSM。如需用**同一个**检查点比较关闭 RSM 的推理结果，可在测试命令后添加 `--cfg-options model.test_cfg.rcnn.rsm.enabled=False`。RSM 统计量应由该检查点对应版本的训练集标注生成。

如果只需按图片顺序模拟巡检、保存检测可视化结果，不需要标注文件，可以运行：

```bash
python tools/demo_images.py configs/vsr_rcnn_r50_fpn_10class.py \
  path/to/checkpoint.pth data/line_fittings/val2017 \
  --output-dir outputs/inspection_demo
```

`--score-thr` 只控制画出哪些框；计算 AP 等量化指标仍使用带 COCO 标注的 `tools/test.py`。演示脚本会打印每张图片的推理耗时。

安装 PyTorch 后可运行 RSM 规则检查：

```bash
python -m unittest discover -s tests -v
```

这些检查只覆盖 RSM 行为，不能验证完整训练或论文报告的数值。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`configs/`](configs/) | 十类模型、数据与训练配置 |
| [`mmdet_overlay/`](mmdet_overlay/) | GA-RPN、关系推理模块、检测器与 RSM |
| [`tools/`](tools/) | 安装、标注转换、先验生成、训练、评估与逐图演示脚本 |
| [`RSM_STATUS.md`](RSM_STATUS.md) | RSM 配对、得分更新和推理参数 |

## 许可证

本仓库的研究代码按 [Apache License 2.0](LICENSE) 开源。源代码归属说明见 [RIGHTS.md](RIGHTS.md) 和 [THIRD_PARTY_OPENMMLAB_LICENSE.txt](THIRD_PARTY_OPENMMLAB_LICENSE.txt)。
