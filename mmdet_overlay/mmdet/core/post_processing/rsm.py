"""VSR R-CNN re-scoring module following SIVP Eqs. (9)–(10).

The module has no trainable parameters; it uses GA-RPN location predictions.
"""

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def load_rsm_stats(path):
    """Load global thresholds produced from training annotations only."""
    with Path(path).open('r', encoding='utf-8') as stream:
        stats = json.load(stream)
    required = ('mean_r_in', 'mean_r_area', 'pair_count',
                'bolt_label', 'eligible_component_labels')
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError('RSM statistics missing: ' + ', '.join(missing))
    if not (0 <= stats['mean_r_in'] <= 1):
        raise ValueError('RSM mean_r_in must be in [0, 1].')
    if stats['mean_r_area'] <= 0 or stats['pair_count'] <= 0:
        raise ValueError('RSM statistics require positive area and pair count.')
    if not stats['eligible_component_labels']:
        raise ValueError('No eligible component labels in RSM statistics.')
    return stats


def _center_probabilities(boxes, loc_preds, img_shape, pad_shape,
                          anchor_scale, strides):
    """Sample the GA-RPN location level assigned to each box scale."""
    if loc_preds is None:
        raise ValueError('RSM requires GA-RPN location predictions.')
    if len(loc_preds) != len(strides):
        raise ValueError('RSM location levels and configured strides differ.')
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))

    height, width = int(img_shape[0]), int(img_shape[1])
    pad_height, pad_width = int(pad_shape[0]), int(pad_shape[1])
    if not (0 < height <= pad_height and 0 < width <= pad_width):
        raise ValueError('Invalid image and padded image shapes for RSM.')

    area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
            (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    scale = torch.sqrt(area.clamp(min=1e-6))
    minimum_anchor = float(anchor_scale * strides[0])
    levels = torch.floor(torch.log2(scale / minimum_anchor) + 0.5)
    levels = levels.clamp(min=0, max=len(strides) - 1).long()
    centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5).floor().long()
    centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5).floor().long()
    centers_x = centers_x.clamp(min=0, max=width - 1)
    centers_y = centers_y.clamp(min=0, max=height - 1)

    probabilities = boxes.new_zeros((boxes.size(0),))
    for level, prediction in enumerate(loc_preds):
        selected = levels == level
        if not selected.any():
            continue
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        if prediction.dim() != 4 or prediction.size(0) != 1 or prediction.size(1) != 1:
            raise ValueError('Each per-image location level must be [1, 1, H, W].')
        # The feature maps include image padding. Resize to that same canvas
        # before reading a center in scaled-image coordinates.
        map_full = F.interpolate(prediction.sigmoid(),
                                 size=(pad_height, pad_width),
                                 mode='bilinear', align_corners=False)[0, 0]
        probabilities[selected] = map_full[
            centers_y[selected], centers_x[selected]]
    return probabilities


def rescore_detections(boxes, scores, labels, loc_preds, img_shape, pad_shape,
                       stats, alpha_bolt=0.5, alpha_other=0.3,
                       anchor_scale=8, strides=(4, 8, 16, 32, 64)):
    """Apply the SIVP overlap AND area rule before Soft-NMS.

    Every original detection is retained. Rejection skips only that pair.
    For a box in several eligible pairs, the largest single increment wins.
    Scores are clipped to one after applying the increment once.
    """
    if boxes.ndim != 2 or boxes.size(1) != 4:
        raise ValueError('RSM boxes must have shape [N, 4].')
    if scores.ndim != 1 or labels.ndim != 1 or not (
            boxes.size(0) == scores.size(0) == labels.size(0)):
        raise ValueError('RSM boxes, scores, and labels have inconsistent lengths.')
    if not (0 <= alpha_bolt <= 1 and 0 <= alpha_other <= 1):
        raise ValueError('RSM score coefficients must be in [0, 1].')
    updated = scores.clone()
    if scores.numel() == 0:
        return updated

    bolt_label = int(stats['bolt_label'])
    bolt_indices = torch.nonzero(labels == bolt_label, as_tuple=False).flatten()
    eligible = torch.zeros_like(labels, dtype=torch.bool)
    for label in stats['eligible_component_labels']:
        eligible |= labels == int(label)
    other_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if bolt_indices.numel() == 0 or other_indices.numel() == 0:
        return updated

    bolt_boxes = boxes[bolt_indices]
    other_boxes = boxes[other_indices]
    left_top = torch.max(bolt_boxes[:, None, :2], other_boxes[None, :, :2])
    right_bottom = torch.min(bolt_boxes[:, None, 2:], other_boxes[None, :, 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]
    bolt_area = ((bolt_boxes[:, 2] - bolt_boxes[:, 0]).clamp(min=1e-6) *
                 (bolt_boxes[:, 3] - bolt_boxes[:, 1]).clamp(min=1e-6))
    other_area = ((other_boxes[:, 2] - other_boxes[:, 0]).clamp(min=0) *
                  (other_boxes[:, 3] - other_boxes[:, 1]).clamp(min=0))
    r_in = intersection / bolt_area[:, None]
    r_area = other_area[None, :] / bolt_area[:, None]
    qualifying = ((intersection > 0) &
                  (r_in > float(stats['mean_r_in'])) &
                  (r_area <= float(stats['mean_r_area'])))
    if not qualifying.any():
        return updated

    center_prob = _center_probabilities(boxes, loc_preds, img_shape, pad_shape,
                                        anchor_scale, strides)
    bolt_gain = alpha_bolt * r_in * center_prob[bolt_indices, None]
    other_gain = alpha_other * r_in * center_prob[None, other_indices]
    zero = torch.zeros_like(r_in)
    bolt_gain = torch.where(qualifying, bolt_gain, zero).max(dim=1)[0]
    other_gain = torch.where(qualifying, other_gain, zero).max(dim=0)[0]
    updated[bolt_indices] = (scores[bolt_indices] + bolt_gain).clamp(max=1.0)
    updated[other_indices] = (scores[other_indices] + other_gain).clamp(max=1.0)
    return updated
