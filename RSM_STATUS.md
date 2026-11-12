# Re-Scoring Module (RSM): implementation details

The RSM code implements the bolt–component relation and score update in the
SIVP manuscript, Eqs. (9)–(10). Its inputs are predicted boxes, class scores,
class labels, and GA-RPN location maps. The main implementation is
[`rsm.py`](mmdet_overlay/mmdet/core/post_processing/rsm.py); it is called by
[`hk_center_bbox_nms.py`](mmdet_overlay/mmdet/core/post_processing/hk_center_bbox_nms.py).

## Pair selection

For a predicted bolt box `b` and a predicted component box `o`, the code uses

```text
R_in(b,o)   = area(b ∩ o) / area(b)
R_area(b,o) = area(o) / area(b)
qualifies  = area(b ∩ o) > 0
             and R_in > mean(R_in)
             and R_area <= mean(R_area)
```

The two means are calculated once from same-image, overlapping bolt–component
annotation pairs in the **training split** by `tools/build_rsm_stats.py`. The
statistics file records the annotation SHA-256, the thresholds, pair count,
and component categories observed in those pairs. Only those categories are
eligible for pair selection at inference.

## Score update and location map

For every qualifying pair, Eq. (10) supplies a candidate increment:

```text
bolt increment      = 0.5 × R_in × P_b
component increment = 0.3 × R_in × P_o
```

`P_b` and `P_o` are sigmoid probabilities from the GA-RPN location branch at
the respective box centers. The feature-pyramid level is selected by the box
scale rule used for GA-RPN location supervision, and its location map is
bilinearly resized to the padded image before sampling.

A box may belong to several qualifying pairs. The implementation applies its
largest single increment, then clips the updated score to `[0, 1]`. A pair
that fails Eq. (9c) receives no increment; its original detections remain
available to post-processing. The pre-NMS score threshold is `0.001`.
Gaussian Soft-NMS then uses `sigma=0.5` and retains at most 100 detections per
image in the reference configuration.

## Source correspondence

The surviving detector source passes GA-RPN location predictions through the
RoI head to a custom center-NMS call. This repository provides the RSM scoring
step directly in `rsm.py` and uses MMCV Soft-NMS for suppression. The
manuscript does not prescribe every software convention for multiple pairs,
eligible-category selection, clipping, and pyramid-level sampling; the
choices above make those cases explicit and deterministic.

The confidential dataset and result-producing checkpoint are not included.
The manuscript's numerical results therefore require their original
experimental materials for independent reproduction.
