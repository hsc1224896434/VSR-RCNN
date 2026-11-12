# RSM implementation

This module reconstructs the manuscript rule because the historical custom
post-processing operator was not preserved. Its numerical equivalence to the
paper's reported experiment has not been established.

RSM refines detection scores before Gaussian Soft-NMS. Its inputs are predicted
boxes, class scores, class labels, and GA-RPN location maps. Pair selection and
score updates follow Eqs. (9)–(10) of the accompanying paper. The implementation is
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
that fails the selection rule receives no increment; its original detections
remain available to post-processing.

## Inference order and configuration

The detector passes GA-RPN location predictions to the RoI head and then to
`multiclass_center_nms`. Predictions with an original score at or below
`0.001` are removed before re-scoring. RSM updates the remaining scores, and
MMCV's Gaussian Soft-NMS performs class-wise suppression with `sigma=0.5` and
`min_score=0.05`. The [reference configuration](configs/vsr_rcnn_r50_fpn_10class.py)
retains at most 100 detections per image. The multiple-pair rule, category
eligibility, clipping, and pyramid-level sampling above specify the behavior
of this implementation.

The threshold statistics are data dependent. Generate `rsm_stats.json` from
the training annotations used for the corresponding checkpoint before inference.
