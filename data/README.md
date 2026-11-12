# Dataset placement

The dataset is not included. Place the COCO-format training and validation annotations and images under `data/line_fittings/`, as described in the repository README. Create separate annotation copies with the English class names by running `tools/normalize_coco_categories.py` on the training and validation annotation files. Generate `adjacency_info.pkl` and `spatial_adj.pkl` with `tools/build_priors.py`, and `rsm_stats.json` with `tools/build_rsm_stats.py`, from the normalized training annotations.
