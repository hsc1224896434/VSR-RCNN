"""Contract checks for the RSM inference rule."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_rsm_stats import build_rsm_stats

spec = importlib.util.spec_from_file_location(
    'rsm_under_test', ROOT / 'mmdet_overlay' / 'mmdet' / 'core' /
    'post_processing' / 'rsm.py')
rsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rsm)


class RSMContractTest(unittest.TestCase):
    def test_rejection_preserves_boxes_and_multiple_pairs_use_one_gain(self):
        # Bolt 0 has two valid partners; box 3 is too large to pass area rule.
        boxes = torch.tensor([
            [10., 10., 20., 20.],
            [9., 9., 21., 21.],
            [10., 10., 22., 22.],
            [0., 0., 64., 64.],
        ])
        scores = torch.tensor([0.20, 0.40, 0.30, 0.60])
        labels = torch.tensor([2, 1, 1, 1])
        stats = dict(bolt_label=2, eligible_component_labels=[1],
                     mean_r_in=0.89, mean_r_area=2.0)
        # Logit 0 gives center probability 0.5 at every level.
        loc = [torch.zeros(1, 1, 64 // stride, 64 // stride)
               for stride in (4, 8, 16, 32, 64)]
        result = rsm.rescore_detections(
            boxes, scores, labels, loc, (64, 64), (64, 64), stats)
        self.assertAlmostEqual(float(result[0]), 0.45, places=6)
        self.assertAlmostEqual(float(result[1]), 0.55, places=6)
        self.assertAlmostEqual(float(result[2]), 0.45, places=6)
        self.assertAlmostEqual(float(result[3]), 0.60, places=6)
        self.assertEqual(len(result), len(scores))

    def test_training_pair_mean_and_eligible_labels(self):
        from normalize_coco_categories import CLASS_NAMES
        payload = {
            'categories': [{'id': i + 1, 'name': name}
                           for i, name in enumerate(CLASS_NAMES)],
            'images': [{'id': 1, 'width': 100, 'height': 100}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 3,
                 'bbox': [10, 10, 10, 10]},
                {'id': 2, 'image_id': 1, 'category_id': 2,
                 'bbox': [9, 9, 12, 12]},
                {'id': 3, 'image_id': 1, 'category_id': 5,
                 'bbox': [80, 80, 10, 10]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'train.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            stats = build_rsm_stats(path)
        self.assertEqual(stats['pair_count'], 1)
        self.assertEqual(stats['eligible_component_labels'], [1])
        self.assertEqual(stats['mean_r_in'], 1.0)
        self.assertAlmostEqual(stats['mean_r_area'], 1.44)

    def test_box_scale_selects_supervised_location_level(self):
        boxes = torch.tensor([[8., 8., 24., 24.], [0., 0., 64., 64.]])
        maps = [torch.full((1, 1, 64 // stride, 64 // stride), -10.)
                for stride in (4, 8, 16, 32, 64)]
        maps[1].fill_(10.)
        probability = rsm._center_probabilities(
            boxes, maps, (64, 64), (64, 64), 8, (4, 8, 16, 32, 64))
        self.assertLess(float(probability[0]), 0.01)
        self.assertGreater(float(probability[1]), 0.99)


if __name__ == '__main__':
    unittest.main()
