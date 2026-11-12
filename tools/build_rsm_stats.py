#!/usr/bin/env python3
"""Build global RSM thresholds from COCO training boxes only."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from normalize_coco_categories import CLASS_NAMES


def build_rsm_stats(annotation_path):
    with annotation_path.open('r', encoding='utf-8') as stream:
        dataset = json.load(stream)
    categories = sorted(dataset['categories'], key=lambda item: item['id'])
    classes = [item['name'] for item in categories]
    category_ids = [item['id'] for item in categories]
    if tuple(classes) != CLASS_NAMES or category_ids != list(range(1, 11)):
        raise ValueError('Normalize the reference 10-class categories first.')
    images = {item['id']: item for item in dataset['images']}
    if len(images) != len(dataset['images']):
        raise ValueError('Image IDs must be unique.')
    by_image = defaultdict(list)
    for annotation in dataset['annotations']:
        if annotation['image_id'] not in images:
            raise ValueError('Annotation references a missing image.')
        if annotation['category_id'] not in category_ids:
            raise ValueError('Annotation references an unknown category.')
        by_image[annotation['image_id']].append(annotation)
    bolt_label = classes.index('bolt')
    bolt_category_id = category_ids[bolt_label]
    category_to_label = {category_id: index for index, category_id in
                         enumerate(category_ids)}
    r_in_values = []
    r_area_values = []
    component_counts = defaultdict(int)

    for annotations in by_image.values():
        bolts = [item for item in annotations
                 if item['category_id'] == bolt_category_id]
        others = [item for item in annotations
                  if item['category_id'] != bolt_category_id]
        for bolt in bolts:
            bx, by, bw, bh = map(float, bolt['bbox'])
            bolt_area = bw * bh
            if bolt_area <= 0:
                raise ValueError('Bolt annotation has nonpositive area.')
            for component in others:
                ox, oy, ow, oh = map(float, component['bbox'])
                if ow <= 0 or oh <= 0:
                    raise ValueError('Component annotation has nonpositive area.')
                intersection_width = max(
                    0.0, min(bx + bw, ox + ow) - max(bx, ox))
                intersection_height = max(
                    0.0, min(by + bh, oy + oh) - max(by, oy))
                intersection = intersection_width * intersection_height
                if intersection == 0:
                    continue
                r_in_values.append(intersection / bolt_area)
                r_area_values.append((ow * oh) / bolt_area)
                component_counts[category_to_label[component['category_id']]] += 1

    if not r_in_values:
        raise ValueError('No overlapping bolt/component training pairs found.')
    if not all(math.isfinite(value) for value in r_in_values + r_area_values):
        raise ValueError('RSM ratios contain a non-finite value.')
    count = len(r_in_values)
    return {
        'schema_version': 1,
        'classes': classes,
        'category_ids': category_ids,
        'bolt_label': bolt_label,
        'eligible_component_labels': sorted(component_counts),
        'component_pair_counts': {
            str(label): component_counts[label] for label in sorted(component_counts)},
        'pair_count': count,
        'mean_r_in': math.fsum(r_in_values) / count,
        'mean_r_area': math.fsum(r_area_values) / count,
        'pair_definition': 'all same-image, different-class bolt/non-bolt '
                           'annotation boxes with positive intersection',
        'source_annotation_sha256': hashlib.sha256(
            annotation_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.annotations.is_file():
        parser.error('Training annotations not found: ' + str(args.annotations))
    stats = build_rsm_stats(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        json.dump(stats, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print('Wrote {} ({} overlapping pairs)'.format(
        args.output, stats['pair_count']))


if __name__ == '__main__':
    main()
