#!/usr/bin/env python3
"""Build CRM and GRM class-level priors from a COCO training annotation file."""

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPECTED_CLASSES = (
    'grading ring',
    'fastener',
    'bolt',
    'vibration damper',
    'clamp',
    'clevis & tongue',
    'shackle',
    'clevis & tongue of crossarm',
    'yoke plate',
    'composite insulator',
)


def load_coco_annotations(annotation_path):
    with annotation_path.open('r', encoding='utf-8') as stream:
        dataset = json.load(stream)

    categories = sorted(dataset['categories'], key=lambda item: item['id'])
    classes = [item['name'] for item in categories]
    category_ids = [item['id'] for item in categories]
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError(
            'Categories must use the reference 10-class English names in '
            'ascending category-id order. Normalize the annotations first.')
    if len(set(category_ids)) != len(category_ids):
        raise ValueError('COCO category IDs must be unique.')
    images = {item['id']: item for item in dataset['images']}
    if len(images) != len(dataset['images']):
        raise ValueError('COCO image IDs must be unique.')
    annotations_by_image = defaultdict(list)

    for annotation in dataset['annotations']:
        if annotation['image_id'] not in images:
            raise ValueError(
                f"Annotation references unknown image {annotation['image_id']}")
        if annotation['category_id'] not in category_ids:
            raise ValueError(
                f"Annotation references unknown category {annotation['category_id']}")
        annotations_by_image[annotation['image_id']].append(annotation)

    return classes, category_ids, images, annotations_by_image


def build_priors(annotation_path):
    classes, category_ids, images, annotations_by_image = load_coco_annotations(
        annotation_path)
    category_to_index = {
        category_id: index for index, category_id in enumerate(category_ids)
    }
    class_count = len(classes)
    if class_count == 0 or not images:
        raise ValueError('The training annotation file must contain images and classes.')

    image_cooccurrence = np.zeros((class_count, class_count), dtype=np.float64)
    object_cooccurrence = np.zeros_like(image_cooccurrence)
    feature_sums = np.zeros((class_count, 5), dtype=np.float64)
    feature_counts = np.zeros(class_count, dtype=np.int64)

    for image_id, image in images.items():
        width, height = image['width'], image['height']
        if width <= 0 or height <= 0:
            raise ValueError(f'Image {image_id} has invalid dimensions.')

        image_annotations = annotations_by_image.get(image_id, [])
        instance_counts = np.zeros(class_count, dtype=np.int64)
        for annotation in image_annotations:
            class_index = category_to_index[annotation['category_id']]
            instance_counts[class_index] += 1

            bbox = annotation['bbox']
            if len(bbox) != 4:
                raise ValueError(f"Annotation {annotation['id']} must have 4 box values.")
            x, y, box_width, box_height = map(float, bbox)
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"Annotation {annotation['id']} has an invalid box.")
            if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                raise ValueError(f"Annotation {annotation['id']} has non-finite box values.")
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            # Match the manuscript's normalized five-dimensional descriptor.
            feature_sums[class_index] += np.asarray([
                center_x / width,
                center_y / height,
                box_width / width,
                box_height / height,
                math.log(box_width / box_height),
            ])
            feature_counts[class_index] += 1

        present = np.flatnonzero(instance_counts)
        image_cooccurrence[present, present] += 1
        for row in present:
            image_cooccurrence[row, present] += 1
            image_cooccurrence[row, row] -= 1

        for row in range(class_count):
            if instance_counts[row] == 0:
                continue
            object_cooccurrence[row, row] += instance_counts[row]
            for column in range(class_count):
                if row != column and instance_counts[column] > 0:
                    object_cooccurrence[row, column] += instance_counts[column]

    missing_classes = [
        classes[index] for index, count in enumerate(feature_counts) if count == 0
    ]
    if missing_classes:
        raise ValueError(
            'Cannot compute geometric priors; no training boxes for: '
            + ', '.join(missing_classes))

    image_presence = np.diag(image_cooccurrence)
    if np.any(image_presence == 0):
        raise ValueError('At least one class has no image-level co-occurrences.')
    image_adjacency = image_cooccurrence / image_presence[:, None]
    image_adjacency_with_prevalence = image_adjacency.copy()
    prevalence = np.diag(image_cooccurrence) / len(images)
    np.fill_diagonal(image_adjacency_with_prevalence, prevalence)

    object_row_max = object_cooccurrence.max(axis=1)
    if np.any(object_row_max == 0):
        raise ValueError('At least one class has no object-level co-occurrences.')
    object_adjacency = object_cooccurrence / object_row_max[:, None]
    np.fill_diagonal(object_adjacency, prevalence)
    object_adjacency[object_adjacency < 0.05] = 0.0001

    geometric_features = feature_sums / feature_counts[:, None]
    norms = np.linalg.norm(geometric_features, axis=1)
    if np.any(norms == 0):
        raise ValueError('A class geometric prototype has zero norm.')
    spatial_adjacency = (geometric_features @ geometric_features.T) / np.outer(
        norms, norms)

    adjacency_info = {
        'classes': classes,
        'coco_cat_id': category_ids,
        'A_img_lvl_eye': image_adjacency,
        'A_img_lvl': image_adjacency_with_prevalence,
        'A_obj_lvl': object_adjacency,
    }
    spatial_info = {
        'classes': classes,
        'spatial_adj': spatial_adjacency,
    }
    return adjacency_info, spatial_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    if not args.annotations.is_file():
        parser.error(f'Annotation file not found: {args.annotations}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjacency_info, spatial_info = build_priors(args.annotations)

    outputs = {
        'adjacency_info.pkl': adjacency_info,
        'spatial_adj.pkl': spatial_info,
    }
    for filename, payload in outputs.items():
        output_path = args.output_dir / filename
        with output_path.open('wb') as stream:
            # Python 3.7 (the manuscript runtime) cannot read protocol 5.
            pickle.dump(payload, stream, protocol=4)
        print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
