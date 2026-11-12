#!/usr/bin/env python3
"""Write a COCO annotation copy with the reference English class names."""

import argparse
import json
from pathlib import Path


SOURCE_TO_ENGLISH = {
    'jyh': 'grading ring',
    'jgj': 'fastener',
    'ls': 'bolt',
    'fzc': 'vibration damper',
    'xj': 'clamp',
    'gb': 'clevis & tongue',
    'gh': 'shackle',
    'gb_hd': 'clevis & tongue of crossarm',
    'lb': 'yoke plate',
    'jyz_fh': 'composite insulator',
}
CLASS_NAMES = tuple(SOURCE_TO_ENGLISH.values())


def normalize_categories(input_path, output_path):
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError('Choose a separate output path to preserve the original file.')

    with input_path.open('r', encoding='utf-8') as stream:
        dataset = json.load(stream)

    categories = dataset.get('categories')
    if not isinstance(categories, list):
        raise ValueError('The COCO file must contain a categories list.')
    categories_by_id = {category['id']: category for category in categories}
    expected_ids = list(range(1, len(CLASS_NAMES) + 1))
    if len(categories_by_id) != len(categories) or sorted(categories_by_id) != expected_ids:
        raise ValueError(
            'Expected the reference 10-class category IDs 1 through 10. '
            'Check the dataset and category order before conversion.')

    current_names = [categories_by_id[index]['name'] for index in expected_ids]
    if all(name in SOURCE_TO_ENGLISH for name in current_names):
        normalized_names = [SOURCE_TO_ENGLISH[name] for name in current_names]
    elif all(name in CLASS_NAMES for name in current_names):
        normalized_names = current_names
    else:
        raise ValueError(
            'Category names do not match the known 10-class source labels. '
            'No categories were converted.')
    if tuple(normalized_names) != CLASS_NAMES:
        raise ValueError(
            'The category IDs do not follow the reference class order. '
            'No categories were converted.')
    for category_id, class_name in zip(expected_ids, normalized_names):
        categories_by_id[category_id]['name'] = class_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(dataset, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return len(dataset.get('images', [])), len(dataset.get('annotations', []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input annotation file not found: {args.input}')

    try:
        image_count, annotation_count = normalize_categories(args.input, args.output)
    except (KeyError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(
        f'Wrote English class names for {image_count} images and '
        f'{annotation_count} annotations to {args.output}')


if __name__ == '__main__':
    main()
