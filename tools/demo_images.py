#!/usr/bin/env python3
"""Run VSR R-CNN on a folder of images, one image at a time."""

import argparse
import time
from pathlib import Path

import torch

from mmdet.apis import inference_detector, init_detector


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_images(source):
    """Return a stable list of supported images from a file or flat folder."""
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    if source.is_dir():
        return sorted(path for path in source.iterdir()
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return []


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='VSR R-CNN config file')
    parser.add_argument('checkpoint', type=Path, help='trained checkpoint')
    parser.add_argument('input', type=Path, help='image file or image folder')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='folder for images with predicted boxes')
    parser.add_argument('--device', default='cuda:0',
                        help='PyTorch inference device (default: cuda:0)')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='score threshold for drawing boxes only')
    parser.add_argument('--max-images', type=int, default=None,
                        help='optional limit for a short demo run')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise SystemExit('Config and checkpoint files must exist.')
    if not 0 <= args.score_thr <= 1:
        raise SystemExit('--score-thr must be between 0 and 1.')
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit('--max-images must be positive.')
    images = find_images(args.input)
    if args.max_images is not None:
        images = images[:args.max_images]
    if not images:
        raise SystemExit('No supported images found in the input path.')
    source_dir = args.input if args.input.is_dir() else args.input.parent
    if args.output_dir.resolve() == source_dir.resolve():
        raise SystemExit('Output folder must differ from the input image folder.')

    model = init_detector(str(args.config), str(args.checkpoint),
                          device=args.device)
    # Old checkpoints can lack class metadata; use the configured class order.
    classes = tuple(model.cfg.get('classes', ()))
    if classes and len(getattr(model, 'CLASSES', ())) != len(classes):
        model.CLASSES = classes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    for index, image in enumerate(images, start=1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = inference_detector(model, str(image))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        destination = args.output_dir / image.name
        model.show_result(str(image), result,
                          score_thr=args.score_thr,
                          out_file=str(destination))
        print(f'[{index}/{len(images)}] {image.name}: '
              f'inference {elapsed:.3f} s -> {destination}')


if __name__ == '__main__':
    main()
