#!/usr/bin/env python3
"""Copy the VSR-RCNN method modules into an MMDetection 2.24.1 checkout."""

import argparse
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mmdet-root',
        required=True,
        type=Path,
        help='Path to the root of an MMDetection v2.24.1 source checkout.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    overlay_root = repo_root / 'mmdet_overlay' / 'mmdet'
    target_root = args.mmdet_root.resolve()
    version_file = target_root / 'mmdet' / 'version.py'
    if not version_file.is_file():
        parser.error(f'No MMDetection source package found under {target_root}')

    version_text = version_file.read_text(encoding='utf-8')
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", version_text)
    if not match or match.group(1) != '2.24.1':
        parser.error('This overlay supports MMDetection 2.24.1 only.')

    copied = 0
    for source in overlay_root.rglob('*.py'):
        relative_path = source.relative_to(overlay_root)
        destination = target_root / 'mmdet' / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    print(f'Installed {copied} method files into {target_root / "mmdet"}.')
    print('An editable MMDetection install will use the copied files directly.')


if __name__ == '__main__':
    main()
