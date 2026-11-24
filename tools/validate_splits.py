#!/usr/bin/env python3
"""Fail fast when explicit IMPACTSeg split files overlap."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import read_cases, validate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, required=True)
    args = parser.parse_args()
    partitions = {name: read_cases(args.split_dir/f'{name}.txt') for name in ('train', 'validation', 'test')}
    counts = validate(partitions)
    fit_path = args.split_dir/'fit.txt'
    if not fit_path.is_file():
        raise SystemExit(
            f'Missing required split file: {fit_path}. '
            'Regenerate the split with tools/splits.py.'
        )
    fit = read_cases(fit_path)
    expected = partitions['train'] + partitions['validation']
    if fit != expected:
        raise SystemExit('fit.txt must be the ordered union of train.txt and validation.txt.')
    print(json.dumps({'split_dir': str(args.split_dir), 'counts': counts, 'fit_count': len(fit), 'status': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
