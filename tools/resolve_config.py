#!/usr/bin/env python3
"""Resolve explicit IMPACTSeg environment tokens without mutating YAML templates."""
from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from impactseg.labels import FINE_NETWORK_OUTPUTS, MAP_559_NETWORK_OUTPUTS  # noqa: E402

TOKEN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=None)
    parser.add_argument('--splits-root', type=Path, default=None)
    parser.add_argument('--split-name', type=str, default=None)
    parser.add_argument('--dataset-format', type=str, default=None)
    parser.add_argument('--modalities', nargs='+', default=None)
    parser.add_argument('--train-name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--memory-budget', type=str, default=None)
    args = parser.parse_args()
    dataset_root = args.dataset_root or (Path(os.environ['IMPACTSEG_DATASET_ROOT']) if 'IMPACTSEG_DATASET_ROOT' in os.environ else None)
    if dataset_root is None:
        raise SystemExit('Define --dataset-root or IMPACTSEG_DATASET_ROOT.')
    splits_root = args.splits_root or Path(os.environ.get('IMPACTSEG_SPLITS_ROOT', ROOT/'splits'))
    values = dict(os.environ)
    values.update({
        'IMPACTSEG_DATASET_ROOT': str(dataset_root.resolve()),
        'IMPACTSEG_SPLITS_ROOT': str(splits_root.resolve()),
        'IMPACTSEG_FINE_NETWORK_OUTPUTS': str(FINE_NETWORK_OUTPUTS),
        'IMPACTSEG_559_NETWORK_OUTPUTS': str(MAP_559_NETWORK_OUTPUTS),
    })
    raw = args.template.read_text()
    missing: set[str] = set()
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            missing.add(name)
            return match.group(0)
        return values[name]
    rendered = TOKEN.sub(replace, raw)
    if missing:
        raise SystemExit(f'Unresolved configuration token(s): {sorted(missing)}')
    if any((args.split_name, args.dataset_format, args.modalities, args.train_name,
            args.epochs is not None, args.memory_budget is not None)):
        yaml = YAML()
        document = yaml.load(rendered)
        trainer = document['Trainer']
        dataset = trainer['Dataset']
        if args.split_name:
            split_dir = splits_root.resolve() / args.split_name
            dataset['subset'] = str(split_dir / 'fit.txt')
            dataset['validation'] = str(split_dir / 'validation.txt')
        if args.dataset_format or args.modalities:
            if not args.dataset_format or not args.modalities:
                raise SystemExit('--dataset-format and --modalities must be provided together.')
            dataset['dataset_filenames'] = [
                f'{dataset_root.resolve()}/{modality}:a:{args.dataset_format}'
                for modality in args.modalities
            ]
        if args.train_name:
            trainer['train_name'] = args.train_name
        if args.epochs is not None:
            if args.epochs <= 0:
                raise SystemExit('--epochs must be strictly positive.')
            trainer['epochs'] = args.epochs
        if args.memory_budget is not None:
            dataset['memory_budget'] = args.memory_budget
        output = io.StringIO()
        yaml.dump(document, output)
        rendered = output.getvalue()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)


if __name__ == '__main__':
    main()
