#!/usr/bin/env python3
"""Materialise and validate explicit v2 splits from the preserved legacy fold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_cases(path: Path) -> list[str]:
    cases = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(cases) != len(set(cases)):
        duplicates = sorted({case for case in cases if cases.count(case) > 1})
        raise ValueError(f"Duplicate case IDs in {path}: {duplicates[:10]}")
    return cases


def digest(cases: list[str]) -> str:
    return hashlib.sha256(("\n".join(cases) + "\n").encode()).hexdigest()


def validate(partitions: dict[str, list[str]]) -> dict[str, int]:
    sets = {name: set(cases) for name, cases in partitions.items()}
    result: dict[str, int] = {name: len(values) for name, values in sets.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"Data leakage: {left} ∩ {right} has {len(overlap)} cases: {sorted(overlap)[:10]}")
    return result


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True, help='Folder of the original fold lists.')
    parser.add_argument('--output-root', type=Path, default=repo/'splits')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--name', type=str, default=None, help='Output directory name (default: fold<N>).')
    parser.add_argument('--exclude-prefix', action='append', default=[], help='Case-ID prefix excluded from every partition, recorded in the manifest.')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    prefix = f'ct_cbct_mri_fold{args.fold}'
    source = {
        'train': args.legacy_root/f'{prefix}_train_raw.txt',
        'validation': args.legacy_root/f'{prefix}_val.txt',
        'test': args.legacy_root/f'{prefix}_test.txt',
    }
    partitions = {name: read_cases(path) for name, path in source.items()}
    excluded: dict[str, list[str]] = {}
    for name, cases in partitions.items():
        removed = [case for case in cases if any(case.startswith(prefix) for prefix in args.exclude_prefix)]
        excluded[name] = removed
        partitions[name] = [case for case in cases if case not in set(removed)]
    counts = validate(partitions)
    if args.check_only:
        print(json.dumps({'fold': args.fold, 'counts': counts, 'status': 'disjoint'}, indent=2))
        return
    output = args.output_root/(args.name or f'fold{args.fold}')
    output.mkdir(parents=True, exist_ok=True)
    for name, cases in partitions.items():
        (output/f'{name}.txt').write_text('\n'.join(cases) + '\n')
    # KonfAI 1.8's DataTrain takes a population subset then removes its explicit
    # validation selector.  Keep the technical union separate from train.txt.
    fit = partitions['train'] + partitions['validation']
    (output/'fit.txt').write_text('\n'.join(fit) + '\n')
    manifest = {
        'schema': 'impactseg-v2-explicit-splits/v1',
        'fold': args.fold,
        'source': {name: str(path.relative_to(repo)) for name, path in source.items()},
        'counts': counts,
        'sha256': {name: digest(cases) for name, cases in partitions.items()} | {'fit': digest(fit)},
        'fit_count': len(fit),
        'fit_definition': 'ordered union of explicit train and validation; KonfAI removes validation from this population',
        'excluded_case_prefixes': args.exclude_prefix,
        'excluded_cases': excluded,
        'guarantee': 'train, validation and test are pairwise disjoint by case ID',
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
