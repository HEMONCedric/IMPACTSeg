#!/usr/bin/env python3
"""Materialize IMPACTSeg deterministic training transforms with KonfAI.

The raw dataset tree is never modified. Selected cases are standardized and
resampled into an uncompressed modality/case layout. KonfAI's Write implementation
owns per-entry publication; reruns with ``--overwrite`` recompute from source
rather than transforming an already transformed output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from konfai.api import plan_transform, transform
from konfai.data.transform import Resample, Standardize, TensorCast, Write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from decompress_training_dataset import index_cases, read_selection  # noqa: E402

DEFAULT_MODALITIES = ("CT", "CBCT", "MRI", "SynthRADSegMRI", "SynthRADSegCBCT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument(
        "--partitions", nargs="+", choices=("train", "validation", "test"),
        default=["train", "validation"],
    )
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--memory-budget", default="64 GiB")
    parser.add_argument("--transforms-root", type=Path, default=None)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "Optional preprocessing report destination. By default the dataset-level "
            "_provenance/preprocessing_report.json file is used."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def chains(destination: Path) -> dict[str, dict[str, list[object]]]:
    output = str(destination) + ":nii"
    return {
        "image": {
            "image": [
                Standardize(lazy=False, mean=None, std=None, mask=None, inverse=False),
                TensorCast(dtype="float32", inverse=False),
                Resample(spacing=[1.5, 1.5, 2.0], align="extent", inverse=False),
                Write(dataset=output, group="image"),
            ]
        },
        "mask_fine": {
            "mask_fine": [
                TensorCast(dtype="uint8", inverse=False),
                Resample(spacing=[1.5, 1.5, 2.0], align="extent", inverse=False),
                Write(dataset=output, group="mask_fine"),
            ]
        },
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.gpus and args.workers != 6:
        raise SystemExit("Use either --gpus or --workers; do not override both.")

    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    splits_root = args.splits_root.resolve()
    if source_root == destination_root:
        raise SystemExit("Source and destination must differ.")
    if not source_root.is_dir():
        raise SystemExit(f"Source dataset not found: {source_root}")
    if not destination_root.is_dir():
        raise SystemExit(f"Destination dataset not found: {destination_root}")
    if not args.plan and not args.overwrite:
        raise SystemExit(
            "The destination already contains the raw uncompressed view. "
            "Pass --overwrite to replace it from the immutable source."
        )

    selected = read_selection(splits_root, args.folds, args.partitions)
    # A partial modality run intentionally excludes cases owned by the other
    # known modality roots; genuinely missing cases in the requested roots
    # remain selected and are therefore still reported as errors below.
    excluded: set[str] = set()
    for modality in set(DEFAULT_MODALITIES).difference(args.modalities):
        modality_root = source_root / modality
        if modality_root.is_dir():
            excluded.update(path.name for path in modality_root.iterdir() if path.is_dir())
    scoped_selection = selected.difference(excluded)
    indexed = index_cases(
        source_root, destination_root, args.modalities, scoped_selection, ["image", "mask_fine"]
    )
    by_modality: dict[str, list[str]] = defaultdict(list)
    for case in indexed:
        by_modality[case.modality].append(case.case_id)
    if args.limit is not None:
        by_modality = {
            modality: names[: args.limit] for modality, names in by_modality.items() if names
        }

    transforms_root = (
        args.transforms_root.resolve()
        if args.transforms_root is not None
        else destination_root / "_provenance" / "preprocessing_konfai"
    )
    report_path = (
        args.report_path.resolve()
        if args.report_path is not None
        else destination_root / "_provenance" / "preprocessing_report.json"
    )
    report = {
        "schema": "impactseg-preprocessed-training-view/v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "PLANNED" if args.plan else "RUNNING",
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "splits_root": str(splits_root),
        "folds": args.folds,
        "partitions": args.partitions,
        "modalities": args.modalities,
        "selected_unique_cases": sum(len(names) for names in by_modality.values()),
        "cases_by_modality": {modality: len(names) for modality, names in sorted(by_modality.items())},
        "image_chain": ["Standardize", "TensorCast(float32)", "Resample(1.5,1.5,2.0,align=extent)"],
        "mask_chain": ["TensorCast(uint8)", "Resample(1.5,1.5,2.0,align=extent)"],
        "konfai_runs": [],
    }
    print(json.dumps(report, indent=2), flush=True)

    runner = plan_transform if args.plan else transform
    for modality in args.modalities:
        names = sorted(by_modality.get(modality, []))
        if not names:
            continue
        name = f"impactseg_preprocess_{modality}"
        source = str(source_root / modality) + ":a:nii.gz"
        destination = destination_root / modality
        print(f"[IMPACTSeg] {modality}: {len(names)} cases", flush=True)
        result = runner(
            name=name,
            datasets=source,
            chains=chains(destination),
            memory_budget=args.memory_budget,
            on_fallback="warn",
            dataset_options={"subset": names},
            gpu=args.gpus,
            cpu=1 if args.gpus else args.workers,
            overwrite=args.overwrite,
            transforms_dir=transforms_root,
        )
        run_record = {"modality": modality, "cases": len(names)}
        if not args.plan:
            run_record.update({
                "workspace": str(result.workspace),
                "config": str(result.config),
                "outputs": result.outputs,
            })
        report["konfai_runs"].append(run_record)
        if not args.plan:
            atomic_json(report_path, report)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "PLAN_COMPLETE" if args.plan else "PASS"
    if not args.plan:
        atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
