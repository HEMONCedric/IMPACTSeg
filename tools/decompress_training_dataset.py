#!/usr/bin/env python3
"""Build an uncompressed NIfTI view for cases used across IMPACTSeg folds.

The source dataset is immutable. For every selected case, ``image.nii.gz`` and
``mask_fine.nii.gz`` are decompressed byte-for-byte to ``.nii`` files while
preserving the modality/case directory layout. gzip CRC failures, missing
inputs, duplicate case identifiers, partial outputs, and geometry mismatches
are fatal. An interrupted conversion can be continued with ``--resume``.
"""

from __future__ import annotations

import argparse
import filecmp
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_MODALITIES = (
    "CT",
    "CBCT",
    "MRI",
    "SynthRADSegMRI",
    "SynthRADSegCBCT",
)
DEFAULT_FILES = ("image", "mask_fine")
DEFAULT_PARTITIONS = ("train", "validation")


@dataclass(frozen=True)
class Case:
    case_id: str
    modality: str
    source: Path
    destination: Path


@dataclass
class CaseResult:
    case_id: str
    modality: str
    converted: int
    resumed: int
    bytes_written: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument(
        "--partitions", nargs="+", default=list(DEFAULT_PARTITIONS),
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep existing outputs only when their size matches the gzip payload.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing derived files. Never modifies source files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_selection(splits_root: Path, folds: list[int], partitions: list[str]) -> set[str]:
    selected: set[str] = set()
    for fold in folds:
        split_dir = splits_root / f"fold{fold}"
        for partition in partitions:
            path = split_dir / f"{partition}.txt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing split file: {path}")
            entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if len(entries) != len(set(entries)):
                raise ValueError(f"Duplicate case IDs inside {path}")
            selected.update(entries)
    if not selected:
        raise ValueError("The requested folds/partitions selected no cases")
    return selected


def index_cases(
    source_root: Path,
    destination_root: Path,
    modalities: list[str],
    selected: set[str],
    stems: list[str],
) -> list[Case]:
    matches: dict[str, Case] = {}
    for modality in modalities:
        modality_root = source_root / modality
        if not modality_root.is_dir():
            raise FileNotFoundError(f"Missing modality root: {modality_root}")
        for case_dir in modality_root.iterdir():
            if not case_dir.is_dir() or case_dir.name not in selected:
                continue
            if case_dir.name in matches:
                previous = matches[case_dir.name].source
                raise ValueError(
                    f"Duplicate selected case ID {case_dir.name}: {previous} and {case_dir}"
                )
            missing = [stem for stem in stems if not (case_dir / f"{stem}.nii.gz").is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{modality}/{case_dir.name} is missing: "
                    + ", ".join(f"{stem}.nii.gz" for stem in missing)
                )
            matches[case_dir.name] = Case(
                case_id=case_dir.name,
                modality=modality,
                source=case_dir,
                destination=destination_root / modality / case_dir.name,
            )
    missing_ids = sorted(selected - matches.keys())
    if missing_ids:
        raise FileNotFoundError(
            f"{len(missing_ids)} selected case IDs were not found; examples: {missing_ids[:10]}"
        )
    return sorted(matches.values(), key=lambda case: (case.modality, case.case_id))


def gzip_payload_size(path: Path) -> int:
    with path.open("rb") as stream:
        stream.seek(-4, os.SEEK_END)
        return struct.unpack("<I", stream.read(4))[0]


def existing_output_is_resumable(source: Path, destination: Path) -> bool:
    # NIfTI volumes in IMPACTSeg are below gzip's 4 GiB ISIZE wrap boundary.
    return destination.is_file() and destination.stat().st_size == gzip_payload_size(source)


def decompress_one(source: Path, destination: Path, resume: bool, overwrite: bool) -> tuple[str, int]:
    if destination.exists():
        if resume and existing_output_is_resumable(source, destination):
            return "resumed", 0
        if not overwrite:
            raise FileExistsError(
                f"Refusing existing or partial output {destination}; use --resume for a valid "
                "completed file or --overwrite to replace it"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{threading.get_ident()}"
    )
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb") as output:
            command = ["pigz", "-p", "1", "-dc", "--", str(source)]
            completed = subprocess.run(command, stdout=output, stderr=subprocess.PIPE)
            output.flush()
            os.fsync(output.fileno())
        if completed.returncode != 0:
            detail = completed.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"pigz failed for {source}: {detail}")
        expected = gzip_payload_size(source)
        actual = temporary.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"Decompressed size mismatch for {source}: expected {expected}, got {actual}"
            )
        os.replace(temporary, destination)
        shutil.copystat(source, destination)
        return "converted", actual
    finally:
        temporary.unlink(missing_ok=True)


def copy_source_metadata(case: Case, resume: bool, overwrite: bool) -> None:
    source = case.source / "source.json"
    if not source.is_file():
        return
    destination = case.destination / "source.json"
    if destination.exists():
        if resume and filecmp.cmp(source, destination, shallow=False):
            return
        if not overwrite:
            raise FileExistsError(f"Refusing different metadata output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_pair(case: Case) -> None:
    image_path = case.destination / "image.nii"
    mask_path = case.destination / "mask_fine.nii"
    if not image_path.is_file() or not mask_path.is_file():
        return
    image = nib.load(str(image_path), mmap=True)
    mask = nib.load(str(mask_path), mmap=True)
    if image.shape != mask.shape:
        raise ValueError(
            f"Geometry mismatch for {case.modality}/{case.case_id}: "
            f"image shape {image.shape}, mask shape {mask.shape}"
        )
    if not np.allclose(image.affine, mask.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"Affine mismatch for {case.modality}/{case.case_id}")


def process_case(case: Case, stems: list[str], resume: bool, overwrite: bool) -> CaseResult:
    converted = resumed = bytes_written = 0
    for stem in stems:
        status, written = decompress_one(
            case.source / f"{stem}.nii.gz",
            case.destination / f"{stem}.nii",
            resume,
            overwrite,
        )
        converted += status == "converted"
        resumed += status == "resumed"
        bytes_written += written
    copy_source_metadata(case, resume, overwrite)
    if {"image", "mask_fine"}.issubset(stems):
        validate_pair(case)
    return CaseResult(case.case_id, case.modality, converted, resumed, bytes_written)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if shutil.which("pigz") is None:
        raise SystemExit("pigz is required")

    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    splits_root = args.splits_root.resolve()
    if source_root == destination_root:
        raise SystemExit("Source and destination roots must differ")
    if not source_root.is_dir():
        raise SystemExit(f"Source dataset does not exist: {source_root}")

    selected = read_selection(splits_root, args.folds, args.partitions)
    cases = index_cases(
        source_root, destination_root, args.modalities, selected, args.files
    )
    counts = dict(sorted(Counter(case.modality for case in cases).items()))
    source_files = [
        case.source / f"{stem}.nii.gz" for case in cases for stem in args.files
    ]
    estimated_bytes = sum(gzip_payload_size(path) for path in source_files)
    print(json.dumps({
        "selected_unique_cases": len(cases),
        "cases_by_root": counts,
        "files_to_materialise": len(source_files),
        "estimated_uncompressed_gib": round(estimated_bytes / 2**30, 2),
        "destination": str(destination_root),
        "dry_run": args.dry_run,
    }, indent=2))
    if args.dry_run:
        return

    destination_root.mkdir(parents=True, exist_ok=True)
    if any(destination_root.iterdir()) and not (args.resume or args.overwrite):
        raise SystemExit(
            f"Destination is not empty: {destination_root}. Use --resume after an "
            "interrupted run or --overwrite intentionally."
        )
    free_bytes = shutil.disk_usage(destination_root).free
    if not args.resume and free_bytes < int(estimated_bytes * 1.05):
        raise SystemExit(
            f"Insufficient free space: need about {estimated_bytes / 2**30:.2f} GiB "
            f"plus 5%, have {free_bytes / 2**30:.2f} GiB"
        )

    started = time.monotonic()
    results: list[CaseResult] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_case, case, args.files, args.resume, args.overwrite): case
            for case in cases
        }
        for done, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # report every anomaly; final status is failure
                failures.append({
                    "case_id": case.case_id,
                    "modality": case.modality,
                    "error": f"{type(error).__name__}: {error}",
                })
                print(f"ERROR {case.modality}/{case.case_id}: {error}", file=sys.stderr)
            if done == 1 or done % 25 == 0 or done == len(cases):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{done}/{len(cases)}] failures={len(failures)} "
                    f"rate={done / elapsed:.2f} cases/s",
                    flush=True,
                )

    report = {
        "schema": "impactseg-uncompressed-training-view/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "splits_root": str(splits_root),
        "folds": args.folds,
        "partitions": args.partitions,
        "modalities": args.modalities,
        "files": args.files,
        "unique_cases": len(cases),
        "cases_by_root": counts,
        "converted_files": sum(result.converted for result in results),
        "resumed_files": sum(result.resumed for result in results),
        "bytes_written": sum(result.bytes_written for result in results),
        "failures": failures,
    }
    provenance = destination_root / "_provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "selected_case_ids.txt").write_text(
        "".join(f"{case_id}\n" for case_id in sorted(selected))
    )
    write_report(provenance / "conversion_report.json", report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"Conversion failed for {len(failures)} case(s); see {provenance}")


if __name__ == "__main__":
    main()
