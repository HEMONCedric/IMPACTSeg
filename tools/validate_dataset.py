#!/usr/bin/env python3
"""Targeted, read-only IMPACTSeg data and label-convention validator.

It intentionally samples each source family instead of silently claiming a full
corpus audit.  A non-zero exit means a case has an invalid label ID, an image /
mask grid mismatch, or a detected anatomy-order contradiction.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from impactseg_v2.labels import (  # noqa: E402
    CONVENTION_ID,
    FINE_LABELS,
    MAP_559_LABELS,
    invalid_labels,
)


def source_family(modality: str, case: str) -> str:
    """Stable family labels for representative sampling and reporting."""
    if modality != "MRI":
        return modality
    if case.startswith("AMOS22_"):
        return "MRI/AMOS22"
    if case.startswith("Totalsegmentator_"):
        return "MRI/TotalSegmentator"
    if case.startswith("big_mac_"):
        return "MRI/BigMAC"
    if case.startswith("SynthRAD"):
        return "MRI/SynthRAD"
    return "MRI/other"


def centroid_world(data: np.ndarray, affine: np.ndarray, label: int) -> np.ndarray | None:
    voxels = np.where(data == label)
    if voxels[0].size == 0:
        return None
    voxel = np.array([axis.mean() for axis in voxels] + [1.0])
    return (affine @ voxel)[:3]


def anatomy_checks(data: np.ndarray, affine: np.ndarray) -> list[str]:
    """Cheap semantic checks focused on IDs whose MRI convention previously differed."""
    warnings: list[str] = []
    vertebrae = [(label, centroid_world(data, affine, label)) for label in range(18, 42)]
    present = [(label, point) for label, point in vertebrae if point is not None]
    for (a, pa), (b, pb) in zip(present, present[1:]):
        # CT formalism: L5 (18) -> C1 (41), increasingly superior in RAS world z.
        if not pa[2] < pb[2]:
            warnings.append(f"vertebra_order:{a}->{b} has RAS-z {pa[2]:.1f}>={pb[2]:.1f}")
    # In RAS world coordinates, anatomical left has lower X than anatomical right.
    pairs = [(3, 2), (12, 11), (50, 51), (52, 53), (58, 59), (60, 61), (62, 63), (64, 65), (66, 67)]
    for left, right in pairs:
        p_left, p_right = centroid_world(data, affine, left), centroid_world(data, affine, right)
        if p_left is not None and p_right is not None and not p_left[0] < p_right[0]:
            warnings.append(f"laterality:{left}/{right} has left RAS-x >= right")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=2, help="Representative cases per source family.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-mask559-brain", action="store_true", help="Require label 11 in the sampled masks.")
    args = parser.parse_args()
    if args.per_family < 1:
        raise SystemExit("--per-family must be positive")

    candidates: dict[str, list[Path]] = defaultdict(list)
    for modality_dir in sorted(path for path in args.dataset_root.iterdir() if path.is_dir()):
        for mask in sorted(modality_dir.glob("*/mask_fine.nii.gz")):
            candidates[source_family(modality_dir.name, mask.parent.name)].append(mask.parent)
    selected = [case for family in sorted(candidates) for case in candidates[family][:args.per_family]]
    if not selected:
        raise SystemExit("No case with mask_fine.nii.gz found.")

    results: list[dict[str, object]] = []
    failures: list[str] = []
    observed_559: set[int] = set()
    for case_dir in selected:
        image_path, fine_path, map559_path = (case_dir / "image.nii.gz", case_dir / "mask_fine.nii.gz", case_dir / "mask_559.nii.gz")
        entry: dict[str, object] = {"case": str(case_dir.relative_to(args.dataset_root)), "family": source_family(case_dir.parent.name, case_dir.name)}
        if not image_path.exists():
            failures.append(f"{entry['case']}: missing image")
            results.append(entry)
            continue
        image, fine = nib.load(image_path), nib.load(fine_path)
        fine_data = np.asanyarray(fine.dataobj)
        fine_ids = {int(v) for v in np.unique(fine_data)}
        fine_invalid = invalid_labels(fine_ids, FINE_LABELS)
        map559_ids: set[int] = set()
        map559_invalid: set[int] = set()
        map559_present = map559_path.exists()
        map559_grid_ok = True
        if map559_present:
            map559 = nib.load(map559_path)
            map559_data = np.asanyarray(map559.dataobj)
            map559_ids = {int(v) for v in np.unique(map559_data)}
            observed_559 |= map559_ids
            map559_invalid = invalid_labels(map559_ids, MAP_559_LABELS)
            map559_grid_ok = map559.shape == image.shape and np.allclose(map559.affine, image.affine)
        grid_ok = image.shape == fine.shape and np.allclose(image.affine, fine.affine) and map559_grid_ok
        anatomy = anatomy_checks(fine_data, fine.affine) if case_dir.parent.name == "MRI" else []
        entry.update({
            "image_shape": list(image.shape), "fine_ids": sorted(fine_ids), "mask_559_ids": sorted(map559_ids),
            "fine_invalid": sorted(fine_invalid), "mask_559_present": map559_present, "mask_559_invalid": sorted(map559_invalid), "grid_ok": bool(grid_ok),
            "anatomy_warnings": anatomy,
        })
        if fine_invalid or map559_invalid or not grid_ok or anatomy:
            failures.append(f"{entry['case']}: invalid={sorted(fine_invalid | map559_invalid)}, grid_ok={grid_ok}, anatomy={anatomy}")
        results.append(entry)

    report = {
        "convention": CONVENTION_ID, "sampled_cases": len(results), "per_family": args.per_family,
        "families": {family: len(cases) for family, cases in sorted(candidates.items())},
        "observed_mask_559_ids": sorted(observed_559), "mask_559_brain_observed": 11 in observed_559,
        "failures": failures, "cases": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if args.require_mask559_brain and 11 not in observed_559:
        failures.append("mask_559 label 11 (brain) was absent from the representative sample")
    if failures:
        raise SystemExit(f"FAIL: {len(failures)} validation issue(s)")


if __name__ == "__main__":
    main()
