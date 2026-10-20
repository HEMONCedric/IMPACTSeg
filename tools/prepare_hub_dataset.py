#!/usr/bin/env python3
"""Lay out a Hugging Face download of HCedric/IMPACTSeg as the raw training tree.

The split files use the case IDs of the original curation (for example ``TotalSegV2_CADS__s0650``),
whereas the Hub stores cases as ``CT/TotalSeg/s0650``. ``splits/hub_crosswalk.csv`` maps one to the
other (by name, and by image SHA-256 for BIG-MAC; a random sample of 150 name matches was checked
by SHA-256). This tool creates

    <output-root>/raw/<modality>/<case_id>/{image,mask_fine}.nii.gz   (symbolic links into the download)
    <output-root>/splits/fold<K>/{train,validation,test,fit}.txt     (split files restricted to the Hub)

The 107 VISCERAL CT cases of the original splits are not redistributed on the Hub: they are dropped
from the copied split files, and every drop is listed in ``splits/fold<K>/manifest.json``. The test
partitions therefore differ slightly from the manuscript's internal evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "validation", "test")


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub-root", type=Path, required=True,
                        help="Local download of the dataset repository (contains CT/, CBCT/, MRI/).")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, default=ROOT / "splits")
    parser.add_argument("--crosswalk", type=Path, default=ROOT / "splits" / "hub_crosswalk.csv")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    hub_root, output_root = args.hub_root.resolve(), args.output_root.resolve()
    crosswalk = {row["case_id"]: row for row in csv.DictReader(args.crosswalk.open())}
    available, missing_files = {}, []
    for case_id, row in crosswalk.items():
        if not row["hub_case_path"]:
            continue
        source = hub_root / row["hub_case_path"]
        if not all((source / f"{name}.nii.gz").is_file() for name in ("image", "mask_fine")):
            missing_files.append(row["hub_case_path"])
            continue
        destination = output_root / "raw" / row["local_modality"] / case_id
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("image", "mask_fine"):
            link = destination / f"{name}.nii.gz"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(source / f"{name}.nii.gz")
        available[case_id] = row

    for fold in args.folds:
        source_dir, target_dir = args.splits_root / f"fold{fold}", output_root / "splits" / f"fold{fold}"
        target_dir.mkdir(parents=True, exist_ok=True)
        kept, dropped = {}, {}
        for partition in PARTITIONS:
            cases = [c for c in (source_dir / f"{partition}.txt").read_text().split() if c]
            kept[partition] = [c for c in cases if c in available]
            dropped[partition] = [c for c in cases if c not in available]
            (target_dir / f"{partition}.txt").write_text("".join(f"{c}\n" for c in kept[partition]))
        (target_dir / "fit.txt").write_text("".join(f"{c}\n" for c in kept["train"] + kept["validation"]))
        manifest = {
            "schema": "impactseg-v2-explicit-splits/v1", "fold": fold,
            "source": {"split_dir": str(source_dir), "restricted_to": "cases available on the Hugging Face dataset"},
            "counts": {p: len(kept[p]) for p in PARTITIONS},
            "sha256": {p: sha256_text(target_dir / f"{p}.txt") for p in (*PARTITIONS, "fit")},
            "fit_count": len(kept["train"]) + len(kept["validation"]),
            "dropped_not_on_hub": dropped,
        }
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"fold{fold}: " + ", ".join(f"{p} {len(kept[p])} (-{len(dropped[p])})" for p in PARTITIONS))
    print(f"{len(available)} cases linked under {output_root / 'raw'}")
    if missing_files:
        print(f"WARNING: {len(missing_files)} crosswalk cases are missing from the download, e.g. {missing_files[:3]}")


if __name__ == "__main__":
    main()
