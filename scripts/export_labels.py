#!/usr/bin/env python3
"""Regenerate configs/labels.json from impactseg_v2/labels.py, the only source of the label definitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from impactseg_v2.labels import CONVENTION_ID, FINE_LABELS, MAP_559_LABELS, NOT_PREDICTED_FINE_IDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "labels.json")
    args = parser.parse_args()
    source = ROOT / "impactseg_v2" / "labels.py"
    value = {"schema": "impactseg-labels/v1", "convention_id": CONVENTION_ID,
             "authoritative_source": "impactseg_v2/labels.py",
             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "fine_map": {"foreground_classes": len(FINE_LABELS) - 1,
                          "network_outputs_including_background": len(FINE_LABELS), "labels": FINE_LABELS,
                          "not_output_by_the_models": {i: FINE_LABELS[i] for i in NOT_PREDICTED_FINE_IDS},
                          "structures_output_by_the_models": len(FINE_LABELS) - 1 - len(NOT_PREDICTED_FINE_IDS)},
             "coarse_map": {"foreground_classes": len(MAP_559_LABELS) - 1,
                            "network_outputs_including_background": len(MAP_559_LABELS),
                            "hub_filename": "mask_coarse.nii.gz", "labels": MAP_559_LABELS},
             "note": "These maps define IDs, not guaranteed per-case presence or annotation quality."}
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
