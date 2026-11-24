# Contributing

This is a research repository. Discuss scientific changes before changing label semantics, normalization,
patch geometry, splits or metrics. Do not commit medical images, checkpoints, credentials or cluster-specific
configuration.

`impactseg/labels.py` is the only source of the label definitions; regenerate `configs/labels.json` with
`python scripts/export_labels.py` instead of editing it. Run
`python tools/validate_splits.py --split-dir splits/fold<K>` before a training or inference change, and record
which checkpoint, split hashes and metric aggregation produced each result.

New augmentations must preserve label interpolation and padding semantics, and declare whether they are
pointwise, patch-local, regridding or need the whole image (KonfAI patch streaming depends on it).

Bug reports should include versions, a sanitized configuration, image geometry and the traceback, never
identifiable patient data.
