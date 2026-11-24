"""Opt-in BEST/LAST aliases outside native KonfAI BEST pruning.

The native asynchronous writer publishes a complete checkpoint before invoking
_update_best_checkpoint. Capture LAST at that boundary, before a non-best file
is unlinked, then capture the selected BEST. KonfAI still owns serialization,
scores, optimizer state and selection. Aliases live in a subdirectory ignored
by its non-recursive *.pt scan, so a future resume cannot prune them.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


def atomic_alias(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a complete native checkpoint: {source}")
    destination.parent.mkdir(exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("Refusing symlink checkpoint retention directory")
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        try:
            os.link(source, staged)
        except OSError:
            # Some filesystems disallow hardlinks; copy before atomic publication.
            shutil.copyfile(source, staged)
        os.replace(staged, destination)


def install_checkpoint_retention(trainer) -> None:
    cls = trainer._Trainer
    if getattr(cls, "_impactseg_retained_checkpoints", False):
        return
    if not hasattr(cls, "_update_best_checkpoint"):
        if os.environ.get("IMPACTSEG_KEEP_LAST") == "1":
            raise RuntimeError("KonfAI checkpoint API changed; BEST/LAST retention must be adapted")
        return
    original_update = cls._update_best_checkpoint

    def update(self, checkpoint_path: Path, loss: float):
        enabled = os.environ.get("IMPACTSEG_KEEP_LAST") == "1"
        retained = checkpoint_path.parent / "retained"
        if enabled:
            atomic_alias(checkpoint_path, retained / "LAST.pt")
        result = original_update(self, checkpoint_path, loss)
        if enabled and self._best_checkpoint_path is not None:
            atomic_alias(self._best_checkpoint_path, retained / "BEST.pt")
        return result

    cls._update_best_checkpoint = update
    cls._impactseg_retained_checkpoints = True
