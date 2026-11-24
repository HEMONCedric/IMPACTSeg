"""KonfAI TRAIN entrypoint with the IMPACTSeg Dataset override installed."""

from __future__ import annotations

import itertools
import math
import os
import runpy

from impactseg.sampling import ForegroundSamplingDataTrain
from impactseg.streaming import install_extend_slice_region_streaming_fix
from impactseg.checkpoint_retention import install_checkpoint_retention


class _IterationLimitedLoader:
    """Expose at most ``limit`` batches while preserving the loader dataset."""

    def __init__(self, loader, limit: int) -> None:
        self.loader = loader
        self.limit = int(limit)
        self.dataset = loader.dataset

    def __iter__(self):
        return itertools.islice(self.loader, self.limit)

    def __len__(self) -> int:
        return min(self.limit, len(self.loader))

    def __getattr__(self, name):
        return getattr(self.loader, name)


def _install_iteration_limit(trainer) -> None:
    """Stop exactly at IMPACTSEG_MAX_ITERATIONS without replacing KonfAI's train loop."""

    if getattr(trainer._Trainer, "_impactseg_iteration_limit", False):
        return
    original_run = trainer._Trainer.run
    original_train = trainer._Trainer.train

    def configured_limit() -> int | None:
        raw = os.environ.get("IMPACTSEG_MAX_ITERATIONS")
        if raw is None:
            return None
        value = int(raw)
        if value <= 0:
            raise ValueError("IMPACTSEG_MAX_ITERATIONS must be strictly positive.")
        return value

    def run(self):
        limit = configured_limit()
        if limit is not None and self.it < limit:
            remaining = limit - self.it
            required_epochs = self.epoch + math.ceil(remaining / len(self.dataloader_training))
            self.epochs = max(self.epochs, required_epochs)
        return original_run(self)

    def train(self):
        limit = configured_limit()
        if limit is None:
            return original_train(self)
        remaining = limit - self.it
        if remaining <= 0:
            self.early_stopping.stop()
            return None
        loader = self.dataloader_training
        if remaining < len(loader):
            self.dataloader_training = _IterationLimitedLoader(loader, remaining)
        try:
            result = original_train(self)
        finally:
            self.dataloader_training = loader
        if self.it >= limit:
            if self.global_rank == 0:
                print(f"[IMPACTSeg] reached exact training limit at iteration {self.it}.")
            self.early_stopping.stop()
        return result

    trainer._Trainer.run = run
    trainer._Trainer.train = train
    trainer._Trainer._impactseg_iteration_limit = True


def install() -> None:
    """Bind KonfAI's trainer annotation to the IMPACTSeg dataset before YAML resolution."""

    import konfai.trainer as trainer

    install_extend_slice_region_streaming_fix()
    trainer.DataTrain = ForegroundSamplingDataTrain
    trainer.Trainer.__init__.__annotations__["dataset"] = ForegroundSamplingDataTrain
    _install_iteration_limit(trainer)
    install_checkpoint_retention(trainer)


def main() -> None:
    install()
    runpy.run_module("konfai.main", run_name="__main__")


if __name__ == "__main__":
    main()
