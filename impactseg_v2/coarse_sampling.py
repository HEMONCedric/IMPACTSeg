"""Opt-in modality/case-balanced patch sampling for coarse-label experiments."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Sampler
from konfai.utils.config import config
from konfai.data.data_manager.order import PatchReadOrder


class EpochPatchReadOrder(PatchReadOrder):
    """Pinned-KonfAI read-plan hint with an epoch size independent of grid size.

    Native PatchReadOrder allocates len(mapping), suitable for an exhaustive
    traversal. This experiment samples a fixed number of patches with replacement.
    Only its shared planning buffer changes; indices still address the real mapping.
    Create before worker fork so the sampler and readers share the same buffer.
    """
    def __init__(self, mapping, batch_size: int, epoch_size: int):
        super().__init__(mapping, batch_size)
        self._order = torch.zeros(epoch_size, dtype=torch.int64).share_memory_()


@config("modality_sampling")
class ModalitySampling:
    def __init__(self, enabled: bool = False, index_file: str = "", samples_per_epoch: int = 6000,
                 seed: int = 42) -> None:
        if samples_per_epoch < 3 or samples_per_epoch % 3:
            raise ValueError("samples_per_epoch must be positive and divisible by 3")
        self.enabled = bool(enabled)
        self.index_file = index_file
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)


class ModalityCaseForegroundSampler(Sampler[int]):
    """Equal CT/CBCT/MRI draws, uniform cases, then foreground/uniform patches.

    Three-draw blocks contain one draw of each modality, in a shuffled order.
    A fixed epoch seed gives reproducible varying epochs. Validation is never
    replaced. Unlike patch-weighted balancing, large volumes gain no extra weight.
    """
    def __init__(self, mapping, case_modalities: dict[int, str], foreground_indices: list[int],
                 foreground_probability: float, samples_per_epoch: int, seed: int, read_order) -> None:
        if samples_per_epoch % 3 or samples_per_epoch < 3:
            raise ValueError("Balanced epoch size must be divisible by 3")
        if not 0 <= foreground_probability <= 1:
            raise ValueError("Invalid foreground probability")
        self.mapping = mapping
        self.by_modality = defaultdict(list)
        self.by_case = defaultdict(list)
        self.foreground_by_case = defaultdict(list)
        foreground = set(foreground_indices)
        for index, (case, _, _) in enumerate(mapping):
            self.by_case[case].append(index)
            if index in foreground:
                self.foreground_by_case[case].append(index)
        for case in sorted(self.by_case):
            modality = case_modalities[case]
            self.by_modality[modality].append(case)
        if set(self.by_modality) != {"CT", "CBCT", "MRI"}:
            raise ValueError("All three modalities must have training cases on this rank")
        self.probability = foreground_probability
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.read_order = read_order

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = []
        for _ in range(self.samples_per_epoch // 3):
            modalities = ["CT", "CBCT", "MRI"]
            rng.shuffle(modalities)
            for modality in modalities:
                case = rng.choice(self.by_modality[modality])
                eligible = self.foreground_by_case[case]
                candidates = eligible if eligible and rng.random() < self.probability else self.by_case[case]
                order.append(rng.choice(candidates))
        self.read_order.publish(torch.tensor(order, dtype=torch.int64))
        return iter(order)


def case_modalities(dataset, settings: ModalitySampling) -> dict[int, str]:
    raw = json.loads(Path(settings.index_file).read_text())
    indexed = {row["case_id"]: row for row in raw["cases"]}
    result = {}
    for index, manager in enumerate(dataset.data["image"]):
        row = indexed.get(manager.name)
        if row is None or row["partition"] != "train":
            raise ValueError(f"Case missing from the explicit training index: {manager.name}")
        result[index] = row["family"]
    return result
