"""Helpers for running the released IMPACT-Seg models with a plain ``konfai PREDICTION``."""

from __future__ import annotations

import torch
from konfai.data.transform import Transform
from konfai.utils.dataset import Attribute


class LoadWholeVolume(Transform):
    """Identity stage that keeps KonfAI's default WHOLE_VOLUME patch locality.

    It makes KonfAI load each case whole instead of streaming disk regions, which the 2.5D
    slice-context patches (``extend_slice``) need, and matches how the released results were produced.
    """

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor
