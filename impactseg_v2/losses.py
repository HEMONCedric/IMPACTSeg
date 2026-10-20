"""Small IMPACTSeg-specific criteria retained for scientific equivalence."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from konfai.metric.measure import Criterion


def _foreground_dice(probabilities: torch.Tensor, target: torch.Tensor, smooth: float) -> tuple[torch.Tensor, torch.Tensor]:
    target = functional.interpolate(target.float(), probabilities.shape[2:], mode="nearest").long().squeeze(1)
    one_hot = functional.one_hot(target, num_classes=probabilities.shape[1]).movedim(-1, 1).float()
    probabilities, one_hot = probabilities[:, 1:], one_hot[:, 1:]
    reduce_dims = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice, one_hot.sum(dim=reduce_dims) > 0


class SegCrossEntropy(Criterion):
    """Historical weighted CE: background weight and smoothing are IMPACTSeg choices."""

    def __init__(self, background_weight: float = 0.2, label_smoothing: float = 0.02) -> None:
        super().__init__()
        self.background_weight = float(background_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = functional.interpolate(targets[0].float(), output.shape[2:], mode="nearest").long().squeeze(1)
        weights = torch.ones(output.shape[1], device=output.device, dtype=output.dtype)
        weights[0] = self.background_weight
        return functional.cross_entropy(output, target, weight=weights, label_smoothing=self.label_smoothing)


class DiceNoBackground(Criterion):
    """Historical foreground Dice on probabilities, optionally ignoring absent target labels."""

    maximize = True

    def __init__(self, smooth: float = 1e-6, only_present: bool = True, as_loss: bool = True, coefficient: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.only_present = bool(only_present)
        self.as_loss = bool(as_loss)
        self.coefficient = float(coefficient)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        dice, present = _foreground_dice(output, targets[0], self.smooth)
        score = dice[present].mean() if self.only_present and present.any() else (
            output.new_zeros(()) if self.only_present else dice.mean()
        )
        return self.coefficient * (1.0 - score) if self.as_loss else score


class DiceNoBackgroundAbsentAware(DiceNoBackground):
    """Foreground Dice term that penalises probability mass on absent labels."""

    def __init__(self, smooth: float = 1e-6, coefficient: float = 0.1) -> None:
        super().__init__(smooth=smooth, only_present=False, as_loss=True, coefficient=coefficient)


class DiceNoBackgroundMetric(DiceNoBackground):
    """Logged foreground Dice, retaining the old metric's present-label policy."""

    def __init__(self, smooth: float = 1e-6, only_present: bool = True) -> None:
        super().__init__(smooth=smooth, only_present=only_present, as_loss=False)
