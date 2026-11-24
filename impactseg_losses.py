"""Dot-free KonfAI criterion import aliases.

KonfAI 1.8 criterion-loader configuration paths are dot-separated. Keeping this
top-level module lets custom criterion settings be bound without splitting the
Python module name impactseg.losses.
"""

from impactseg.losses import (
    DiceNoBackground,
    DiceNoBackgroundAbsentAware,
    DiceNoBackgroundMetric,
    SegCrossEntropy,
)

__all__ = [
    "DiceNoBackground",
    "DiceNoBackgroundAbsentAware",
    "DiceNoBackgroundMetric",
    "SegCrossEntropy",
]
