from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _probabilities(values: Sequence[float] | Mapping[str, float]) -> np.ndarray:
    raw = list(values.values()) if isinstance(values, Mapping) else values
    array = np.asarray(raw, dtype=float)
    if array.ndim != 1 or len(array) == 0 or np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("probabilities must be a finite non-negative vector")
    if not math.isclose(float(array.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    return array


def predictive_entropy(probabilities: Sequence[float] | Mapping[str, float], *, normalized: bool = False) -> float:
    values = _probabilities(probabilities)
    entropy = float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum())
    if normalized and len(values) > 1:
        entropy /= math.log(len(values))
    return entropy


def top2_margin(probabilities: Sequence[float] | Mapping[str, float]) -> float:
    values = _probabilities(probabilities)
    if len(values) == 1:
        return float(values[0])
    top = np.partition(values, -2)[-2:]
    return float(top.max() - top.min())


@dataclass(frozen=True)
class UncertaintyScores:
    entropy: float
    normalized_entropy: float
    top2_margin: float


def uncertainty_scores(probabilities: Sequence[float] | Mapping[str, float]) -> UncertaintyScores:
    return UncertaintyScores(
        entropy=predictive_entropy(probabilities),
        normalized_entropy=predictive_entropy(probabilities, normalized=True),
        top2_margin=top2_margin(probabilities),
    )
