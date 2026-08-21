from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SelectiveLLMPolicy:
    entropy_threshold: float | None = None
    margin_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.entropy_threshold is None and self.margin_threshold is None:
            raise ValueError("at least one selective-LLM threshold is required")

    def decide(self, probabilities: Sequence[float]) -> dict:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or len(values) < 2 or np.any(values < 0) or not math.isclose(float(values.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("probabilities must be a normalized one-dimensional distribution")
        entropy = float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum())
        ordered = np.sort(values)[::-1]
        margin = float(ordered[0] - ordered[1])
        entropy_trigger = self.entropy_threshold is not None and entropy > self.entropy_threshold
        margin_trigger = self.margin_threshold is not None and margin < self.margin_threshold
        return {"call_llm": bool(entropy_trigger or margin_trigger), "entropy": entropy, "top2_margin": margin,
                "entropy_trigger": entropy_trigger, "margin_trigger": margin_trigger}
