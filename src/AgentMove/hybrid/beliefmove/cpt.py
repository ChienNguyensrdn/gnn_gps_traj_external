from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np


def _context_key(value: object) -> Hashable:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _context_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_context_key(item) for item in value)
    if not isinstance(value, Hashable):
        raise ValueError("CPT context must be hashable or recursively serializable")
    return value


def _labels(labels: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(label) for label in labels)
    if not result or len(set(result)) != len(result) or any(not label for label in result):
        raise ValueError("labels must be unique non-empty strings")
    return result


class _BaseCPT:
    def __init__(self, labels: Sequence[str], *, smoothing: float = 1.0) -> None:
        self.labels = _labels(labels)
        self._index = {label: index for index, label in enumerate(self.labels)}
        self.smoothing = float(smoothing)
        if not math.isfinite(self.smoothing) or self.smoothing < 0.0:
            raise ValueError("smoothing must be finite and non-negative")
        self._counts: dict[Hashable, np.ndarray] = {}
        self._global = np.zeros(len(self.labels), dtype=float)

    def _reset(self) -> None:
        self._counts = {}
        self._global = np.zeros(len(self.labels), dtype=float)

    def _add(self, context: object, values: np.ndarray, weight: float = 1.0) -> None:
        key = _context_key(context)
        if key not in self._counts:
            self._counts[key] = np.zeros(len(self.labels), dtype=float)
        self._counts[key] += weight * values
        self._global += weight * values

    def _probabilities(self, counts: np.ndarray) -> np.ndarray:
        values = counts.astype(float, copy=True) + self.smoothing
        total = float(values.sum())
        if total <= 0.0:
            return np.full(len(self.labels), 1.0 / len(self.labels), dtype=float)
        return values / total

    def probabilities(self, context: object) -> np.ndarray:
        key = _context_key(context)
        counts = self._counts.get(key, self._global)
        return self._probabilities(counts)

    def distribution(self, context: object) -> dict[str, float]:
        probabilities = self.probabilities(context)
        return {label: float(probabilities[index]) for index, label in enumerate(self.labels)}


class DataCPT(_BaseCPT):
    """Categorical CPT fitted from hard next-region labels in train only."""

    def fit(self, observations: Iterable[tuple[object, str]], *, split: str) -> "DataCPT":
        if split != "train":
            raise ValueError("DataCPT may only be fitted on split='train'")
        self._reset()
        for context, raw_label in observations:
            label = str(raw_label)
            if label not in self._index:
                raise ValueError(f"unknown CPT label: {label!r}")
            values = np.zeros(len(self.labels), dtype=float)
            values[self._index[label]] = 1.0
            self._add(context, values)
        return self


class TeacherSoftCountCPT(_BaseCPT):
    """CPT distilled from normalized teacher region posteriors as soft counts."""

    def fit(
        self,
        observations: Iterable[tuple[object, Mapping[str, float]] | tuple[object, Mapping[str, float], float]],
        *,
        split: str,
    ) -> "TeacherSoftCountCPT":
        if split != "train":
            raise ValueError("TeacherSoftCountCPT may only be fitted on split='train'")
        self._reset()
        for observation in observations:
            if len(observation) == 2:
                context, distribution = observation
                weight = 1.0
            else:
                context, distribution, raw_weight = observation
                weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("soft-count weight must be finite and non-negative")
            unknown = set(map(str, distribution)) - set(self.labels)
            if unknown:
                raise ValueError(f"teacher distribution has unknown labels: {sorted(unknown)}")
            values = np.array([float(distribution.get(label, 0.0)) for label in self.labels], dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
                raise ValueError("teacher distributions must be finite, non-negative and normalized")
            self._add(context, values, weight)
        return self


@dataclass(frozen=True)
class AlphaSelection:
    alpha: float
    mean_nll: float
    validation_size: int
    scores: Mapping[float, float]


class BlendedCPT:
    """Convex blend ``alpha*data + (1-alpha)*teacher``.

    This follows the notation locked in the BeliefMove design document.  Thus
    ``alpha=1`` is the data-only endpoint and ``alpha=0`` is the teacher-only
    endpoint.  Alpha may only be selected on validation data.
    """

    def __init__(self, data_cpt: DataCPT, teacher_cpt: TeacherSoftCountCPT, *, alpha: float = 1.0) -> None:
        if data_cpt.labels != teacher_cpt.labels:
            raise ValueError("data and teacher CPTs must use identical label order")
        self.data_cpt = data_cpt
        self.teacher_cpt = teacher_cpt
        self.labels = data_cpt.labels
        self.alpha = self._validate_alpha(alpha)

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = float(alpha)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be finite and in [0,1]")
        return value

    def probabilities(self, context: object, *, alpha: float | None = None) -> np.ndarray:
        weight = self.alpha if alpha is None else self._validate_alpha(alpha)
        values = weight * self.data_cpt.probabilities(context) + (1.0 - weight) * self.teacher_cpt.probabilities(context)
        return values / values.sum()

    def distribution(self, context: object, *, alpha: float | None = None) -> dict[str, float]:
        values = self.probabilities(context, alpha=alpha)
        return {label: float(values[index]) for index, label in enumerate(self.labels)}

    def select_alpha(
        self,
        validation: Iterable[tuple[object, str]],
        *,
        split: str,
        grid: Sequence[float] | None = None,
    ) -> AlphaSelection:
        if split != "validation":
            raise ValueError("alpha may only be selected on split='validation'")
        rows = list(validation)
        if not rows:
            raise ValueError("validation observations must not be empty")
        candidates = tuple(grid if grid is not None else (0.0, 0.25, 0.5, 0.75, 1.0))
        if not candidates:
            raise ValueError("alpha grid must not be empty")
        scores: dict[float, float] = {}
        for raw_alpha in candidates:
            alpha = self._validate_alpha(raw_alpha)
            losses: list[float] = []
            for context, raw_label in rows:
                label = str(raw_label)
                if label not in self.data_cpt._index:
                    raise ValueError(f"unknown validation label: {label!r}")
                probability = self.probabilities(context, alpha=alpha)[self.data_cpt._index[label]]
                losses.append(-math.log(max(float(probability), 1e-12)))
            scores[alpha] = float(np.mean(losses))
        best_alpha = min(scores, key=lambda candidate: (scores[candidate], candidate))
        self.alpha = best_alpha
        return AlphaSelection(best_alpha, scores[best_alpha], len(rows), dict(sorted(scores.items())))
