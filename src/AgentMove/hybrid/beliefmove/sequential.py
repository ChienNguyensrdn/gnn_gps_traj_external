from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _normalize(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be a non-empty finite non-negative vector")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def fit_transition_matrix(
    sequences: Iterable[Sequence[str]],
    labels: Sequence[str],
    *,
    split: str,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Fit a first-order transition matrix from train trajectories only."""

    if split != "train":
        raise ValueError("transition matrix may only be fitted on split='train'")
    states = tuple(str(label) for label in labels)
    if not states or len(set(states)) != len(states):
        raise ValueError("labels must be unique and non-empty")
    smooth = float(smoothing)
    if not math.isfinite(smooth) or smooth < 0.0:
        raise ValueError("smoothing must be finite and non-negative")
    index = {label: position for position, label in enumerate(states)}
    counts = np.full((len(states), len(states)), smooth, dtype=float)
    for sequence in sequences:
        values = [str(value) for value in sequence]
        unknown = set(values) - set(states)
        if unknown:
            raise ValueError(f"transition sequence has unknown labels: {sorted(unknown)}")
        for previous, current in zip(values, values[1:]):
            counts[index[previous], index[current]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0.0
    if np.any(zero_rows):
        counts[zero_rows] = 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


class SequentialBeliefState:
    """Predict/update/reset state over a fixed train-fitted region vocabulary."""

    def __init__(
        self,
        labels: Sequence[str],
        transition: Sequence[Sequence[float]] | np.ndarray,
        *,
        initial: Sequence[float] | Mapping[str, float] | None = None,
    ) -> None:
        self.labels = tuple(str(label) for label in labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique and non-empty")
        self._index = {label: position for position, label in enumerate(self.labels)}
        matrix = np.asarray(transition, dtype=float)
        if matrix.shape != (len(self.labels), len(self.labels)) or np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("transition must be a finite non-negative square matrix matching labels")
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("transition rows must sum to one")
        self.transition = matrix.copy()
        if initial is None:
            initial_values = np.full(len(self.labels), 1.0 / len(self.labels), dtype=float)
        else:
            initial_values = self._vector(initial, name="initial")
        self._initial = initial_values.copy()
        self._belief = initial_values.copy()

    def _vector(self, values: Sequence[float] | Mapping[str, float], *, name: str) -> np.ndarray:
        if isinstance(values, Mapping):
            unknown = set(map(str, values)) - set(self.labels)
            if unknown:
                raise ValueError(f"{name} has unknown labels: {sorted(unknown)}")
            array = np.array([float(values.get(label, 0.0)) for label in self.labels], dtype=float)
        else:
            array = np.asarray(values, dtype=float)
            if array.shape != (len(self.labels),):
                raise ValueError(f"{name} length must match labels")
        return _normalize(array, name=name)

    @property
    def belief(self) -> np.ndarray:
        return self._belief.copy()

    def distribution(self) -> dict[str, float]:
        return {label: float(self._belief[index]) for index, label in enumerate(self.labels)}

    def predict(self) -> np.ndarray:
        self._belief = _normalize(self._belief @ self.transition, name="predicted belief")
        return self.belief

    def update(self, likelihood: Sequence[float] | Mapping[str, float]) -> np.ndarray:
        evidence = self._vector(likelihood, name="likelihood")
        posterior = self._belief * evidence
        if float(posterior.sum()) <= 0.0:
            posterior = self._belief
        self._belief = _normalize(posterior, name="posterior")
        return self.belief

    def step(self, likelihood: Sequence[float] | Mapping[str, float]) -> np.ndarray:
        self.predict()
        return self.update(likelihood)

    def reset(self, initial: Sequence[float] | Mapping[str, float] | None = None) -> np.ndarray:
        self._belief = self._initial.copy() if initial is None else self._vector(initial, name="initial")
        return self.belief
