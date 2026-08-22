from __future__ import annotations

import numpy as np


class SequentialBelief:
    def __init__(self, initial, transition) -> None:
        self.transition = self._matrix(transition)
        self.belief = self._normalize(initial)
        if self.transition.shape != (len(self.belief), len(self.belief)):
            raise ValueError("transition shape must match belief state count")
        row_sums = self.transition.sum(axis=1)
        if np.any(self.transition < 0) or not np.allclose(row_sums, 1.0):
            raise ValueError("transition rows must be normalized probabilities")

    @staticmethod
    def _matrix(values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("transition must be a finite matrix")
        return matrix

    @staticmethod
    def _normalize(values):
        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1 or np.any(vector < 0) or not np.all(np.isfinite(vector)) or vector.sum() <= 0:
            raise ValueError("belief/evidence must be a non-negative finite vector")
        return vector / vector.sum()

    def predict(self):
        self.belief = self._normalize(self.belief @ self.transition)
        return self.belief.copy()

    def update(self, likelihood):
        likelihood = self._normalize(likelihood)
        self.belief = self._normalize(self.belief * likelihood)
        return self.belief.copy()

    def step(self, likelihood):
        self.predict()
        return self.update(likelihood)
