from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=float) / max(float(temperature), 1e-8)
    values -= np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: Iterable[Sequence[float]], labels: Iterable[int]) -> "TemperatureScaler":
        matrix = np.asarray(list(logits), dtype=float)
        targets = np.asarray(list(labels), dtype=int)
        if matrix.ndim != 2 or len(matrix) != len(targets) or len(targets) == 0:
            raise ValueError("logits must be a non-empty 2D array aligned with labels")

        def nll(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            scaled = matrix / temperature
            scaled -= scaled.max(axis=1, keepdims=True)
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return float(-log_probs[np.arange(len(targets)), targets].mean())

        # Deterministic golden-section search avoids both scipy and the former
        # 2,001 full-matrix evaluations (prohibitively expensive for thousands
        # of Shanghai POIs).
        lower, upper = -4.0, 4.0
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)
        left_loss, right_loss = nll(left), nll(right)
        for _ in range(64):
            if left_loss <= right_loss:
                upper, right, right_loss = right, left, left_loss
                left = upper - ratio * (upper - lower); left_loss = nll(left)
            else:
                lower, left, left_loss = left, right, right_loss
                right = lower + ratio * (upper - lower); right_loss = nll(right)
        self.temperature = float(np.exp((lower + upper) / 2.0))
        return self

    def predict_proba(self, logits: Sequence[float]) -> np.ndarray:
        return softmax(logits, self.temperature)

    def to_dict(self) -> dict:
        return {"temperature": self.temperature}


class BinaryPlattCalibrator:
    """Calibrate a scalar compatibility score against candidate correctness."""

    def __init__(self) -> None:
        self.constant: float | None = None
        self.alpha: float | None = None
        self.beta: float | None = None

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "BinaryPlattCalibrator":
        x = np.asarray(list(scores), dtype=float).reshape(-1, 1)
        y = np.asarray(list(labels), dtype=int)
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("scores and labels must be non-empty and aligned")
        if len(np.unique(y)) < 2:
            self.constant = float((y.sum() + 1.0) / (len(y) + 2.0))
        else:
            self.constant = None
            values = x[:, 0]
            alpha, beta = 0.0, float(np.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6)))
            # Newton updates for one-dimensional logistic regression with L2
            # stabilization. Calibration sets are small, so this is sufficient.
            for _ in range(100):
                linear = np.clip(alpha * values + beta, -30.0, 30.0)
                probabilities = 1.0 / (1.0 + np.exp(-linear))
                weights = probabilities * (1.0 - probabilities)
                gradient = np.array([
                    np.sum((probabilities - y) * values) + 1e-6 * alpha,
                    np.sum(probabilities - y),
                ])
                hessian = np.array([
                    [np.sum(weights * values * values) + 1e-6, np.sum(weights * values)],
                    [np.sum(weights * values), np.sum(weights) + 1e-6],
                ])
                step = np.linalg.solve(hessian, gradient)
                alpha -= float(step[0])
                beta -= float(step[1])
                if np.linalg.norm(step) < 1e-8:
                    break
            self.alpha, self.beta = alpha, beta
        return self

    def predict(self, scores: Iterable[float]) -> np.ndarray:
        x = np.asarray(list(scores), dtype=float).reshape(-1, 1)
        if self.constant is not None:
            return np.full(len(x), self.constant, dtype=float)
        if self.alpha is None or self.beta is None:
            return np.clip(x[:, 0], 1e-6, 1 - 1e-6)
        linear = np.clip(self.alpha * x[:, 0] + self.beta, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-linear))

    def to_dict(self) -> dict:
        if self.constant is not None:
            return {"constant": self.constant}
        if self.alpha is None or self.beta is None:
            return {"alpha": 1.0, "beta": 0.0, "fitted": False}
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "fitted": True,
        }


class LikelihoodRatioCalibrator(BinaryPlattCalibrator):
    """Convert a discriminative Platt posterior into a Bayes likelihood ratio.

    Logistic calibration estimates P(Y=1|score). Bayesian fusion requires
    P(score|Y=1)/P(score|Y=0), so the training-set base odds must be removed.
    """

    def __init__(self, max_ratio: float = 20.0) -> None:
        super().__init__(); self.prevalence: float | None = None; self.max_ratio = max_ratio

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "LikelihoodRatioCalibrator":
        score_rows, label_rows = list(scores), list(labels)
        if not label_rows:
            raise ValueError("scores and labels must be non-empty and aligned")
        self.prevalence = float((sum(label_rows) + 1.0) / (len(label_rows) + 2.0))
        super().fit(score_rows, label_rows); return self

    def predict_likelihood_ratio(self, scores: Iterable[float]) -> np.ndarray:
        posterior = np.clip(super().predict(scores), 1e-6, 1 - 1e-6)
        prevalence = self.prevalence if self.prevalence is not None else 0.5
        base_odds = prevalence / max(1.0 - prevalence, 1e-6)
        ratios = (posterior / (1.0 - posterior)) / base_odds
        return np.clip(ratios, 1.0 / self.max_ratio, self.max_ratio)

    def to_dict(self) -> dict:
        result = super().to_dict(); result.update({
            "kind": "likelihood_ratio_from_platt", "prevalence": self.prevalence,
            "max_ratio": self.max_ratio,
        }); return result
