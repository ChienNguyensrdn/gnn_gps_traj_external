from __future__ import annotations

import numpy as np


def linear_cka(left, right) -> float:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be 2-D with the same sample count")
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denominator = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / denominator) if denominator else 0.0


def transition_cosine(left, right) -> float:
    x, y = np.diff(np.asarray(left, dtype=float), axis=1), np.diff(np.asarray(right, dtype=float), axis=1)
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError("transition tensors must have equal shape [batch,time,...]")
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    valid = denominator > 0
    return float(np.mean(np.sum(x[valid] * y[valid], axis=1) / denominator[valid])) if np.any(valid) else 0.0
