from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .schemas import Prediction


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[int], bins: int = 15) -> float:
    confidence = np.asarray(confidences, dtype=float)
    outcomes = np.asarray(correct, dtype=float)
    if len(confidence) == 0 or len(confidence) != len(outcomes):
        raise ValueError("confidence and correctness arrays must be non-empty and aligned")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & ((confidence < upper) if index < bins - 1 else (confidence <= upper))
        if mask.any():
            result += float(mask.mean() * abs(outcomes[mask].mean() - confidence[mask].mean()))
    return result


def summarize(predictions: Iterable[Prediction], bins: int = 15) -> Dict[str, float]:
    rows = list(predictions)
    if not rows:
        raise ValueError("No predictions to summarize")
    ranks: List[int | None] = []
    confidence: List[float] = []
    correct: List[int] = []
    nll: List[float] = []
    brier: List[float] = []
    for row in rows:
        rank = row.ranking.index(row.true_id) + 1 if row.true_id in row.ranking else None
        ranks.append(rank)
        top_correct = int(bool(row.ranking) and row.ranking[0] == row.true_id)
        top_confidence = row.probabilities[0] if row.probabilities else 0.0
        confidence.append(top_confidence)
        correct.append(top_correct)
        true_probability = row.probabilities[rank - 1] if rank is not None else 1e-12
        nll.append(-np.log(max(true_probability, 1e-12)))
        probability_by_id = dict(zip(row.ranking, row.probabilities))
        brier.append(sum((probability - float(candidate_id == row.true_id)) ** 2 for candidate_id, probability in probability_by_id.items()))
    llm_latency = [sum(item.latency_seconds for item in row.evidence) for row in rows]
    output: Dict[str, float] = {
        "queries": float(len(rows)),
        "acc@1": float(np.mean([rank == 1 for rank in ranks])),
        "acc@5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
        "ndcg@5": float(np.mean([
            1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0
            for rank in ranks
        ])),
        "acc@10": float(np.mean([rank is not None and rank <= 10 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank if rank else 0.0 for rank in ranks])),
        "candidate_recall": float(np.mean([row.true_id in row.candidate_ids for row in rows])),
        "ece": expected_calibration_error(confidence, correct, bins),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "input_tokens_mean": float(np.mean([row.input_tokens for row in rows])),
        "output_tokens_mean": float(np.mean([row.output_tokens for row in rows])),
        "api_calls_mean": float(np.mean([row.api_calls for row in rows])),
        "invalid_evidence_rate": float(
            np.mean([not item.valid for row in rows for item in row.evidence])
        ) if any(row.evidence for row in rows) else 0.0,
        "latency_mean": float(np.mean([row.timings["total_seconds"] for row in rows])),
        "latency_median": float(np.median([row.timings["total_seconds"] for row in rows])),
        "latency_p95": float(np.quantile([row.timings["total_seconds"] for row in rows], 0.95)),
        # Cached ablations replay the original Evidence records. These values
        # therefore retain measured model-call latency instead of timing the
        # fast cache lookup performed while generating predictions.
        "llm_latency_mean": float(np.mean(llm_latency)),
        "llm_latency_median": float(np.median(llm_latency)),
        "llm_latency_p95": float(np.quantile(llm_latency, 0.95)),
    }
    return output


def grouped_summary(predictions: Iterable[Prediction], field: str, bins: int = 15) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Prediction]] = defaultdict(list)
    for prediction in predictions:
        groups[str(getattr(prediction, field) or "unknown")].append(prediction)
    return {name: summarize(rows, bins) for name, rows in sorted(groups.items())}


def paired_bootstrap_delta(
    first: Iterable[Prediction], second: Iterable[Prediction], metric: str = "acc@1", samples: int = 2000, seed: int = 42
) -> Dict[str, float]:
    a = {row.query_id: row for row in first}
    b = {row.query_id: row for row in second}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("Paired runs have no common query_id")
    rng = np.random.default_rng(seed)
    deltas = []
    if metric == "acc@1":
        first_values = np.asarray([float(bool(a[item].ranking) and a[item].ranking[0] == a[item].true_id) for item in ids])
        second_values = np.asarray([float(bool(b[item].ranking) and b[item].ranking[0] == b[item].true_id) for item in ids])
        paired_delta = first_values - second_values
        for _ in range(samples):
            selected = rng.integers(0, len(ids), size=len(ids))
            deltas.append(float(paired_delta[selected].mean()))
        observed = float(paired_delta.mean())
    else:
        for _ in range(samples):
            selected = rng.choice(ids, size=len(ids), replace=True)
            deltas.append(summarize([a[item] for item in selected])[metric] - summarize([b[item] for item in selected])[metric])
        observed = summarize([a[item] for item in ids])[metric] - summarize([b[item] for item in ids])[metric]
    values = np.asarray(deltas)
    return {
        "metric": metric,
        "paired_queries": float(len(ids)),
        "delta": float(observed),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_delta_le_zero": float(np.mean(values <= 0.0)),
    }
