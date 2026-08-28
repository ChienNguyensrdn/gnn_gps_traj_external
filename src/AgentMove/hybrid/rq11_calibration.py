from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint_models import build_checkpoint_model
from .dual_evolution import _device
from .neural_cgm import _batches, _torch, build_examples
from .rq7_belief_memory import (fuse, sequence_queries, training_statistics,
                                transition_prior)


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))


def normalize_log_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores.astype(np.float64, copy=False) / temperature
    log_probabilities = scaled - _logsumexp(scaled)
    return np.exp(log_probabilities)


def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int, adaptive: bool) -> list[dict]:
    if bins < 2: raise ValueError("reliability bins must be at least 2")
    if adaptive:
        groups = np.array_split(np.argsort(confidence, kind="stable"), bins)
    else:
        edges = np.linspace(0.0, 1.0, bins + 1); groups = []
        for index in range(bins):
            upper = confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
            groups.append(np.flatnonzero((confidence >= edges[index]) & upper))
    rows = []
    for index, indices in enumerate(groups):
        rows.append({"bin": index, "count": int(len(indices)),
                     "confidence": float(confidence[indices].mean()) if len(indices) else None,
                     "accuracy": float(correct[indices].mean()) if len(indices) else None})
    return rows


def arrays_from_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty(len(labels), dtype=np.int32)
    for index, label in enumerate(labels): ranks[index] = int(np.where(order[index] == label)[0][0]) + 1
    confidence = probabilities.max(axis=1); top1 = order[:, 0]; true = probabilities[np.arange(len(labels)), labels]
    brier = np.sum(probabilities * probabilities, axis=1) - 2 * true + 1
    return {"labels": labels.astype(np.int64), "top1": top1.astype(np.int64), "ranks": ranks,
            "reciprocal_rank": (1.0 / ranks).astype(np.float32), "true_probability": true.astype(np.float32),
            "confidence": confidence.astype(np.float32), "top1_correct": (top1 == labels).astype(np.int8),
            "brier": brier.astype(np.float32)}


def summarize(arrays: dict[str, np.ndarray], bins: int) -> dict:
    confidence = arrays["confidence"].astype(float); correct = arrays["top1_correct"].astype(float)
    equal = reliability(confidence, correct, bins, False); adaptive = reliability(confidence, correct, bins, True)
    ece = sum(row["count"] * abs(row["accuracy"] - row["confidence"]) for row in equal if row["count"]) / len(confidence)
    aece = sum(row["count"] * abs(row["accuracy"] - row["confidence"]) for row in adaptive if row["count"]) / len(confidence)
    return {"queries": int(len(confidence)), "recall@1": float(np.mean(arrays["ranks"] <= 1)),
            "recall@5": float(np.mean(arrays["ranks"] <= 5)), "recall@10": float(np.mean(arrays["ranks"] <= 10)),
            "mrr": float(np.mean(arrays["reciprocal_rank"])),
            "nll": float(-np.mean(np.log(np.clip(arrays["true_probability"], 1e-12, 1)))),
            "brier": float(np.mean(arrays["brier"])), "ece": float(ece), "adaptive_ece": float(aece),
            "accuracy_confidence_gap": float(correct.mean() - confidence.mean()),
            "reliability_equal_width": equal, "reliability_equal_frequency": adaptive}


def last_query_batches(model, frame: pd.DataFrame, user_map: dict, batch_size: int, device, seed: int):
    examples = build_examples(frame, user_map, all_prefixes=False); torch = _torch()
    with torch.no_grad():
        for batch in _batches(examples, batch_size, False, seed):
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            yield model(poi, slots, lengths, users, targets).cpu().numpy(), labels.cpu().numpy()


def belief_batches(model, frame: pd.DataFrame, user_map: dict, batch_size: int, device, seed: int,
                   variant: str, weight: float, global_prior: np.ndarray, transitions, smoothing: float):
    queries = sequence_queries(frame, user_map); torch = _torch()
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        examples = [(q["pois"][:q["step"]], q["slots"][:q["step"]], q["user"], q["target_slot"], q["label"]) for q in chunk]
        batch = next(_batches(examples, len(examples), False, seed))
        with torch.no_grad():
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            logits = model(poi, slots, lengths, users, targets).cpu().numpy()
        bases = normalize_log_scores(logits, 1.0); probabilities = np.empty_like(bases)
        for index, query in enumerate(chunk):
            if variant == "B0-static" or weight == 0.0: probabilities[index] = bases[index]
            elif variant == "B3-dbn":
                prior = transition_prior(query["pois"][query["step"] - 1], transitions, global_prior, smoothing)
                probabilities[index] = fuse(bases[index], prior, weight)
            else: raise ValueError(f"RQ11 supports B0-static or B3-dbn, got {variant}")
        yield np.log(np.clip(probabilities, 1e-12, 1.0)), labels.cpu().numpy()


def fit_temperature(batches, temperatures: list[float]) -> float:
    losses = np.zeros(len(temperatures)); count = 0
    for scores, labels in batches:
        count += len(labels)
        for index, temperature in enumerate(temperatures):
            scaled = scores / temperature; logp = scaled - _logsumexp(scaled)
            losses[index] -= logp[np.arange(len(labels)), labels].sum()
    if count == 0: raise ValueError("validation split produced zero calibration examples")
    return float(temperatures[int(np.argmin(losses / count))])


def test_outputs(batches, temperature: float) -> tuple[dict, dict]:
    before_rows, after_rows = [], []
    for scores, batch_labels in batches:
        before_rows.append(arrays_from_probabilities(normalize_log_scores(scores, 1.0), batch_labels))
        after_rows.append(arrays_from_probabilities(normalize_log_scores(scores, temperature), batch_labels))
    if not before_rows: raise ValueError("test split produced zero calibration examples")
    combine = lambda rows: {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    return combine(before_rows), combine(after_rows)


def evaluate(args):
    torch = _torch(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_checkpoint_model(checkpoint); model.load_state_dict(checkpoint["model_state"])
    device = _device(torch, args.device); model.to(device).eval()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")); temperatures = [float(x) for x in config["temperature_grid"]]
    validation = pd.read_csv(args.validation_csv); test = pd.read_csv(args.test_csv)
    protocol_details = {}; train = None
    if args.protocol == "last-query":
        factory = lambda frame: last_query_batches(model, frame, checkpoint["user_map"], args.batch_size, device, args.seed)
    else:
        if not args.train_csv or not args.rq7_metrics: raise ValueError("all-prefix requires --train-csv and --rq7-metrics")
        train = pd.read_csv(args.train_csv); rq7 = json.loads(Path(args.rq7_metrics).read_text(encoding="utf-8"))
        if rq7.get("fit_splits") != ["train", "validation"] or rq7.get("evaluation_split") != "test":
            raise ValueError("RQ7 artifact violates split protocol")
        weight = float(rq7["selected_weights"][args.variant]); smoothing = float(config["belief_smoothing"])
        prior, transitions = training_statistics(train, int(checkpoint["config"]["num_pois"]), smoothing)
        factory = lambda frame: belief_batches(model, frame, checkpoint["user_map"], args.batch_size, device,
                                               args.seed, args.variant, weight, prior, transitions, smoothing)
        protocol_details = {"belief_weight": weight, "belief_weight_source": "RQ7 validation", "belief_statistics_fit": "train"}
    temperature = fit_temperature(factory(validation), temperatures)
    before, after = test_outputs(factory(test), temperature)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "before.predictions.npz", **before, query_index=np.arange(len(before["labels"])))
    np.savez_compressed(output / "after.predictions.npz", **after, query_index=np.arange(len(after["labels"])))
    result = {"rq": "RQ11", "variant": args.variant, "protocol": args.protocol, "seed": args.seed,
              "calibrator": "temperature_scaling", "temperature": temperature,
              "temperature_fit_split": "validation", "evaluation_split": "test",
              "metrics_before": summarize(before, int(config["reliability_bins"])),
              "metrics_after": summarize(after, int(config["reliability_bins"])), **protocol_details}
    (output / "rq11.metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "temperature": temperature,
                      "nll_before": result["metrics_before"]["nll"], "nll_after": result["metrics_after"]["nll"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description="Validation-only temperature calibration for RQ11")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True); parser.add_argument("--protocol", choices=["last-query", "all-prefix"], required=True)
    parser.add_argument("--train-csv"); parser.add_argument("--rq7-metrics")
    parser.add_argument("--config", default="configs/beliefmove_evo/calibration.json")
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42); evaluate(parser.parse_args())


if __name__ == "__main__": main()
