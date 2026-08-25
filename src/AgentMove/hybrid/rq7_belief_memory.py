from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .dual_evolution import _device
from .evaluate_student import prediction_arrays, summarize_logits
from .neural_cgm import ModelConfig, _batches, _slot, _torch, build_model


VARIANTS = ("B0-static", "B1-history", "B2-sequential", "B3-dbn")


def sequence_queries(frame: pd.DataFrame, user_map: dict[str, int]):
    queries = []
    for trajectory_id, rows in frame.groupby("trajectory_id", sort=False):
        rows = rows.sort_values("UTC_time", kind="stable")
        pois = rows["POI_id"].astype(int).tolist(); slots = _slot(rows["UTC_time"])
        if len(pois) < 2: continue
        user = user_map.get(str(rows.iloc[0]["user_id"]), len(user_map))
        for stop in range(1, len(pois)):
            example = (pois[:stop], slots[:stop], user, slots[stop], pois[stop])
            queries.append({"trajectory_id": str(trajectory_id), "step": stop,
                            "prefix": pois[:stop], "example": example})
    return queries


def training_statistics(frame: pd.DataFrame, size: int, smoothing: float):
    destination = np.full(size, smoothing, dtype=np.float64)
    transitions: dict[int, Counter] = defaultdict(Counter)
    for _, rows in frame.groupby("trajectory_id", sort=False):
        pois = rows.sort_values("UTC_time", kind="stable")["POI_id"].astype(int).tolist()
        for poi in pois:
            if 0 <= poi < size: destination[poi] += 1
        for source, target in zip(pois, pois[1:]):
            if 0 <= source < size and 0 <= target < size: transitions[source][target] += 1
    destination /= destination.sum()
    return destination, transitions


def smoothed_counts(indices, global_prior: np.ndarray, smoothing: float) -> np.ndarray:
    result = smoothing * global_prior.copy()
    if indices:
        unique, counts = np.unique(np.asarray(indices, dtype=int), return_counts=True)
        valid = (unique >= 0) & (unique < len(result)); result[unique[valid]] += counts[valid]
    return result / result.sum()


def transition_prior(source: int, transitions, global_prior: np.ndarray, smoothing: float) -> np.ndarray:
    row = transitions.get(source, {})
    result = smoothing * global_prior.copy()
    for target, count in row.items(): result[target] += count
    return result / result.sum()


def fuse(base: np.ndarray, evidence: np.ndarray, weight: float) -> np.ndarray:
    logp = np.log(np.clip(base, 1e-12, 1.0)) + weight * np.log(np.clip(evidence, 1e-12, 1.0))
    logp -= logp.max(); result = np.exp(logp)
    return result / result.sum()


def apply_variant(base: np.ndarray, queries: list[dict], variant: str, weight: float,
                  global_prior: np.ndarray, transitions, smoothing: float) -> np.ndarray:
    output = np.empty_like(base); previous: dict[str, np.ndarray] = {}
    for index, query in enumerate(queries):
        trajectory = query["trajectory_id"]
        if variant == "B0-static": evidence = global_prior; current = base[index]
        elif variant == "B1-history":
            evidence = smoothed_counts(query["prefix"], global_prior, smoothing)
            current = fuse(base[index], evidence, weight)
        elif variant == "B2-sequential":
            evidence = previous.get(trajectory, global_prior)
            current = fuse(base[index], evidence, weight)
        elif variant == "B3-dbn":
            evidence = transition_prior(query["prefix"][-1], transitions, global_prior, smoothing)
            current = fuse(base[index], evidence, weight)
        else: raise ValueError(f"unknown RQ7 variant: {variant}")
        output[index] = current; previous[trajectory] = current
    return output


def infer(model, queries, batch_size, device, seed):
    examples = [query["example"] for query in queries]; logits = []
    torch = _torch()
    with torch.no_grad():
        for batch in _batches(examples, batch_size, False, seed):
            poi, slots, lengths, users, targets, _ = [value.to(device) for value in batch]
            logits.append(model(poi, slots, lengths, users, targets).cpu().numpy())
    if not logits: raise ValueError("split produced zero sequential queries")
    logits = np.concatenate(logits); logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.asarray([query["example"][-1] for query in queries], dtype=np.int64)
    return probabilities, labels


def score(probabilities, labels) -> float:
    metrics = summarize_logits(np.log(np.clip(probabilities, 1e-12, 1.0)), labels)
    return metrics["recall@1"] + metrics["recall@10"]


def evaluate(args):
    torch = _torch(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**checkpoint["config"])); model.load_state_dict(checkpoint["model_state"])
    device = _device(torch, args.device); model.to(device).eval()
    config = json.loads(Path(args.config).read_text()); grid = [float(x) for x in config["weight_grid"]]
    smoothing = float(config["smoothing"]); size = int(checkpoint["config"]["num_pois"])
    train = pd.read_csv(args.train_csv); validation = pd.read_csv(args.validation_csv); test = pd.read_csv(args.test_csv)
    global_prior, transitions = training_statistics(train, size, smoothing)
    validation_queries = sequence_queries(validation, checkpoint["user_map"])
    test_queries = sequence_queries(test, checkpoint["user_map"])
    validation_base, validation_labels = infer(model, validation_queries, args.batch_size, device, args.seed)
    test_base, test_labels = infer(model, test_queries, args.batch_size, device, args.seed)
    selected = {"B0-static": 0.0}; validation_metrics = {}; test_metrics = {}
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        candidates = [0.0] if variant == "B0-static" else grid
        values = [(score(apply_variant(validation_base, validation_queries, variant, weight,
                                       global_prior, transitions, smoothing), validation_labels), weight)
                  for weight in candidates]
        weight = max(values, key=lambda item: (item[0], -item[1]))[1]; selected[variant] = weight
        validation_probs = apply_variant(validation_base, validation_queries, variant, weight,
                                         global_prior, transitions, smoothing)
        test_probs = apply_variant(test_base, test_queries, variant, weight,
                                   global_prior, transitions, smoothing)
        validation_metrics[variant] = summarize_logits(np.log(np.clip(validation_probs, 1e-12, 1)), validation_labels)
        test_metrics[variant] = summarize_logits(np.log(np.clip(test_probs, 1e-12, 1)), test_labels)
        arrays = prediction_arrays(np.log(np.clip(test_probs, 1e-12, 1)), test_labels)
        np.savez_compressed(output_dir / f"{variant}.test.predictions.npz", **arrays,
                            query_index=np.arange(len(test_labels)),
                            trajectory_id=np.asarray([q["trajectory_id"] for q in test_queries]),
                            step=np.asarray([q["step"] for q in test_queries], dtype=np.int32))
    result = {"rq": "RQ7", "seed": args.seed, "device": str(device), "checkpoint": str(Path(args.checkpoint).resolve()),
              "protocol": "all chronological prefixes; reset belief per trajectory",
              "fit_splits": ["train", "validation"], "evaluation_split": "test",
              "selected_weights": selected, "validation_metrics": validation_metrics, "test_metrics": test_metrics}
    path = output_dir / "rq7.metrics.json"; path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path), "selected_weights": selected, "test_metrics": test_metrics}))
    return result


def main():
    parser = argparse.ArgumentParser(description="RQ7 belief-memory evaluation on a frozen E5-dual checkpoint")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True); parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/beliefmove_evo/belief_memory.json")
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())


if __name__ == "__main__": main()
