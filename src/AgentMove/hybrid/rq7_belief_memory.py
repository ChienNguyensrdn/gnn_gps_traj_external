from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .dual_evolution import _device
from .metrics import expected_calibration_error
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
            queries.append({"trajectory_id": str(trajectory_id), "step": stop,
                            "pois": pois, "slots": slots, "user": user,
                            "target_slot": slots[stop], "label": pois[stop]})
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
    output = np.empty_like(base); previous = None; previous_trajectory = None
    for index, query in enumerate(queries):
        trajectory = query["trajectory_id"]
        prefix = query.get("prefix") or query["pois"][:query["step"]]
        if variant == "B0-static": evidence = global_prior; current = base[index]
        elif variant == "B1-history":
            evidence = smoothed_counts(prefix, global_prior, smoothing)
            current = fuse(base[index], evidence, weight)
        elif variant == "B2-sequential":
            evidence = previous if trajectory == previous_trajectory else global_prior
            current = fuse(base[index], evidence, weight)
        elif variant == "B3-dbn":
            evidence = transition_prior(prefix[-1], transitions, global_prior, smoothing)
            current = fuse(base[index], evidence, weight)
        else: raise ValueError(f"unknown RQ7 variant: {variant}")
        output[index] = current; previous = current; previous_trajectory = trajectory
    return output


def _examples(queries):
    return [(query["pois"][:query["step"]], query["slots"][:query["step"]], query["user"],
             query["target_slot"], query["label"]) for query in queries]


def stream_variants(model, queries, specifications, batch_size, device, seed,
                    global_prior, transitions, smoothing):
    """Run the backbone once and retain only scalar per-query statistics."""
    count = len(queries)
    outputs = {}
    states = {}
    for variant, weight in specifications:
        outputs[(variant, weight)] = {
            "labels": np.empty(count, dtype=np.int64), "top1": np.empty(count, dtype=np.int64),
            "ranks": np.empty(count, dtype=np.int32), "reciprocal_rank": np.empty(count, dtype=np.float32),
            "true_probability": np.empty(count, dtype=np.float32), "confidence": np.empty(count, dtype=np.float32),
            "top1_correct": np.empty(count, dtype=np.int8), "brier": np.empty(count, dtype=np.float32),
        }
        states[(variant, weight)] = (None, None)
    torch = _torch()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            chunk = queries[start:start + batch_size]
            batch = next(_batches(_examples(chunk), len(chunk), False, seed))
            poi, slots, lengths, users, targets, _ = [value.to(device) for value in batch]
            logits = model(poi, slots, lengths, users, targets).cpu().numpy()
            logits -= logits.max(axis=1, keepdims=True); bases = np.exp(logits)
            bases /= bases.sum(axis=1, keepdims=True)
            for local_index, query in enumerate(chunk):
                index = start + local_index; base = bases[local_index].astype(np.float64, copy=False)
                trajectory = query["trajectory_id"]; prefix = query["pois"][:query["step"]]
                history = None; transition = None
                for variant, weight in specifications:
                    previous, previous_trajectory = states[(variant, weight)]
                    if variant == "B0-static": current = base
                    elif variant == "B1-history":
                        if history is None: history = smoothed_counts(prefix, global_prior, smoothing)
                        current = fuse(base, history, weight)
                    elif variant == "B2-sequential":
                        evidence = previous if trajectory == previous_trajectory else global_prior
                        current = fuse(base, evidence, weight)
                    elif variant == "B3-dbn":
                        if transition is None:
                            transition = transition_prior(prefix[-1], transitions, global_prior, smoothing)
                        current = fuse(base, transition, weight)
                    else: raise ValueError(f"unknown RQ7 variant: {variant}")
                    label = query["label"]; top1 = int(np.argmax(current)); true = current[label]
                    rank = 1 + int(np.count_nonzero(current > true)) + int(np.count_nonzero(current[:label] == true))
                    arrays = outputs[(variant, weight)]
                    arrays["labels"][index] = label; arrays["top1"][index] = top1; arrays["ranks"][index] = rank
                    arrays["reciprocal_rank"][index] = 1.0 / rank; arrays["true_probability"][index] = true
                    arrays["confidence"][index] = current[top1]; arrays["top1_correct"][index] = top1 == label
                    arrays["brier"][index] = np.dot(current, current) - 2 * true + 1
                    states[(variant, weight)] = (current, trajectory)
    return outputs


def variant_arrays(base, queries, variant, weight, global_prior, transitions, smoothing):
    count = len(queries); labels = np.empty(count, dtype=np.int64); ranks = np.empty(count, dtype=np.int32)
    top1 = np.empty(count, dtype=np.int64); confidence = np.empty(count, dtype=np.float32)
    true_probability = np.empty(count, dtype=np.float32); brier = np.empty(count, dtype=np.float32)
    previous = None; previous_trajectory = None
    for index, query in enumerate(queries):
        trajectory = query["trajectory_id"]; prefix = query["pois"][:query["step"]]
        if variant == "B0-static": current = np.asarray(base[index], dtype=np.float64)
        elif variant == "B1-history":
            current = fuse(base[index], smoothed_counts(prefix, global_prior, smoothing), weight)
        elif variant == "B2-sequential":
            evidence = previous if trajectory == previous_trajectory else global_prior
            current = fuse(base[index], evidence, weight)
        elif variant == "B3-dbn":
            current = fuse(base[index], transition_prior(prefix[-1], transitions, global_prior, smoothing), weight)
        else: raise ValueError(f"unknown RQ7 variant: {variant}")
        label = query["label"]; labels[index] = label; top1[index] = int(np.argmax(current))
        confidence[index] = current[top1[index]]; true_probability[index] = current[label]
        # Stable descending rank without allocating a full argsort matrix.
        ranks[index] = 1 + int(np.count_nonzero(current > current[label])) + int(
            np.count_nonzero(current[:label] == current[label]))
        brier[index] = np.dot(current, current) - 2 * current[label] + 1
        previous = current; previous_trajectory = trajectory
    return {"labels": labels, "top1": top1, "ranks": ranks,
            "reciprocal_rank": (1.0 / ranks).astype(np.float32),
            "true_probability": true_probability, "confidence": confidence,
            "top1_correct": (top1 == labels).astype(np.int8), "brier": brier}


def summarize_arrays(arrays):
    ranks = arrays["ranks"]
    return {"queries": int(len(ranks)), "recall@1": float(np.mean(ranks <= 1)),
            "recall@5": float(np.mean(ranks <= 5)), "recall@10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(arrays["reciprocal_rank"])),
            "nll": float(-np.mean(np.log(np.clip(arrays["true_probability"], 1e-12, 1)))),
            "brier": float(np.mean(arrays["brier"])),
            "ece": expected_calibration_error(arrays["confidence"], arrays["top1_correct"])}


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
    del train, validation, test
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    # Remove caches left by the pre-streaming evaluator after an OOM/SIGBUS.
    for stale in (output_dir / ".validation_base.npy", output_dir / ".test_base.npy"):
        stale.unlink(missing_ok=True)
    selected = {"B0-static": 0.0}; validation_metrics = {}; test_metrics = {}
    validation_specs = [(variant, weight) for variant in VARIANTS
                        for weight in ([0.0] if variant == "B0-static" else grid)]
    validation_outputs = stream_variants(model, validation_queries, validation_specs, args.batch_size,
                                         device, args.seed, global_prior, transitions, smoothing)
    for variant in VARIANTS:
        candidates = [0.0] if variant == "B0-static" else grid; values = []
        for weight in candidates:
            metrics = summarize_arrays(validation_outputs[(variant, weight)])
            values.append((metrics["recall@1"] + metrics["recall@10"], weight, metrics))
        _, weight, metrics = max(values, key=lambda item: (item[0], -item[1]))
        selected[variant] = weight; validation_metrics[variant] = metrics
    del validation_outputs, validation_queries
    test_specs = [(variant, selected[variant]) for variant in VARIANTS]
    test_outputs = stream_variants(model, test_queries, test_specs, args.batch_size, device, args.seed,
                                   global_prior, transitions, smoothing)
    for variant, weight in test_specs:
        arrays = test_outputs[(variant, weight)]; test_metrics[variant] = summarize_arrays(arrays)
        np.savez_compressed(output_dir / f"{variant}.test.predictions.npz", **arrays,
                            query_index=np.arange(len(test_queries)),
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
