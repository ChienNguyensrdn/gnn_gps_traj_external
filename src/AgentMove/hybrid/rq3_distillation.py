from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .neural_cgm import _slot
from .rq2_data_only import empty_arrays, fill, normalize, sparse_prior
from .rq7_belief_memory import summarize_arrays
from .rq8_routing import read_jsonl


VARIANTS = ("M1-data-only", "M2-llm", "M3-quantitative", "M4-both")


def resolve_bundle(path: Path, value: str) -> Path:
    target = Path(value)
    if target.is_file():
        return target
    for candidate in (path.parent / target.name, path.parent.parent / target.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing bundle artifact: {target}")


def load_split(path: Path, limit: int | None):
    rows = read_jsonl(path); bundle = rows.pop(0)["_bundle"]
    if limit is not None:
        rows = rows[:limit]
    ids = [str(value) for value in json.loads(resolve_bundle(path, bundle["candidate_ids"]).read_text())]
    logits = np.load(resolve_bundle(path, bundle["logits"]), mmap_mode="r")
    return rows, logits, ids


def fit_bn_statistics(frame: pd.DataFrame, size: int, alpha: float):
    global_counts = np.full(size, alpha)
    users: dict[str, Counter] = defaultdict(Counter); times: dict[int, Counter] = defaultdict(Counter)
    slots = _slot(frame["UTC_time"]); matched = 0
    for row, slot in zip(frame.itertuples(index=False), slots):
        try:
            poi = int(row.POI_id)
        except (TypeError, ValueError):
            continue
        if poi < 0 or poi >= size:
            continue
        matched += 1; user = str(row.user_id); global_counts[poi] += 1; users[user][poi] += 1; times[slot][poi] += 1
    if matched == 0:
        raise ValueError("no train POI_id matches the hybrid candidate space")
    return normalize(global_counts), users, times, matched


def platt_likelihood_ratio(scores, config):
    values = np.asarray(scores, dtype=float)
    if "constant" in config:
        posterior = np.full(values.shape, float(config["constant"]))
    else:
        alpha = float(config.get("alpha", 1.0)); beta = float(config.get("beta", 0.0))
        linear = np.clip(alpha * values + beta, -30.0, 30.0)
        posterior = 1.0 / (1.0 + np.exp(-linear))
    posterior = np.clip(posterior, 1e-6, 1 - 1e-6)
    prevalence = float(config.get("prevalence") or 0.5)
    base_odds = prevalence / max(1.0 - prevalence, 1e-6)
    maximum = float(config.get("max_ratio", 20.0))
    return np.clip((posterior / (1.0 - posterior)) / base_odds, 1.0 / maximum, maximum)


def evidence_index(path: Path):
    result = {}
    for row in read_jsonl(path):
        evidence = row["evidence"]
        result[(str(row["query_id"]), str(evidence["candidate_id"]))] = evidence
    return result


def source_probabilities(rows, logits, ids, global_prior, users, times, evidence, calibration, alpha):
    size = len(ids); temperature = float(calibration["temperature"]["temperature"])
    quantitative = np.empty((len(rows), size)); bn = np.empty_like(quantitative); llm = np.ones_like(quantitative)
    labels = np.empty(len(rows), dtype=np.int64); valid_evidence = 0; total_evidence = 0
    queries_with_evidence = 0
    index = {candidate_id: position for position, candidate_id in enumerate(ids)}
    for position, row in enumerate(rows):
        scores = np.asarray(logits[int(row["_row_index"])], dtype=float) / temperature
        scores -= scores.max(); q = np.exp(scores); quantitative[position] = q / q.sum()
        target_slot = _slot(pd.Series([row.get("target_time")]))[0]
        user_prior = sparse_prior(users.get(str(row.get("user_id"))), size, alpha, global_prior)
        time_prior = sparse_prior(times.get(target_slot), size, alpha, global_prior)
        bn[position] = normalize(np.sqrt(user_prior * time_prior))
        labels[position] = index[str(row["true_id"])]
        query_id = str(row["query_id"])
        query_has_evidence = False
        for candidate_id, candidate_position in index.items():
            item = evidence.get((query_id, candidate_id))
            if item is None:
                continue
            query_has_evidence = True
            total_evidence += 1
            if not bool(item.get("valid", True)):
                continue
            habit = platt_likelihood_ratio([float(item["habit_score"])], calibration["habit"])[0]
            semantic = platt_likelihood_ratio([float(item["semantic_score"])], calibration["semantic"])[0]
            llm[position, candidate_position] = np.sqrt(habit * semantic); valid_evidence += 1
        queries_with_evidence += int(query_has_evidence)
    return bn, quantitative, llm, labels, {
        "queries": len(rows),
        "queries_with_cached_evidence": queries_with_evidence,
        "query_coverage": queries_with_evidence / len(rows) if rows else 0.0,
        "cached_evidence": total_evidence,
        "valid_evidence": valid_evidence,
        "valid_evidence_rate": valid_evidence / total_evidence if total_evidence else 0.0,
    }


def fuse_sources(bn, quantitative, llm, quantitative_weight: float, llm_weight: float):
    scores = (np.log(np.clip(bn, 1e-12, None))
              + quantitative_weight * np.log(np.clip(quantitative, 1e-12, None))
              + llm_weight * np.log(np.clip(llm, 1e-12, None)))
    scores -= scores.max(axis=1, keepdims=True); values = np.exp(scores)
    return values / values.sum(axis=1, keepdims=True)


def prediction_arrays(probabilities, labels):
    arrays = empty_arrays(len(labels))
    for index, label in enumerate(labels):
        fill(arrays, index, probabilities[index], int(label))
    return arrays


def select_weights(bn, quantitative, llm, labels, grid):
    specs = {
        "M1-data-only": [(0.0, 0.0)],
        "M2-llm": [(0.0, weight) for weight in grid],
        "M3-quantitative": [(weight, 0.0) for weight in grid],
        "M4-both": [(q_weight, llm_weight) for q_weight in grid for llm_weight in grid],
    }
    selected = {}; metrics = {}
    for variant, candidates in specs.items():
        choices = []
        for q_weight, llm_weight in candidates:
            arrays = prediction_arrays(fuse_sources(bn, quantitative, llm, q_weight, llm_weight), labels)
            summary = summarize_arrays(arrays)
            score = summary["recall@1"] + summary["recall@10"]
            choices.append((score, -(q_weight + llm_weight), -q_weight, q_weight, llm_weight, summary))
        _, _, _, q_weight, llm_weight, summary = max(choices)
        selected[variant] = {"quantitative": q_weight, "llm": llm_weight}; metrics[variant] = summary
    return selected, metrics


def run(args):
    validation_rows, validation_logits, validation_ids = load_split(Path(args.validation), args.limit)
    test_rows, test_logits, test_ids = load_split(Path(args.test), args.limit)
    if validation_ids != test_ids:
        raise ValueError("validation and test candidate spaces differ")
    calibration = json.loads(Path(args.calibration).read_text()); evidence = evidence_index(Path(args.evidence_cache))
    train = pd.read_csv(args.train_csv)
    global_prior, users, times, matched_train_rows = fit_bn_statistics(train, len(test_ids), args.alpha); del train
    validation = source_probabilities(validation_rows, validation_logits, validation_ids, global_prior, users, times,
                                      evidence, calibration, args.alpha)
    test = source_probabilities(test_rows, test_logits, test_ids, global_prior, users, times,
                               evidence, calibration, args.alpha)
    for split, diagnostics in (("validation", validation[4]), ("test", test[4])):
        if diagnostics["queries_with_cached_evidence"] != diagnostics["queries"]:
            raise ValueError(f"LLM cache does not cover every {split} query: {diagnostics}")
    grid = [float(value) for value in args.weight_grid]
    selected, validation_metrics = select_weights(*validation[:4], grid)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); test_metrics = {}
    for variant in VARIANTS:
        weights = selected[variant]
        probabilities = fuse_sources(test[0], test[1], test[2], weights["quantitative"], weights["llm"])
        arrays = prediction_arrays(probabilities, test[3]); test_metrics[variant] = summarize_arrays(arrays)
        np.savez_compressed(output / f"{variant}.test.predictions.npz", **arrays,
                            query_index=np.arange(len(test_rows)),
                            query_id=np.asarray([str(row["query_id"]) for row in test_rows]))
    payload = {"rq":"RQ3","protocol":"bounded matched last-query","city":args.city,"seed":args.seed,
               "limit":args.limit,"fit_splits":["train","validation"],"evaluation_split":"test",
               "weight_grid":grid,"selected_weights":selected,"validation_metrics":validation_metrics,
               "test_metrics":test_metrics,"matched_train_rows":matched_train_rows,
               "validation_evidence":validation[4],"test_evidence":test[4]}
    (output / "rq3.metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output":str(output / "rq3.metrics.json"),"selected_weights":selected,
                      "test_metrics":test_metrics}, indent=2)); return payload


def main():
    parser=argparse.ArgumentParser(description="RQ3 LLM knowledge distillation/fusion evaluation")
    parser.add_argument("--train-csv",required=True); parser.add_argument("--validation",required=True)
    parser.add_argument("--test",required=True); parser.add_argument("--evidence-cache",required=True)
    parser.add_argument("--calibration",required=True); parser.add_argument("--output-dir",required=True)
    parser.add_argument("--city",default="Tokyo"); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--limit",type=int,default=200); parser.add_argument("--alpha",type=float,default=1.0)
    parser.add_argument("--weight-grid",nargs="+",type=float,default=[0.0,0.25,0.5,0.75,1.0])
    run(parser.parse_args())


if __name__ == "__main__": main()
