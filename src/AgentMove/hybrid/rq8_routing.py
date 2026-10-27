from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


POLICIES = ("never", "always", "entropy", "margin", "random-budget-matched")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split(path: Path, limit: int | None):
    rows = read_jsonl(path); bundle = rows.pop(0)["_bundle"]
    if limit is not None: rows = rows[:limit]
    def resolve(value):
        target = Path(value)
        if target.is_file(): return target
        for candidate in (path.parent / target.name, path.parent.parent / target.name):
            if candidate.is_file(): return candidate
        raise FileNotFoundError(f"missing bundle artifact: {target}")
    logits_path = resolve(bundle["logits"])
    ids = [str(value) for value in json.loads(resolve(bundle["candidate_ids"]).read_text())]
    logits = np.load(logits_path, mmap_mode="r")
    return rows, logits, ids


def load_always(path: Path, expected_ids):
    by_id = {}
    for row in read_jsonl(path):
        query_id = str(row["query_id"])
        if query_id in by_id:
            raise ValueError(f"Always-LLM cache has duplicate query_id={query_id}: {path}")
        by_id[query_id] = row
    missing = [query_id for query_id in expected_ids if query_id not in by_id]
    if missing: raise ValueError(f"Always-LLM cache missing {len(missing)} queries; first={missing[0]}")
    unexpected = sorted(set(by_id) - set(expected_ids))
    if unexpected:
        raise ValueError(f"Always-LLM cache has unexpected query_id={unexpected[0]}: {path}")
    return [by_id[query_id] for query_id in expected_ids]


def stage_arrays(rows, logits, ids, temperature):
    id_to_index = {value: index for index, value in enumerate(ids)}
    ranks = np.empty(len(rows), dtype=np.int32); entropy = np.empty(len(rows)); margin = np.empty(len(rows))
    for position, row in enumerate(rows):
        scores = np.asarray(logits[int(row["_row_index"])], dtype=float) / temperature
        scores -= scores.max(); probabilities = np.exp(scores); probabilities /= probabilities.sum()
        order = np.argsort(-scores, kind="stable"); true_index = id_to_index[str(row["true_id"])]
        ranks[position] = int(np.where(order == true_index)[0][0]) + 1
        entropy[position] = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum()
        largest = np.partition(probabilities, -2)[-2:]; margin[position] = largest.max() - largest.min()
    return ranks, entropy, margin


def metrics(ranks, calls, always_rows):
    calls = np.asarray(calls, dtype=bool); ranks = np.asarray(ranks)
    latency = np.asarray([float(row.get("latency_seconds", 0)) for row in always_rows]) * calls
    input_tokens = np.asarray([float(row.get("input_tokens", 0)) for row in always_rows]) * calls
    output_tokens = np.asarray([float(row.get("output_tokens", 0)) for row in always_rows]) * calls
    api_requests = np.asarray([float(row.get("api_calls", 1)) for row in always_rows]) * calls
    invalid = np.asarray([not bool(row.get("valid", True)) for row in always_rows], dtype=float) * calls
    return {"queries": int(len(ranks)), "recall@1": float(np.mean(ranks <= 1)),
            "recall@5": float(np.mean(ranks <= 5)), "recall@10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)),
            # ``llm_call_rate`` is retained for backward compatibility.  The
            # explicit fields distinguish selected queries from physical API
            # requests when a cache row required more than one request.
            "llm_call_rate": float(calls.mean()), "selected_query_rate": float(calls.mean()),
            "api_requests_per_query": float(api_requests.mean()),
            "invalid_outputs_per_query": float(invalid.mean()),
            "latency_mean": float(latency.mean()), "latency_p50": float(np.quantile(latency, .5)),
            "latency_p95": float(np.quantile(latency, .95)),
            "input_tokens_per_query": float(input_tokens.mean()),
            "output_tokens_per_query": float(output_tokens.mean()),
            "tokens_per_query": float((input_tokens + output_tokens).mean())}


def routed(stage_ranks, llm_ranks, calls):
    return np.where(np.asarray(calls, dtype=bool), llm_ranks, stage_ranks)


def choose_threshold(kind, uncertainty, stage_ranks, llm_ranks, configured, budget, quantiles):
    auto = np.quantile(uncertainty, np.linspace(0, 1, quantiles + 1))
    candidates = sorted(set(float(x) for x in list(configured) + auto.tolist()))
    choices = []
    for threshold in candidates:
        calls = uncertainty > threshold if kind == "entropy" else uncertainty < threshold
        rate = float(calls.mean())
        if rate <= budget + 1e-12:
            ranks = routed(stage_ranks, llm_ranks, calls)
            score = float(np.mean(ranks <= 1) + np.mean(1.0 / ranks))
            choices.append((score, -rate, threshold))
    if not choices:
        return float("inf") if kind == "entropy" else float("-inf")
    return max(choices)[2]


def threshold_mask(kind, uncertainty, threshold):
    return uncertainty > threshold if kind == "entropy" else uncertainty < threshold


def budget_capped_mask(kind, uncertainty, threshold, budget):
    """Apply a validation-selected threshold under a deterministic test budget.

    A threshold that respects the call-rate cap on validation can exceed it on
    test after an uncertainty-distribution shift.  The cap projection uses only
    test uncertainty scores (never labels or LLM outcomes) and keeps the most
    uncertain threshold-eligible queries, with stable query order breaking ties.
    """
    uncertainty = np.asarray(uncertainty, dtype=float)
    eligible = np.asarray(threshold_mask(kind, uncertainty, threshold), dtype=bool)
    maximum = int(np.floor(float(budget) * len(eligible) + 1e-12))
    if int(eligible.sum()) <= maximum:
        return eligible
    order = (np.argsort(-uncertainty, kind="stable") if kind == "entropy"
             else np.argsort(uncertainty, kind="stable"))
    selected = order[eligible[order]][:maximum]
    result = np.zeros(len(eligible), dtype=bool)
    result[selected] = True
    return result


def oracle_mask(stage_ranks, llm_ranks, budget):
    gain = ((llm_ranks <= 1).astype(float) - (stage_ranks <= 1).astype(float)
            + 1.0 / llm_ranks - 1.0 / stage_ranks)
    order = np.argsort(-gain, kind="stable"); limit = int(np.floor(budget * len(gain)))
    selected = order[:limit]; selected = selected[gain[selected] > 0]
    mask = np.zeros(len(gain), dtype=bool); mask[selected] = True
    return mask


def evaluate(args):
    config_path = Path(args.config); calibration_path = Path(args.calibration)
    validation_path = Path(args.validation); test_path = Path(args.test)
    validation_always_path = Path(args.validation_always); test_always_path = Path(args.test_always)
    config = json.loads(config_path.read_text()); temperature = float(
        json.loads(calibration_path.read_text())["temperature"]["temperature"])
    val_rows, val_logits, val_ids = load_split(validation_path, args.limit)
    test_rows, test_logits, test_ids = load_split(test_path, args.limit)
    if val_ids != test_ids: raise ValueError("validation and test candidate spaces differ")
    val_query_ids = [str(row["query_id"]) for row in val_rows]; test_query_ids = [str(row["query_id"]) for row in test_rows]
    val_always = load_always(validation_always_path, val_query_ids)
    test_always = load_always(test_always_path, test_query_ids)
    val_stage, val_entropy, val_margin = stage_arrays(val_rows, val_logits, val_ids, temperature)
    test_stage, test_entropy, test_margin = stage_arrays(test_rows, test_logits, test_ids, temperature)
    val_llm = np.asarray([row["true_rank"] for row in val_always], dtype=np.int32)
    test_llm = np.asarray([row["true_rank"] for row in test_always], dtype=np.int32)
    budget = float(config["target_call_rate"]); quantiles = int(config["auto_quantile_candidates"])
    budget_enforcement = config.get("test_budget_enforcement")
    if budget_enforcement != "deterministic_uncertainty_rank_hard_cap":
        raise ValueError("routing config must declare deterministic test budget enforcement")
    budget_sweep = {}; selected_by_budget = {}
    for candidate_budget in [float(value) for value in config.get("call_budgets", [budget])]:
        entropy_value = choose_threshold("entropy", val_entropy, val_stage, val_llm,
                                         config["entropy_thresholds"], candidate_budget, quantiles)
        margin_value = choose_threshold("margin", val_margin, val_stage, val_llm,
                                        config["margin_thresholds"], candidate_budget, quantiles)
        selected_by_budget[str(candidate_budget)] = {"entropy": entropy_value, "margin": margin_value}
        budget_sweep[str(candidate_budget)] = {}
        for kind, uncertainty, threshold in (("entropy", test_entropy, entropy_value),
                                             ("margin", test_margin, margin_value)):
            calls = budget_capped_mask(kind, uncertainty, threshold, candidate_budget)
            budget_sweep[str(candidate_budget)][kind] = metrics(routed(test_stage, test_llm, calls), calls, test_always)
    primary = selected_by_budget[str(budget)]
    entropy_threshold = primary["entropy"]; margin_threshold = primary["margin"]
    masks = {"never": np.zeros(len(test_rows), dtype=bool), "always": np.ones(len(test_rows), dtype=bool),
             "entropy": budget_capped_mask("entropy", test_entropy, entropy_threshold, budget),
             "margin": budget_capped_mask("margin", test_margin, margin_threshold, budget)}
    random_count = int(masks["entropy"].sum())
    random_mask = np.zeros(len(test_rows), dtype=bool)
    random_mask[np.random.default_rng(args.seed).permutation(len(test_rows))[:random_count]] = True
    masks["random-budget-matched"] = random_mask
    oracle = oracle_mask(test_stage, test_llm, budget)
    summaries = {}; output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for policy in POLICIES:
        ranks = routed(test_stage, test_llm, masks[policy]); summaries[policy] = metrics(ranks, masks[policy], test_always)
        np.savez_compressed(output / f"{policy}.test.predictions.npz", query_id=np.asarray(test_query_ids),
                            user_id=np.asarray([str(row.get("user_id", "")) for row in test_rows]),
                            labels=np.arange(len(test_rows)), query_index=np.arange(len(test_rows)),
                            ranks=ranks, called_llm=masks[policy].astype(np.int8))
    oracle_metrics = metrics(routed(test_stage, test_llm, oracle), oracle, test_always)
    np.savez_compressed(output / "routing_diagnostics.test.npz", query_id=np.asarray(test_query_ids),
                        stage_ranks=test_stage, llm_ranks=test_llm, entropy=test_entropy, margin=test_margin,
                        oracle_called=oracle.astype(np.int8))
    validation_selection = {}
    for kind, uncertainty, threshold in (("entropy", val_entropy, entropy_threshold),
                                         ("margin", val_margin, margin_threshold)):
        calls = threshold_mask(kind, uncertainty, threshold)
        validation_selection[kind] = {"threshold": threshold, "call_rate": float(calls.mean()),
                                      "metrics": metrics(routed(val_stage, val_llm, calls), calls, val_always)}
    payload = {"rq": "RQ8", "seed": args.seed, "fit_split": "validation", "evaluation_split": "test",
               "limit": args.limit, "target_call_rate": budget,
               "routing_mode": "counterfactual_replay_from_always_response_bank",
               "physical_requests_during_routing": 0,
               "reported_request_cost": "simulated_deployment_budget",
               "test_budget_enforcement": budget_enforcement,
               "provenance": {
                   "routing_config_sha256": sha256_file(config_path),
                   "calibration_sha256": sha256_file(calibration_path),
                   "validation_input_sha256": sha256_file(validation_path),
                   "test_input_sha256": sha256_file(test_path),
                   "validation_always_sha256": sha256_file(validation_always_path),
                   "test_always_sha256": sha256_file(test_always_path),
               },
               "selected_thresholds": {"entropy": entropy_threshold, "margin": margin_threshold},
               "validation_selection": validation_selection, "budget_sweep": budget_sweep,
               "oracle_upper_bound": oracle_metrics,
               "random_matched_to": "entropy test call rate", "metrics": summaries}
    (output / "rq8.metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2)); return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate RQ8 uncertainty-aware LLM routing from an immutable Always cache")
    parser.add_argument("--validation", required=True); parser.add_argument("--test", required=True)
    parser.add_argument("--validation-always", required=True); parser.add_argument("--test-always", required=True)
    parser.add_argument("--calibration", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/beliefmove_evo/routing.json")
    parser.add_argument("--limit", type=int); parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())


if __name__ == "__main__": main()
