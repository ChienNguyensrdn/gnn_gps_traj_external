from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from hybrid.paired_order_test import holm_adjust, load_npz
from hybrid.rq8_routing import POLICIES, budget_capped_mask
from phase2.aggregate_reports import CITIES, dataset_specs


QUALITY_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
PRIMARY_METRICS = ("recall@1", "mrr")
ROUTING_METRICS = QUALITY_METRICS + ("selected_query_rate", "api_requests_per_query",
                                     "latency_mean", "tokens_per_query")
ROUTING_POLICIES = tuple(POLICIES)


def _sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_unique(path: Path) -> tuple[list[dict], dict[str, dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required Direct-1Call artifact: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if "query_id" not in row:
                raise ValueError(f"missing query_id at {path}:{line_number}")
            rows.append(row)
    by_id: dict[str, dict] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in by_id:
            raise ValueError(f"duplicate query_id={query_id} in {path}")
        by_id[query_id] = row
    return rows, by_id


def _require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=1e-8, atol=1e-10):
        raise ValueError(f"{label} mismatch: observed={actual}, expected={expected}")


def _rank_metrics(ranks: np.ndarray) -> dict[str, float]:
    values = np.asarray(ranks, dtype=float)
    return {"queries": int(len(values)),
            "recall@1": float(np.mean(values <= 1)),
            "recall@5": float(np.mean(values <= 5)),
            "recall@10": float(np.mean(values <= 10)),
            "mrr": float(np.mean(1.0 / values))}


def _expected_direct_metrics(rows: list[dict], split: str, limit: int) -> dict:
    quantitative = np.asarray([row["quantitative_rank"] for row in rows], dtype=np.int32)
    direct = np.asarray([row["true_rank"] for row in rows], dtype=np.int32)
    return {
        "protocol": "direct-query-llm-v1", "split": split, "limit": limit,
        "quantitative": _rank_metrics(quantitative),
        "direct-1call": _rank_metrics(direct),
        "candidate_recall": float(np.mean([bool(row["true_in_top_k"]) for row in rows])),
        "invalid_output_rate": float(np.mean([not bool(row["valid"]) for row in rows])),
        "transport_error_rate": float(np.mean([row.get("transport_success") is False for row in rows])),
        "attempted_without_prediction": 0,
        "api_requests_per_query": float(np.mean([float(row["api_calls"]) for row in rows])),
        "input_tokens_per_query": float(np.mean([float(row["input_tokens"]) for row in rows])),
        "output_tokens_per_query": float(np.mean([float(row["output_tokens"]) for row in rows])),
        "latency_mean": float(np.mean([float(row["latency_seconds"]) for row in rows])),
        "latency_p95": float(np.quantile([float(row["latency_seconds"]) for row in rows], .95)),
    }


def _validate_metric_tree(observed: dict, expected: dict, prefix: str) -> None:
    for key, value in expected.items():
        if isinstance(value, dict):
            if not isinstance(observed.get(key), dict):
                raise ValueError(f"{prefix}.{key} is missing or not an object")
            _validate_metric_tree(observed[key], value, f"{prefix}.{key}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if key not in observed:
                raise ValueError(f"{prefix}.{key} is missing")
            _require_close(observed[key], value, f"{prefix}.{key}")
        elif observed.get(key) != value:
            raise ValueError(f"{prefix}.{key} mismatch: observed={observed.get(key)!r}, expected={value!r}")


def _direct_split_paths(folder: Path, split: str) -> dict[str, Path]:
    root = folder / "always-cache" / split
    return {"protocol": root / "protocol.lock.json", "predictions_jsonl": root / "predictions.jsonl",
            "requests_jsonl": root / "requests.jsonl", "metrics": root / "metrics.json",
            "predictions_npz": root / "predictions.npz"}


def _input_hashes(payload: dict) -> dict[str, str]:
    inputs = payload.get("input", {})
    return {key: str(inputs.get(key, {}).get("sha256", ""))
            for key in ("jsonl", "logits", "candidate_ids", "candidate_metadata")}


def _validate_direct_split(folder: Path, split: str, limit: int, model_slug: str) -> dict:
    paths = _direct_split_paths(folder, split)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required Direct-1Call artifacts: " + ", ".join(map(str, missing)))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    predictions, prediction_by_id = _read_jsonl_unique(paths["predictions_jsonl"])
    requests, request_by_id = _read_jsonl_unique(paths["requests_jsonl"])
    transport_errors = [row for row in requests if row.get("status") == "transport_error"]
    if list(request_by_id) != list(prediction_by_id):
        missing_requests = sorted(set(prediction_by_id) - set(request_by_id))
        missing_predictions = sorted(set(request_by_id) - set(prediction_by_id))
        raise ValueError(f"journal/prediction query set or order mismatch for {split}: "
                         f"missing_requests={missing_requests[:5]}, missing_predictions={missing_predictions[:5]}")
    if protocol.get("protocol") != "direct-query-llm-v1" or protocol.get("split") != split:
        raise ValueError(f"Direct-1Call protocol/split mismatch: {paths['protocol']}")
    if protocol.get("limit") != limit or protocol.get("queries") != len(predictions):
        raise ValueError(f"Direct-1Call limit/query-count mismatch: {paths['protocol']}")
    contract = protocol.get("prompt_contract", {})
    if contract.get("uses_structured_llm_evidence") is not False \
            or contract.get("uses_test_labels_in_prompt") is not False:
        raise ValueError(f"illegal Direct-1Call prompt inputs: {paths['protocol']}")
    if protocol.get("physical_requests_per_uncached_query") != 1 \
            or protocol.get("automatic_retries") is not False:
        raise ValueError(f"Direct-1Call request budget mismatch: {paths['protocol']}")
    if protocol.get("prompt_contract_sha256") != _sha256_json(contract):
        raise ValueError(f"prompt-contract content hash mismatch: {paths['protocol']}")
    query_ids = [str(row["query_id"]) for row in predictions]
    if protocol.get("query_ids_sha256") != _sha256_json(query_ids):
        raise ValueError(f"query content hash mismatch: {paths['protocol']}")
    if protocol.get("selected_rows_sha256") != protocol.get("input", {}).get("selected_rows_sha256"):
        raise ValueError(f"selected-row content hash mismatch: {paths['protocol']}")
    if any(not value for value in _input_hashes(protocol).values()):
        raise ValueError(f"missing input content hashes: {paths['protocol']}")
    model = str(protocol.get("model", ""))
    if model.replace(":", "-").replace("/", "-") != model_slug:
        raise ValueError(f"model slug mismatch in {paths['protocol']}: model={model}, expected={model_slug}")
    model_digest = protocol.get("model_digest")
    if not model_digest or model_digest == "unknown":
        raise ValueError(f"missing concrete model digest: {paths['protocol']}")

    for row in predictions:
        query_id = str(row["query_id"]); request = request_by_id[query_id]
        if row.get("protocol") != protocol["protocol"] or row.get("model") != model \
                or row.get("model_digest") != model_digest:
            raise ValueError(f"prediction protocol/model mismatch: split={split}, query={query_id}")
        if row.get("world_mode") != contract.get("world_mode"):
            raise ValueError(f"prediction world-mode mismatch: split={split}, query={query_id}")
        _require_close(row.get("temperature", -1), protocol.get("temperature", -2),
                       f"{split}.{query_id}.temperature")
        if request.get("split") != split or int(request.get("physical_attempts", 0)) != 1 \
                or int(request.get("api_calls", 0)) != 1:
            raise ValueError(f"request split/physical-attempt mismatch: split={split}, query={query_id}")
        status = request.get("status")
        if status not in {"success", "transport_error"} \
                or (request.get("success") is True) != (status == "success") \
                or (row.get("transport_success") is True) != (status == "success"):
            raise ValueError(f"request transport-status mismatch: split={split}, query={query_id}")
        if int(row.get("api_calls", 0)) != 1 or int(row.get("physical_attempts", 0)) != 1:
            raise ValueError(f"prediction request count mismatch: split={split}, query={query_id}")
        if request.get("prompt_sha256") != row.get("prompt_sha256"):
            raise ValueError(f"prompt SHA mismatch: split={split}, query={query_id}")
        if row.get("truncated") is not False \
                or row.get("original_prompt_sha256") != row.get("prompt_sha256") \
                or row.get("sent_prompt_sha256") != row.get("prompt_sha256") \
                or row.get("original_prompt_chars") != row.get("sent_prompt_chars") \
                or row.get("original_prompt_estimated_tokens") != row.get("sent_prompt_estimated_tokens"):
            raise ValueError(f"prompt truncation/provenance mismatch: split={split}, query={query_id}")
        embedded = request.get("prediction")
        if not isinstance(embedded, dict) or embedded != row:
            raise ValueError(f"journaled prediction mismatch: split={split}, query={query_id}")
        for metric in ("input_tokens", "output_tokens", "latency_seconds"):
            _require_close(request.get(metric, -1), row.get(metric, -2),
                           f"{split}.{query_id}.{metric}")
        stage = [str(value) for value in row.get("stage1_top_k", [])]
        direct = [str(value) for value in row.get("direct_top_k", [])]
        if not stage or len(stage) != len(set(stage)) or len(direct) != len(stage) or set(direct) != set(stage):
            raise ValueError(f"candidate permutation mismatch: split={split}, query={query_id}")
        if len(stage) != int(contract.get("top_k", -1)):
            raise ValueError(f"candidate top-k mismatch: split={split}, query={query_id}")
        if row.get("candidate_ids_sha256") != _sha256_json(stage):
            raise ValueError(f"candidate content hash mismatch: split={split}, query={query_id}")
        true_id = str(row.get("true_id", ""))
        true_in_top_k = true_id in stage
        if bool(row.get("true_in_top_k")) != true_in_top_k:
            raise ValueError(f"candidate-recall flag mismatch: split={split}, query={query_id}")
        quantitative_rank = int(row.get("quantitative_rank", 0))
        direct_rank = int(row.get("true_rank", 0))
        if quantitative_rank < 1 or direct_rank < 1:
            raise ValueError(f"non-positive rank: split={split}, query={query_id}")
        expected_quantitative = stage.index(true_id) + 1 if true_in_top_k else quantitative_rank
        expected_direct = direct.index(true_id) + 1 if true_in_top_k else quantitative_rank
        if quantitative_rank != expected_quantitative or direct_rank != expected_direct:
            raise ValueError(f"rank/candidate mismatch: split={split}, query={query_id}")
        if not bool(row.get("valid")) and direct != stage:
            raise ValueError(f"invalid output did not fully fall back to Q0: split={split}, query={query_id}")

    arrays = load_npz(paths["predictions_npz"])
    expected_user_ids = np.asarray([str(row.get("user_id", "")) for row in predictions])
    if np.any(expected_user_ids == ""):
        raise ValueError(f"empty user_id in Direct-1Call predictions: split={split}")
    expected_quantitative = np.asarray([row["quantitative_rank"] for row in predictions], dtype=np.int32)
    expected_direct = np.asarray([row["true_rank"] for row in predictions], dtype=np.int32)
    for key, expected in (("query_id", np.asarray(query_ids)), ("user_id", expected_user_ids),
                          ("quantitative_ranks", expected_quantitative), ("direct_ranks", expected_direct),
                          ("ranks", expected_direct)):
        if key not in arrays or not np.array_equal(arrays[key].astype(str) if key in {"query_id", "user_id"} else arrays[key],
                                                   expected.astype(str) if key in {"query_id", "user_id"} else expected):
            raise ValueError(f"Direct NPZ/JSONL alignment mismatch: split={split}, field={key}")
    expected_metrics = _expected_direct_metrics(predictions, split, limit)
    expected_metrics.update({"prompts_truncated": False})
    _validate_metric_tree(metrics, expected_metrics, f"{split}.metrics")
    return {"protocol": protocol, "metrics": metrics, "rows": predictions, "arrays": arrays,
            "paths": paths, "transport_errors": len(transport_errors),
            "transport_error_query_ids": [str(row["query_id"]) for row in transport_errors]}


def _validate_direct_unit(folder: Path, limit: int, model_slug: str) -> dict:
    validation = _validate_direct_split(folder, "validation", limit, model_slug)
    test = _validate_direct_split(folder, "test", limit, model_slug)
    left, right = validation["protocol"], test["protocol"]
    identity_keys = ("protocol", "candidate_ids_sha256", "prompt_contract_sha256", "temperature",
                     "model", "model_digest", "platform", "temperature_parameter", "max_tokens",
                     "max_input_tokens", "physical_requests_per_uncached_query", "automatic_retries")
    for key in identity_keys:
        if left.get(key) != right.get(key):
            raise ValueError(f"validation/test protocol identity mismatch for {key}: {folder}")
    left_inputs, right_inputs = _input_hashes(left), _input_hashes(right)
    for key in ("candidate_ids", "candidate_metadata"):
        if left_inputs[key] != right_inputs[key]:
            raise ValueError(f"validation/test input content hash mismatch for {key}: {folder}")
    temperature_path = folder / "stage1_temperature.json"
    if not temperature_path.is_file():
        raise FileNotFoundError(f"missing required Direct-1Call artifact: {temperature_path}")
    temperature = json.loads(temperature_path.read_text(encoding="utf-8"))
    if temperature.get("protocol") != left["protocol"] or temperature.get("fit_split") != "validation" \
            or temperature.get("limit") != limit or temperature.get("queries") != left.get("queries"):
        raise ValueError(f"temperature protocol/limit mismatch: {temperature_path}")
    if temperature.get("query_ids_sha256") != left.get("query_ids_sha256") \
            or temperature.get("selected_rows_sha256") != left.get("selected_rows_sha256") \
            or temperature.get("candidate_ids_sha256") != left.get("candidate_ids_sha256"):
        raise ValueError(f"temperature validation content hash mismatch: {temperature_path}")
    if _input_hashes(temperature) != left_inputs:
        raise ValueError(f"temperature validation input hash mismatch: {temperature_path}")
    if temperature.get("temperature", {}).get("temperature") != left.get("temperature"):
        raise ValueError(f"temperature value mismatch: {temperature_path}")
    return {"validation": validation, "test": test, "temperature": temperature,
            "model_digest": left["model_digest"],
            "transport_errors": validation["transport_errors"] + test["transport_errors"]}


def _expected_routing_metrics(ranks: np.ndarray, called: np.ndarray,
                              direct_rows: list[dict]) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=float); called = np.asarray(called, dtype=bool)
    latency = np.asarray([float(row["latency_seconds"]) for row in direct_rows]) * called
    inputs = np.asarray([float(row["input_tokens"]) for row in direct_rows]) * called
    outputs = np.asarray([float(row["output_tokens"]) for row in direct_rows]) * called
    requests = np.asarray([float(row["api_calls"]) for row in direct_rows]) * called
    invalid = np.asarray([not bool(row["valid"]) for row in direct_rows], dtype=float) * called
    return {"queries": int(len(ranks)), "recall@1": float(np.mean(ranks <= 1)),
            "recall@5": float(np.mean(ranks <= 5)), "recall@10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)), "llm_call_rate": float(called.mean()),
            "selected_query_rate": float(called.mean()),
            "api_requests_per_query": float(requests.mean()),
            "invalid_outputs_per_query": float(invalid.mean()),
            "latency_mean": float(latency.mean()), "latency_p50": float(np.quantile(latency, .5)),
            "latency_p95": float(np.quantile(latency, .95)),
            "input_tokens_per_query": float(inputs.mean()),
            "output_tokens_per_query": float(outputs.mean()),
            "tokens_per_query": float((inputs + outputs).mean())}


def _routing_identity(payload: dict) -> dict:
    keys = ("rq", "fit_split", "evaluation_split", "limit", "target_call_rate",
            "routing_mode", "physical_requests_during_routing", "reported_request_cost",
            "test_budget_enforcement",
            "selected_thresholds", "validation_selection", "budget_sweep", "random_matched_to",
            "routing_config_sha256", "config_sha256", "provenance")
    return {key: payload[key] for key in keys if key in payload}


def _validate_routing_unit(folder: Path, seeds: list[int], direct: dict, limit: int) -> list[dict]:
    expected_ids = direct["test"]["arrays"]["query_id"].astype(str)
    expected_users = direct["test"]["arrays"]["user_id"].astype(str)
    quantitative = direct["test"]["arrays"]["quantitative_ranks"]
    direct_ranks = direct["test"]["arrays"]["direct_ranks"]
    direct_rows = direct["test"]["rows"]
    provenance_expected = {
        "calibration_sha256": _sha256_file(folder / "stage1_temperature.json"),
        "validation_input_sha256": _input_hashes(direct["validation"]["protocol"])["jsonl"],
        "test_input_sha256": _input_hashes(direct["test"]["protocol"])["jsonl"],
        "validation_always_sha256": _sha256_file(
            folder / "always-cache/validation/predictions.jsonl"),
        "test_always_sha256": _sha256_file(folder / "always-cache/test/predictions.jsonl"),
    }
    payloads = []; identity = None; entropy_count = None; deterministic_reference = {}
    for seed in seeds:
        route = folder / f"routing/seed-{seed}"
        metric_path = route / "rq8.metrics.json"
        if not metric_path.is_file():
            raise FileNotFoundError(f"missing required routing artifact: {metric_path}")
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        if payload.get("seed") != seed:
            raise ValueError(f"routing seed/directory mismatch: {metric_path}")
        if payload.get("limit") != limit or payload.get("fit_split") != "validation" \
                or payload.get("evaluation_split") != "test":
            raise ValueError(f"routing split/limit mismatch: {metric_path}")
        if payload.get("routing_mode") != "counterfactual_replay_from_always_response_bank" \
                or payload.get("physical_requests_during_routing") != 0 \
                or payload.get("reported_request_cost") != "simulated_deployment_budget" \
                or payload.get("test_budget_enforcement") \
                != "deterministic_uncertainty_rank_hard_cap":
            raise ValueError(f"routing replay protocol mismatch: {metric_path}")
        provenance = payload.get("provenance", {})
        if not provenance.get("routing_config_sha256"):
            raise ValueError(f"missing routing configuration hash: {metric_path}")
        for key, expected in provenance_expected.items():
            if provenance.get(key) != expected:
                raise ValueError(f"routing provenance mismatch for {key}: {metric_path}")
        current_identity = _routing_identity(payload)
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"routing configuration identity differs across seeds: {metric_path}")
        arrays_by_policy = {}
        for policy in ROUTING_POLICIES:
            path = route / f"{policy}.test.predictions.npz"
            arrays = load_npz(path); arrays_by_policy[policy] = arrays
            for key, expected in (("query_id", expected_ids), ("user_id", expected_users)):
                if key not in arrays or not np.array_equal(arrays[key].astype(str), expected):
                    raise ValueError(f"routing/direct alignment mismatch: seed={seed}, policy={policy}, field={key}")
            if np.any(arrays["user_id"].astype(str) == ""):
                raise ValueError(f"empty routing user_id: seed={seed}, policy={policy}")
            if "ranks" not in arrays or "called_llm" not in arrays:
                raise ValueError(f"routing NPZ missing ranks/called_llm: {path}")
            called = arrays["called_llm"]
            if not np.all(np.isin(called, (0, 1))):
                raise ValueError(f"routing mask is not binary: {path}")
            expected_metrics = _expected_routing_metrics(arrays["ranks"], called, direct_rows)
            _validate_metric_tree(payload.get("metrics", {}).get(policy, {}), expected_metrics,
                                  f"routing.seed-{seed}.{policy}")
            if policy != "random-budget-matched":
                signature = (arrays["ranks"], arrays["called_llm"])
                if policy not in deterministic_reference:
                    deterministic_reference[policy] = tuple(value.copy() for value in signature)
                elif any(not np.array_equal(value, reference)
                         for value, reference in zip(signature, deterministic_reference[policy])):
                    raise ValueError(f"deterministic routing differs across seeds: seed={seed}, policy={policy}")
        always, never = arrays_by_policy["always"], arrays_by_policy["never"]
        if not np.array_equal(always["ranks"], direct_ranks) or not np.all(always["called_llm"] == 1):
            raise ValueError(f"routing always policy is not the Direct-1Call cache: seed={seed}")
        if not np.array_equal(never["ranks"], quantitative) or not np.all(never["called_llm"] == 0):
            raise ValueError(f"routing never policy is not Q0 quantitative: seed={seed}")
        entropy = arrays_by_policy["entropy"]; random = arrays_by_policy["random-budget-matched"]
        selected = int(np.sum(entropy["called_llm"])); random_selected = int(np.sum(random["called_llm"]))
        if random_selected != selected:
            raise ValueError(f"random routing request budget differs from entropy: seed={seed}")
        expected_random = np.zeros(len(expected_ids), dtype=bool)
        indices = np.random.default_rng(seed).permutation(len(expected_ids))[:selected]
        expected_random[indices] = True
        if not np.array_equal(random["called_llm"].astype(bool), expected_random):
            raise ValueError(f"random routing mask does not match declared seed: seed={seed}")
        if entropy_count is None:
            entropy_count = selected
        elif selected != entropy_count:
            raise ValueError(f"deterministic entropy routing differs across seeds: seed={seed}")
        target = float(payload.get("target_call_rate", 1.0))
        if selected / max(len(expected_ids), 1) > target + 1e-12:
            raise ValueError(f"observed entropy request rate exceeds configured cap: seed={seed}")
        for kind in ("entropy", "margin"):
            validation_rate = float(payload.get("validation_selection", {})
                                    .get(kind, {}).get("call_rate", 2.0))
            if validation_rate > target + 1e-12:
                raise ValueError(f"validation {kind} request rate exceeds configured cap: seed={seed}")
        diagnostics_path = route / "routing_diagnostics.test.npz"
        diagnostics = load_npz(diagnostics_path)
        for key, expected in (("query_id", expected_ids), ("stage_ranks", quantitative),
                              ("llm_ranks", direct_ranks)):
            actual = diagnostics.get(key)
            if actual is None or not np.array_equal(actual.astype(str) if key == "query_id" else actual,
                                                     expected):
                raise ValueError(f"routing diagnostics mismatch: seed={seed}, field={key}")
        for key in ("entropy", "margin", "oracle_called"):
            if key not in diagnostics or len(diagnostics[key]) != len(expected_ids):
                raise ValueError(f"routing diagnostics missing/alignment mismatch: seed={seed}, field={key}")
        for kind in ("entropy", "margin"):
            threshold = float(payload.get("selected_thresholds", {}).get(kind, np.nan))
            expected_called = budget_capped_mask(kind, diagnostics[kind], threshold, target)
            actual_called = arrays_by_policy[kind]["called_llm"].astype(bool)
            if not np.array_equal(actual_called, expected_called):
                raise ValueError(f"routing mask is not the deterministic capped mask: seed={seed}, policy={kind}")
        for kind in ("entropy", "margin", "random-budget-matched"):
            arrays = arrays_by_policy[kind]
            expected_ranks = np.where(arrays["called_llm"].astype(bool), direct_ranks, quantitative)
            if not np.array_equal(arrays["ranks"], expected_ranks):
                raise ValueError(f"routing ranks do not match selected source: seed={seed}, policy={kind}")
        payloads.append(payload)
    return payloads


def summary(values: list[float]) -> dict:
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def unit_macro(values: dict[str, list[float]]) -> dict:
    if len(values) == 1:
        return summary(next(iter(values.values())))
    return summary([float(np.mean(rows)) for rows in values.values()])


def rank_values(ranks: np.ndarray, metric: str) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=float)
    if metric.startswith("recall@"):
        return (ranks <= int(metric.split("@")[1])).astype(float)
    if metric == "mrr":
        return 1.0 / ranks
    raise ValueError(f"unsupported metric: {metric}")


def clustered_test(blocks: list[tuple[np.ndarray, np.ndarray]], iterations: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("clustered paired test requires blocks")
    rng = np.random.default_rng(seed); width = blocks[0][0].shape[1]
    observed = np.mean([values.mean(axis=0) for values, _ in blocks], axis=0)
    boot = np.zeros((iterations, width)); null = np.zeros((iterations, width))
    for values, users in blocks:
        unique = np.unique(users)
        cluster_sums = np.stack([values[users == user].sum(axis=0) for user in unique])
        cluster_counts = np.asarray([np.count_nonzero(users == user) for user in unique], dtype=float)
        sampled = rng.integers(0, len(unique), size=(iterations, len(unique)))
        sampled_sums = cluster_sums[sampled].sum(axis=1)
        sampled_counts = cluster_counts[sampled].sum(axis=1)[:, None]
        boot += sampled_sums / sampled_counts
        signs = rng.integers(0, 2, size=(iterations, len(unique), 1), dtype=np.int8) * 2 - 1
        null += (cluster_sums[None, :, :] * signs).sum(axis=1) / cluster_counts.sum()
    boot /= len(blocks); null /= len(blocks)
    intervals = np.quantile(boot, (0.025, 0.975), axis=0).T
    pvalues = (1 + np.count_nonzero(np.abs(null) >= np.abs(observed), axis=0)) / (iterations + 1)
    return observed, intervals, pvalues


def direct_root(spec: dict, unit: str, slug: str, limit: int) -> Path:
    return spec["root"] / unit / f"direct-query-llm/{slug}/limit-{limit}"


def collect(root: Path, slug: str, limit: int, cities: list[str], seeds: list[int],
            iterations: int) -> dict:
    datasets, tests, missing = {}, [], []
    all_model_digests: set[str] = set()
    total_transport_errors = 0
    pending_tests = []
    for spec_index, spec in enumerate(dataset_specs(root, cities)):
        direct_runs: dict[str, dict] = {}; routing_runs: dict[str, list[dict]] = {}
        direct_blocks = []; routing_blocks = []; unit_provenance = {}
        for unit in spec["units"]:
            folder = direct_root(spec, unit, slug, limit)
            required = [folder / "stage1_temperature.json"]
            for split in ("validation", "test"):
                required.extend(_direct_split_paths(folder, split).values())
            for seed in seeds:
                route = folder / f"routing/seed-{seed}"
                required.append(route / "rq8.metrics.json")
                required.append(route / "routing_diagnostics.test.npz")
                required.extend(route / f"{policy}.test.predictions.npz" for policy in ROUTING_POLICIES)
            absent = [path for path in required if not path.is_file()]
            missing.extend(str(path) for path in absent)
            if not absent:
                direct_artifacts = _validate_direct_unit(folder, limit, slug)
                route_rows = _validate_routing_unit(folder, seeds, direct_artifacts, limit)
                unit_provenance[unit] = direct_artifacts
                all_model_digests.add(direct_artifacts["model_digest"])
                total_transport_errors += direct_artifacts["transport_errors"]
                direct_metric = direct_artifacts["test"]["metrics"]
                direct_runs[unit] = direct_metric
                routing_runs[unit] = route_rows
                direct = direct_artifacts["test"]["arrays"]
                direct_columns = np.column_stack([
                    rank_values(direct["direct_ranks"], metric) - rank_values(direct["quantitative_ranks"], metric)
                    for metric in PRIMARY_METRICS])
                direct_blocks.append((direct_columns, direct["user_id"].astype(str)))
                first_route = folder / f"routing/seed-{seeds[0]}"
                entropy = load_npz(first_route / "entropy.test.predictions.npz")
                random_rows = [load_npz(folder / f"routing/seed-{seed}/random-budget-matched.test.predictions.npz")
                               for seed in seeds]
                if any(not np.array_equal(entropy["query_id"], row["query_id"])
                       or not np.array_equal(entropy["user_id"], row["user_id"])
                       for row in random_rows):
                    raise ValueError(f"unaligned direct-query routing predictions: {spec['key']}/{unit}")
                routing_columns = np.column_stack([
                    rank_values(entropy["ranks"], metric) - np.mean(
                        [rank_values(row["ranks"], metric) for row in random_rows], axis=0)
                    for metric in PRIMARY_METRICS])
                routing_blocks.append((routing_columns, entropy["user_id"].astype(str)))
        if len(direct_runs) != len(spec["units"]):
            continue
        methods = {}
        for method in ("quantitative", "direct-1call"):
            methods[method] = {metric: unit_macro({unit: [row[method][metric]]
                                                    for unit, row in direct_runs.items()})
                               for metric in QUALITY_METRICS}
        diagnostics = {
            metric: unit_macro({unit: [row[metric]] for unit, row in direct_runs.items()})
            for metric in ("candidate_recall", "invalid_output_rate", "transport_error_rate",
                           "api_requests_per_query",
                           "input_tokens_per_query", "output_tokens_per_query", "latency_mean", "latency_p95")
        }
        policies = {}
        for policy in POLICIES:
            policies[policy] = {}
            for metric in ROUTING_METRICS:
                per_unit = {}
                for unit, rows in routing_runs.items():
                    selected = rows if policy == "random-budget-matched" else rows[:1]
                    per_unit[unit] = [row["metrics"][policy][metric] for row in selected]
                policies[policy][metric] = unit_macro(per_unit)
            policies[policy]["runs_per_unit"] = len(seeds) if policy == "random-budget-matched" else 1
        datasets[spec["key"]] = {"label": spec["label"], "methods": methods,
                                  "diagnostics": diagnostics, "routing": policies,
                                  "provenance": {"journal_prediction_alignment": "verified",
                                                 "transport_errors": sum(row["transport_errors"]
                                                                         for row in unit_provenance.values()),
                                                 "model_digests": sorted({row["model_digest"]
                                                                          for row in unit_provenance.values()})}}
        pending_tests.extend([
            (spec["key"], "direct-1call-vs-quantitative", direct_blocks, 700 + spec_index * 2),
            (spec["key"], "entropy-vs-random-budget-matched", routing_blocks, 701 + spec_index * 2),
        ])
    if len(all_model_digests) > 1:
        raise ValueError(f"Direct-1Call model digest differs across units: {sorted(all_model_digests)}")
    for dataset, comparison, blocks, seed in pending_tests:
        effects, intervals, pvalues = clustered_test(blocks, iterations, seed)
        for index, metric in enumerate(PRIMARY_METRICS):
            tests.append({"dataset": dataset, "comparison": comparison, "metric": metric,
                          "effect": float(effects[index]), "ci95": intervals[index].tolist(),
                          "p": float(pvalues[index]), "cluster_unit": "user",
                          "tist2015_macro_unit": "city"})
    adjusted = holm_adjust([row["p"] for row in tests]) if tests else []
    for row, value in zip(tests, adjusted):
        row["holm_p"] = value; row["significant"] = value < .05
    return {"experiment": "direct-query-llm", "model": slug, "limit": limit,
            "estimands": ["direct-1call-vs-quantitative", "entropy-vs-random-budget-matched"],
            "datasets": datasets, "paired_tests": tests, "random_seeds": seeds,
            "model_digests": sorted(all_model_digests), "transport_errors": total_transport_errors,
            "missing": sorted(set(missing)), "gate": "ready-bounded" if not missing else "incomplete"}


def fmt(value: dict, digits: int = 6) -> str:
    return f"{value['mean']:.{digits}f}" if value["std"] is None else \
        f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def verdict(row: dict) -> str:
    if not row["significant"]:
        return "chưa có ý nghĩa sau Holm"
    return "cải thiện có ý nghĩa" if row["effect"] > 0 else "suy giảm có ý nghĩa"


def render(payload: dict) -> str:
    lines = ["# Phase 2 — Đối chứng query-time LLM", "",
             f"> Bounded matched experiment: `LLM_LIMIT={payload['limit']}`, model `{payload['model']}`. Q1 là direct LLM reranker trên frozen quantitative top-10, không phải LLM-only.", "",
             "## Câu hỏi", "", "Một lần query LLM trực tiếp có cải thiện frozen quantitative predictor không, và entropy routing có vượt random control tại cùng ngân sách một request hay không?", "",
             "## Q0 so với Q1", "", "| Dataset | Method | R@1 | R@5 | R@10 | MRR |", "|---|---|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        for method in ("quantitative", "direct-1call"):
            lines.append(f"| {data['label']} | {method} | " +
                         " | ".join(fmt(data["methods"][method][metric]) for metric in QUALITY_METRICS) + " |")
    lines += ["", "## Chi phí và tính hợp lệ của Q1", "",
              "| Dataset | Candidate recall | Invalid output | Transport error | Requests/query | Input tokens/query | Output tokens/query | Latency mean | Latency p95 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        d = data["diagnostics"]
        lines.append(f"| {data['label']} | " + " | ".join(fmt(d[key], 2 if "tokens" in key else 6)
                     for key in ("candidate_recall", "invalid_output_rate", "transport_error_rate",
                                 "api_requests_per_query",
                                 "input_tokens_per_query", "output_tokens_per_query", "latency_mean", "latency_p95")) + " |")
    lines += ["", "## Routing với Direct-1Call", "",
              "| Dataset | Policy | Runs/unit | R@1 | R@5 | R@10 | MRR | Selected rate | Requests/query | Latency mean | Tokens/query |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        for policy in POLICIES:
            row = data["routing"][policy]
            lines.append(f"| {data['label']} | {policy} | {row['runs_per_unit']} | " +
                         " | ".join(fmt(row[metric], 2 if metric == "tokens_per_query" else 6)
                                      for metric in ROUTING_METRICS) + " |")
    lines += ["", "## Kiểm định chính", "",
              "| Dataset | Comparison | Metric | Effect | User-cluster 95% CI | Holm p | Kết luận |",
              "|---|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        label = payload["datasets"][row["dataset"]]["label"]
        lines.append(f"| {label} | {row['comparison']} | {row['metric']} | {row['effect']:.6f} | "
                     f"{row['ci95'][0]:.6f}–{row['ci95'][1]:.6f} | {row['holm_p']:.6g} | {verdict(row)} |")
    lines += ["", "## Protocol gate", "",
              "- Q0 và Q1 dùng cùng query IDs, full candidate space và quantitative top-10.",
              "- Q1 không đọc structured-evidence cache; mỗi uncached query có đúng một physical request, không retry.",
              "- JSON sai hoặc thiếu candidate fallback toàn bộ về Q0 nhưng vẫn tính một request.",
              "- Temperature và router threshold chỉ fit trên validation; test không dùng để tuning.",
              "- Routing được đánh giá bằng counterfactual replay; request, token và latency là chi phí triển khai mô phỏng từ immutable Direct-1Call cache, không phải API call mới trong bước routing.",
              "- TIST2015 là macro đều 12 thành phố; bootstrap/sign-flip theo cụm user.",
              f"- Gate: **{payload['gate']}**; artifact thiếu: **{len(payload['missing'])}**.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched Direct-1Call query-time LLM experiment")
    parser.add_argument("--root", type=Path, default=Path("results/phase2"))
    parser.add_argument("--model-slug", default="qwen2-7b"); parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--cities", nargs="+", default=list(CITIES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 92)))
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("--iterations must be at least 1000")
    payload = collect(args.root, args.model_slug, args.limit, args.cities, args.seeds, args.iterations)
    output = args.root / "aggregated" / f"direct_query_llm_limit-{args.limit}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = Path(__file__).resolve().parent / "direct_query_llm_report.md"
    report.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"], "missing": len(payload["missing"]),
                      "json": str(output), "report": str(report)}))
    if payload["gate"] != "ready-bounded" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
