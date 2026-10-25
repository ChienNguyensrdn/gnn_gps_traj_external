from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .calibration import TemperatureScaler
from .evidence import LLMServiceUnavailable
from .memory import EmbeddingMemoryRetriever
from .rq8_routing import read_jsonl


PROTOCOL_VERSION = "direct-query-llm-v1"
PROMPT_VERSION = "direct-top10-quantitative-rerank-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_bundle(input_path: Path, value: str) -> Path:
    target = Path(value)
    if target.is_file():
        return target.resolve()
    for candidate in (input_path.parent / target.name, input_path.parent.parent / target.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing bundle artifact: {target}")


def load_split(path: Path, limit: int | None) -> tuple[list[dict], np.ndarray, list[str], dict, dict]:
    entries = read_jsonl(path)
    if not entries or "_bundle" not in entries[0]:
        raise ValueError(f"expected bundled hybrid JSONL: {path}")
    bundle = entries.pop(0)["_bundle"]
    rows = entries if limit is None else entries[:limit]
    query_ids = [str(row["query_id"]) for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"duplicate query_id in {path}")
    logits_path = _resolve_bundle(path, bundle["logits"])
    ids_path = _resolve_bundle(path, bundle["candidate_ids"])
    metadata_path = _resolve_bundle(path, bundle["candidate_metadata"])
    ids = [str(value) for value in json.loads(ids_path.read_text(encoding="utf-8"))]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logits = np.load(logits_path, mmap_mode="r")
    resolved = {
        "jsonl": {"path": str(path.resolve()), "sha256": _sha256_file(path)},
        "logits": {"path": str(logits_path), "sha256": _sha256_file(logits_path)},
        "candidate_ids": {"path": str(ids_path), "sha256": _sha256_file(ids_path)},
        "candidate_metadata": {"path": str(metadata_path),
                               "sha256": _sha256_file(metadata_path)},
        "selected_rows_sha256": _sha256_json(rows),
    }
    return rows, logits, ids, metadata, resolved


def _temperature_payload(validation: Path, limit: int) -> dict:
    rows, logits, ids, _, resolved = load_split(validation, limit)
    index = {candidate_id: position for position, candidate_id in enumerate(ids)}
    matrix, labels = [], []
    for row in rows:
        true_id = str(row["true_id"])
        if true_id not in index:
            raise ValueError(f"validation true_id outside candidate space: {true_id}")
        matrix.append(np.asarray(logits[int(row["_row_index"])], dtype=float))
        labels.append(index[true_id])
    scaler = TemperatureScaler().fit(matrix, labels)
    return {
        "protocol": PROTOCOL_VERSION,
        "fit_split": "validation",
        "limit": limit,
        "queries": len(rows),
        "query_ids_sha256": _sha256_json([str(row["query_id"]) for row in rows]),
        "selected_rows_sha256": resolved["selected_rows_sha256"],
        "candidate_ids_sha256": _sha256_json(ids),
        "input": resolved,
        "temperature": scaler.to_dict(),
    }


def fit_temperature(validation: Path, output: Path, limit: int) -> dict:
    payload = _temperature_payload(validation, limit)
    if output.exists():
        current = json.loads(output.read_text(encoding="utf-8"))
        if current != payload:
            raise ValueError(f"temperature lock mismatch; use a new output root: {output}")
    else:
        _atomic_json(output, payload)
    print(json.dumps({"output": str(output), "temperature": payload["temperature"]["temperature"],
                      "queries": payload["queries"]}))
    return payload


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / max(temperature, 1e-8)
    scaled -= scaled.max(); result = np.exp(scaled)
    return result / result.sum()


def candidate_payload(candidate_id: str, probability: float, metadata: dict,
                      world_mode: str) -> dict:
    item = metadata.get(candidate_id, {})
    payload = {"candidate_id": candidate_id, "quantitative_probability": float(probability),
               "category": item.get("category"), "latitude": item.get("latitude"),
               "longitude": item.get("longitude")}
    if world_mode == "full":
        payload["address"] = item.get("address")
    return payload


def build_prompt(row: dict, memory: list[str], candidates: list[dict], world_mode: str) -> str:
    world_rule = ("Use only the supplied category and coordinates; no address/OSM fields are available."
                  if world_mode == "internal_only" else
                  "Use the supplied category, coordinates and address fields.")
    history = list(row.get("history", []))[-20:]
    context = list(row.get("context", []))[-10:]
    return f"""You rerank a frozen quantitative model's candidates for next-location prediction.
Use only the observed prefix and supplied candidate information. Do not invent candidate IDs.
{world_rule}
Return JSON only: {{"ranking":["candidate_id",...]}} with every supplied ID exactly once.
Target time: {row.get('target_time')}
Observed history tail: {json.dumps(history, ensure_ascii=False)}
Recent context: {json.dumps(context, ensure_ascii=False)}
Deterministically retrieved memory: {json.dumps(memory, ensure_ascii=False)}
Frozen quantitative top candidates: {json.dumps(candidates, ensure_ascii=False)}
Return JSON only and include every supplied candidate ID exactly once.
"""


def _estimated_tokens(text: str) -> int:
    """Match the deterministic bound used by ``LLMWrapper.prepare_prompt``."""
    return max(1, (len(text) + 3) // 4)


def _prompt_audit(prompt: str) -> dict:
    digest = _sha256_bytes(prompt.encode("utf-8"))
    return {
        "prompt_sha256": digest,
        "original_prompt_sha256": digest,
        "sent_prompt_sha256": digest,
        "original_prompt_chars": len(prompt),
        "sent_prompt_chars": len(prompt),
        "original_prompt_estimated_tokens": _estimated_tokens(prompt),
        "sent_prompt_estimated_tokens": _estimated_tokens(prompt),
        "truncated": False,
    }


def parse_ranking(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("response has no JSON object")
        payload = json.loads(match.group(0))
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking, list):
        raise ValueError("JSON must contain a ranking list")
    return [str(value) for value in ranking]


def validate_ranking(parsed: Iterable[str], top_ids: list[str]) -> tuple[list[str], bool, str]:
    values = list(parsed)
    valid = len(values) == len(top_ids) and len(set(values)) == len(values) and set(values) == set(top_ids)
    if valid:
        return values, True, ""
    return list(top_ids), False, "invalid_or_incomplete_permutation_fallback_to_quantitative"


def _rank_metrics(ranks: np.ndarray) -> dict:
    values = np.asarray(ranks, dtype=float)
    return {"queries": int(len(values)), "recall@1": float(np.mean(values <= 1)),
            "recall@5": float(np.mean(values <= 5)), "recall@10": float(np.mean(values <= 10)),
            "mrr": float(np.mean(1.0 / values))}


def _load_cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = list(read_jsonl(path)); result = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"duplicate cached query_id={query_id}: {path}")
        result[query_id] = row
    return result


def _journal_diagnostics(expected: list[str], attempted: dict[str, dict],
                         predictions: dict[str, dict]) -> dict:
    expected_set = set(expected); attempted_set = set(attempted); prediction_set = set(predictions)
    attempted_without_prediction = sorted(attempted_set - prediction_set)
    prediction_without_attempt = sorted(prediction_set - attempted_set)
    unexpected_attempts = sorted(attempted_set - expected_set)
    unexpected_predictions = sorted(prediction_set - expected_set)
    prompt_hash_mismatches = sorted(
        query_id for query_id in attempted_set & prediction_set
        if attempted[query_id].get("prompt_sha256") != predictions[query_id].get("prompt_sha256")
    )
    return {
        "attempted_without_prediction": attempted_without_prediction,
        "prediction_without_attempt": prediction_without_attempt,
        "unexpected_attempts": unexpected_attempts,
        "unexpected_predictions": unexpected_predictions,
        "prompt_hash_mismatches": prompt_hash_mismatches,
    }


def _assert_resumable(expected: list[str], attempted: dict[str, dict],
                      predictions: dict[str, dict]) -> None:
    diagnostics = _journal_diagnostics(expected, attempted, predictions)
    if any(diagnostics.values()):
        compact = {key: values[:3] for key, values in diagnostics.items() if values}
        raise RuntimeError(
            "request/prediction journal mismatch; strict protocol will not retry or repair a "
            f"crash window in place: {json.dumps(compact, sort_keys=True)}"
        )


def _recover_dangling_predictions(cache_path: Path, expected: list[str],
                                  attempted: dict[str, dict], predictions: dict[str, dict]) -> int:
    """Materialize journaled predictions after a crash, without another API request."""
    diagnostics = _journal_diagnostics(expected, attempted, predictions)
    fatal = {key: values for key, values in diagnostics.items()
             if key != "attempted_without_prediction" and values}
    if fatal:
        compact = {key: values[:3] for key, values in fatal.items()}
        raise RuntimeError(f"request/prediction journal is not recoverable: {json.dumps(compact, sort_keys=True)}")
    dangling = diagnostics["attempted_without_prediction"]
    if not dangling:
        return 0
    recovered = []
    for query_id in dangling:
        request = attempted[query_id]; prediction = request.get("prediction")
        if not isinstance(prediction, dict):
            raise RuntimeError(
                "request journal records a consumed physical request but has no self-contained "
                f"prediction for query={query_id}; strict protocol forbids another API call"
            )
        if str(prediction.get("query_id")) != query_id:
            raise RuntimeError(f"journaled prediction query_id mismatch for query={query_id}")
        if request.get("prompt_sha256") != prediction.get("prompt_sha256"):
            raise RuntimeError(f"journaled prediction prompt hash mismatch for query={query_id}")
        recovered.append((query_id, prediction))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for query_id, prediction in recovered:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            predictions[query_id] = prediction
        handle.flush(); os.fsync(handle.fileno())
    return len(recovered)


def _protocol(args, rows: list[dict], ids: list[str], resolved: dict, temperature: float,
              prompt_summary: dict) -> dict:
    prompt_contract = {"prompt_version": PROMPT_VERSION, "top_k": args.top_k,
                       "top_m": args.top_m, "world_mode": args.world_mode,
                       "uses_quantitative_probabilities": True,
                       "uses_structured_llm_evidence": False,
                       "uses_test_labels_in_prompt": False}
    return {
        "protocol": PROTOCOL_VERSION,
        "split": args.split,
        "limit": args.limit,
        "queries": len(rows),
        "query_ids_sha256": _sha256_json([str(row["query_id"]) for row in rows]),
        "selected_rows_sha256": resolved["selected_rows_sha256"],
        "candidate_ids_sha256": _sha256_json(ids),
        "prompt_contract": prompt_contract,
        "prompt_contract_sha256": _sha256_json(prompt_contract),
        "temperature": temperature,
        "model": args.model_name,
        "model_digest": args.model_digest,
        "platform": args.platform,
        "temperature_parameter": 0.0,
        "max_tokens": args.max_tokens,
        "max_input_tokens": args.max_input_tokens,
        "prompt_preflight": prompt_summary,
        "physical_requests_per_uncached_query": 1,
        "automatic_retries": False,
        "input": resolved,
    }


def _prepare_queries(args, rows: list[dict], logits: np.ndarray, ids: list[str],
                     metadata: dict, temperature: float) -> tuple[list[dict], dict]:
    retriever = EmbeddingMemoryRetriever(args.top_m)
    id_to_index = {candidate_id: index for index, candidate_id in enumerate(ids)}
    prepared = []
    for row in rows:
        scores = np.asarray(logits[int(row["_row_index"])], dtype=float)
        probabilities = _softmax(scores, temperature)
        order = np.argsort(-probabilities, kind="stable"); top = order[:args.top_k]
        top_ids = [ids[int(index)] for index in top]
        candidates = [candidate_payload(candidate_id, probabilities[int(index)], metadata, args.world_mode)
                      for index, candidate_id in zip(top, top_ids)]
        memory = retriever.retrieve(row.get("history", []), row.get("context", []))
        prompt = build_prompt(row, memory, candidates, args.world_mode)
        audit = _prompt_audit(prompt)
        true_id = str(row["true_id"])
        if true_id not in id_to_index:
            raise ValueError(f"true_id outside candidate space: query={row['query_id']} true_id={true_id}")
        true_index = id_to_index[true_id]
        prepared.append({
            "row": row, "prompt": prompt, "prompt_audit": audit, "top_ids": top_ids,
            "order": order, "true_id": true_id,
            "quantitative_rank": int(np.where(order == true_index)[0][0]) + 1,
        })
    prompt_summary = {
        "checked_queries": len(prepared),
        "max_original_prompt_chars": max((item["prompt_audit"]["original_prompt_chars"]
                                            for item in prepared), default=0),
        "max_original_prompt_estimated_tokens": max(
            (item["prompt_audit"]["original_prompt_estimated_tokens"] for item in prepared), default=0),
        "max_input_tokens": args.max_input_tokens,
        "truncated": False,
    }
    oversized = [item for item in prepared
                 if item["prompt_audit"]["original_prompt_estimated_tokens"] > args.max_input_tokens]
    if oversized:
        first = oversized[0]
        raise ValueError(
            "prompt preflight failed before any API request; truncation is forbidden: "
            f"query={first['row']['query_id']} estimated_tokens="
            f"{first['prompt_audit']['original_prompt_estimated_tokens']} "
            f"max_input_tokens={args.max_input_tokens} prompt_sha256="
            f"{first['prompt_audit']['original_prompt_sha256']}"
        )
    return prepared, prompt_summary


def _write_npz(path: Path, ordered: list[dict]) -> None:
    np.savez_compressed(path,
                        query_id=np.asarray([str(row["query_id"]) for row in ordered]),
                        user_id=np.asarray([str(row["user_id"]) for row in ordered]),
                        query_index=np.arange(len(ordered)), labels=np.arange(len(ordered)),
                        quantitative_ranks=np.asarray([row["quantitative_rank"] for row in ordered], dtype=np.int32),
                        direct_ranks=np.asarray([row["true_rank"] for row in ordered], dtype=np.int32),
                        ranks=np.asarray([row["true_rank"] for row in ordered], dtype=np.int32))


def _prediction_row(args, item: dict, temperature: float, usage: dict,
                    ranking: list[str], valid: bool, error: str,
                    transport_success: bool, raw: str = "") -> dict:
    row = item["row"]; top_ids = item["top_ids"]; true_id = item["true_id"]
    quantitative_rank = item["quantitative_rank"]
    true_rank = ranking.index(true_id) + 1 if true_id in ranking else quantitative_rank
    return {
        "protocol": PROTOCOL_VERSION, "query_id": str(row["query_id"]),
        "user_id": str(row.get("user_id", "")), "city": str(row.get("city", "")),
        "true_id": true_id, "stage1_top_k": top_ids, "direct_top_k": ranking,
        "quantitative_rank": quantitative_rank, "true_rank": true_rank,
        "true_in_top_k": true_id in top_ids, "valid": valid,
        "transport_success": transport_success, "error": error,
        "api_calls": 1, "physical_attempts": int(usage.get("physical_attempts", 1)),
        "input_tokens": int(usage.get("input_tokens", _estimated_tokens(item["prompt"]))),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "token_source": usage.get("token_source", "estimated"),
        "latency_seconds": float(usage.get("latency_seconds", 0.0)),
        **item["prompt_audit"],
        "response_sha256": _sha256_bytes(raw.encode("utf-8")) if raw else None,
        "candidate_ids_sha256": _sha256_json(top_ids),
        "model": args.model_name, "model_digest": args.model_digest,
        "world_mode": args.world_mode, "temperature": temperature,
    }


def collect(args, wrapper=None) -> dict:
    source = Path(args.input); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if not args.model_digest or args.model_digest == "unknown":
        raise ValueError("a concrete model digest is required for the locked Direct-1Call protocol")
    rows, logits, ids, metadata, resolved = load_split(source, args.limit)
    calibration = json.loads(Path(args.temperature_file).read_text(encoding="utf-8"))
    if int(calibration.get("limit", args.limit)) != args.limit:
        raise ValueError("temperature fit limit differs from Direct-1Call limit")
    expected_candidate_hash = calibration.get("candidate_ids_sha256")
    if expected_candidate_hash and expected_candidate_hash != _sha256_json(ids):
        raise ValueError("temperature candidate space differs from Direct-1Call candidate space")
    temperature = float(calibration["temperature"]["temperature"])
    # Preflight every selected query before constructing a client or issuing any
    # request.  LLMWrapper's legacy prepare_prompt method truncates oversized
    # prompts; this protocol forbids that silent change to the information set.
    prepared, prompt_summary = _prepare_queries(args, rows, logits, ids, metadata, temperature)
    protocol = _protocol(args, rows, ids, resolved, temperature, prompt_summary)
    lock_path = output / "protocol.lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError(f"protocol lock mismatch; use a new output directory: {lock_path}")
    else:
        _atomic_json(lock_path, protocol)
    cache_path = output / "predictions.jsonl"; request_path = output / "requests.jsonl"
    cached = _load_cached(cache_path)
    attempted = _load_cached(request_path)
    expected = [str(row["query_id"]) for row in rows]
    recovered = _recover_dangling_predictions(cache_path, expected, attempted, cached)
    _assert_resumable(expected, attempted, cached)
    if recovered:
        print(json.dumps({"phase": "direct-query-recovery", "split": args.split,
                          "materialized_predictions": recovered}), flush=True)
    if wrapper is None:
        from models.llm_api import LLMWrapper
        wrapper = LLMWrapper(model_name=args.model_name, platform=args.platform)
    wrapper.hyperparams["temperature"] = 0.0
    wrapper.hyperparams["max_tokens"] = args.max_tokens
    wrapper.hyperparams["max_input_tokens"] = args.max_input_tokens
    with cache_path.open("a", encoding="utf-8") as cache_handle, request_path.open("a", encoding="utf-8") as request_handle:
        for position, item in enumerate(prepared, 1):
            row = item["row"]
            query_id = str(row["query_id"])
            if query_id in cached:
                continue
            prompt = item["prompt"]; prompt_sha = item["prompt_audit"]["prompt_sha256"]
            before = len(wrapper.get_call_stats())
            raw = ""; transport_success = True; transport_error = ""
            try:
                raw = wrapper.get_response_once(prompt)
            except Exception as exc:
                transport_success = False
                transport_error = f"{type(exc).__name__}: {exc}"
                stats = wrapper.get_call_stats()[before:]
            else:
                stats = wrapper.get_call_stats()[before:]
            if len(stats) > 1 or (stats and int(stats[0].get("physical_attempts", 1)) != 1):
                raise RuntimeError(f"one-call contract violated for query={query_id}: {stats}")
            usage = stats[0] if stats else {
                "input_tokens": _estimated_tokens(prompt), "output_tokens": 0,
                "latency_seconds": 0.0, "physical_attempts": 1,
                "token_source": "estimated", "success": transport_success,
                "error": transport_error,
            }
            if usage.get("prompt_sha256") not in (None, prompt_sha):
                raise RuntimeError(f"sent prompt hash differs from preflight prompt for query={query_id}")
            if transport_success:
                try:
                    ranking, valid, error = validate_ranking(parse_ranking(raw), item["top_ids"])
                except (ValueError, json.JSONDecodeError) as exc:
                    ranking, valid, error = list(item["top_ids"]), False, f"{type(exc).__name__}: {exc}"
            else:
                ranking, valid, error = list(item["top_ids"]), False, transport_error
            result = _prediction_row(args, item, temperature, usage, ranking, valid, error,
                                     transport_success, raw)
            request = dict(usage, query_id=query_id, split=args.split, api_calls=1,
                           **item["prompt_audit"],
                           status="success" if transport_success else "transport_error",
                           error=error, response_sha256=result["response_sha256"],
                           prediction=result)
            # The request journal is self-contained.  If the process dies after
            # this fsync and before the prediction append, resume materializes
            # the exact stored fallback/result without another model request.
            request_handle.write(json.dumps(request, ensure_ascii=False) + "\n"); request_handle.flush()
            os.fsync(request_handle.fileno())
            cached[query_id] = result
            cache_handle.write(json.dumps(result, ensure_ascii=False) + "\n"); cache_handle.flush()
            os.fsync(cache_handle.fileno())
            if position % args.progress_every == 0 or position == len(rows):
                print(json.dumps({"phase": "direct-query-progress", "split": args.split,
                                  "completed": len(cached), "total": len(rows),
                                  "percent": round(100 * len(cached) / max(len(rows), 1), 2)}), flush=True)
    attempted = _load_cached(request_path)
    cached = _load_cached(cache_path)
    _assert_resumable(expected, attempted, cached)
    ordered = [cached[str(row["query_id"])] for row in rows]
    quantitative_ranks = np.asarray([row["quantitative_rank"] for row in ordered], dtype=np.int32)
    direct_ranks = np.asarray([row["true_rank"] for row in ordered], dtype=np.int32)
    metrics = {
        "protocol": PROTOCOL_VERSION, "split": args.split, "limit": args.limit,
        "quantitative": _rank_metrics(quantitative_ranks), "direct-1call": _rank_metrics(direct_ranks),
        "candidate_recall": float(np.mean([row["true_in_top_k"] for row in ordered])),
        "invalid_output_rate": float(np.mean([not row["valid"] for row in ordered])),
        "transport_error_rate": float(np.mean([row.get("transport_success") is False
                                                for row in ordered])),
        "attempted_without_prediction": 0,
        "api_requests_per_query": float(np.mean([row["api_calls"] for row in ordered])),
        "input_tokens_per_query": float(np.mean([row["input_tokens"] for row in ordered])),
        "output_tokens_per_query": float(np.mean([row["output_tokens"] for row in ordered])),
        "latency_mean": float(np.mean([row["latency_seconds"] for row in ordered])),
        "latency_p95": float(np.quantile([row["latency_seconds"] for row in ordered], .95)),
        "max_original_prompt_chars": prompt_summary["max_original_prompt_chars"],
        "max_original_prompt_estimated_tokens": prompt_summary["max_original_prompt_estimated_tokens"],
        "prompts_truncated": False,
    }
    _atomic_json(output / "metrics.json", metrics); _write_npz(output / "predictions.npz", ordered)
    print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False))
    return metrics


def status(input_path: Path, output_dir: Path, limit: int | None, split: str) -> dict:
    rows, _, _, _, _ = load_split(input_path, limit)
    cached = _load_cached(output_dir / "predictions.jsonl")
    attempted = _load_cached(output_dir / "requests.jsonl")
    expected = [str(row["query_id"]) for row in rows]
    diagnostics = _journal_diagnostics(expected, attempted, cached)
    complete = sum(query_id in cached for query_id in expected)
    completed_rows = [cached[query_id] for query_id in expected if query_id in cached]
    mean_latency = (float(np.mean([row.get("latency_seconds", 0.0) for row in completed_rows]))
                    if completed_rows else None)
    payload = {"split": split, "completed": complete, "total": len(expected),
               "remaining": len(expected) - complete,
               "physical_requests": int(sum(row.get("api_calls", 1) for query_id, row in attempted.items()
                                            if query_id in set(expected))),
               "transport_error_rate": (float(np.mean([
                   row.get("transport_success") is False for row in completed_rows
               ])) if completed_rows else None),
               "attempted_without_prediction": len(diagnostics["attempted_without_prediction"]),
               "prediction_without_attempt": len(diagnostics["prediction_without_attempt"]),
               "prompt_hash_mismatches": len(diagnostics["prompt_hash_mismatches"]),
               "unexpected_attempts": len(diagnostics["unexpected_attempts"]),
               "unexpected_predictions": len(diagnostics["unexpected_predictions"]),
               "mean_latency_seconds": mean_latency,
               "eta_seconds": mean_latency * (len(expected) - complete) if mean_latency is not None else None,
               "gate": ("invalid" if any(diagnostics.values()) else
                        "ready" if complete == len(expected) and (output_dir / "metrics.json").is_file()
                        else "incomplete")}
    print(json.dumps(payload)); return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict one-request query-time LLM comparator")
    subparsers = parser.add_subparsers(dest="action", required=True)
    fit = subparsers.add_parser("fit-temperature")
    fit.add_argument("--validation", type=Path, required=True); fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--limit", type=int, default=200)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--input", required=True); collect_parser.add_argument("--temperature-file", required=True)
    collect_parser.add_argument("--output-dir", required=True); collect_parser.add_argument("--split", choices=("validation", "test"), required=True)
    collect_parser.add_argument("--model-name", required=True); collect_parser.add_argument("--model-digest", required=True)
    collect_parser.add_argument("--platform", default="Ollama"); collect_parser.add_argument("--limit", type=int, default=200)
    collect_parser.add_argument("--top-k", type=int, default=10); collect_parser.add_argument("--top-m", type=int, default=5)
    collect_parser.add_argument("--world-mode", choices=("internal_only", "full"), default="internal_only")
    collect_parser.add_argument("--max-tokens", type=int, default=512); collect_parser.add_argument("--max-input-tokens", type=int, default=2000)
    collect_parser.add_argument("--progress-every", type=int, default=10)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--input", type=Path, required=True); status_parser.add_argument("--output-dir", type=Path, required=True)
    status_parser.add_argument("--split", choices=("validation", "test"), required=True); status_parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.action == "fit-temperature":
        fit_temperature(args.validation, args.output, args.limit)
    elif args.action == "collect":
        try:
            collect(args)
        except LLMServiceUnavailable as exc:
            print(str(exc), flush=True); raise SystemExit(75) from exc
    else:
        status(args.input, args.output_dir, args.limit, args.split)


if __name__ == "__main__":
    main()
