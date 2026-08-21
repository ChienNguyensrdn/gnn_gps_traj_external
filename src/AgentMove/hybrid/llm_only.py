from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .evidence import LLMServiceUnavailable
from .io import read_jsonl, write_json


def _parse(raw: str) -> List[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(?:\{.*\}|\[.*\])", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("response has no JSON object or list")
        payload = json.loads(match.group(0))
    if isinstance(payload, list):
        ranking = payload
    elif isinstance(payload, dict):
        ranking = payload.get("prediction", payload.get("ranking"))
    else:
        ranking = None
    if not isinstance(ranking, list):
        raise ValueError("JSON must contain a ranking list")
    return [str(value) for value in ranking]


def _pool(row: Dict[str, Any]) -> List[str]:
    result = []
    for event in row.get("history", []) + row.get("context", []):
        if not event:
            continue
        candidate = str(event[-1])
        if candidate not in result:
            result.append(candidate)
    return result


def _prompt_llmzs(row: Dict[str, Any]) -> str:
    # Faithful adaptation of the LLM-ZS prompt printed in AgentMove Appendix 9.5.
    target_time = row.get("target_time")
    try:
        day_of_week = datetime.fromisoformat(str(target_time).replace("Z", "+00:00")).strftime("%A")
    except ValueError:
        day_of_week = None
    return f"""Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.
Predict <next_place_id> by considering:
1. The user's activity trends gleaned from <historical_stays> and the current activities from <context_stays>.
2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.
Present your answer in a JSON object with:
"prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
Do not include line breaks in your output.
The data:
<historical_stays>: {json.dumps(row.get('history', []), ensure_ascii=False)}
<context_stays>: {json.dumps(row.get('context', []), ensure_ascii=False)}
<target_stay>: {json.dumps({"start_time": target_time, "day_of_week": day_of_week, "next_place_id": "unknown"}, ensure_ascii=False)}
"""


def _llmmob_event(event: List[Any]) -> List[Any]:
    timestamp = str(event[0]) if event else ""
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        start_time = parsed.strftime("%I:%M %p")
        weekday = parsed.strftime("%A")
    except ValueError:
        start_time, weekday = timestamp, None
    place_id = str(event[-1]) if event else ""
    return [start_time, weekday, None, place_id]


def _prompt_llmmob(row: Dict[str, Any]) -> str:
    target_time = str(row.get("target_time", ""))
    try:
        parsed = datetime.fromisoformat(target_time.replace("Z", "+00:00"))
        target = [parsed.strftime("%I:%M %p"), parsed.strftime("%A"), None, "unknown"]
    except ValueError:
        target = [target_time, None, None, "unknown"]
    return f"""Your task is to predict a user's next location based on their activity pattern.
Historical and context stays are chronological and formatted as
[start_time, day_of_week, duration_minutes, place_id]. Duration is unavailable in this dataset.
Predict the five most likely place IDs for the target stay in descending probability order.
Consider repeated visits, recent context, and target temporal information.
Return JSON only with "prediction" (five place IDs) and "reason" (a concise justification).
<historical>: {json.dumps([_llmmob_event(event) for event in row.get('history', [])], ensure_ascii=False)}
<context>: {json.dumps([_llmmob_event(event) for event in row.get('context', [])], ensure_ascii=False)}
<target_stay>: {json.dumps(target, ensure_ascii=False)}
"""


def _prompt(row: Dict[str, Any], prompt_type: str = "llmzs") -> str:
    return _prompt_llmmob(row) if prompt_type == "llmmob" else _prompt_llmzs(row)


def _select_one_session_per_user(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Match AgentMove: sort users/trajectories, keep one session per user, take first n."""
    def natural(value: Any) -> tuple[int, Any]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    ordered = sorted(rows, key=lambda row: (
        natural(row.get("user_id", "")),
        natural(row.get("metadata", {}).get("trajectory_id", row.get("query_id", ""))),
        str(row.get("query_id", "")),
    ))
    selected, seen = [], set()
    for row in ordered:
        user_id = str(row.get("user_id", row.get("query_id", "")))
        if user_id in seen:
            continue
        seen.add(user_id); selected.append(row)
        if len(selected) == count:
            break
    return selected


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ranks, input_tokens, output_tokens, latency, calls, valid = [], [], [], [], [], []
    recall = []
    for row in rows:
        ranking = row["ranking"]
        rank = ranking.index(str(row["true_id"])) + 1 if str(row["true_id"]) in ranking else None
        ranks.append(rank); recall.append(str(row["true_id"]) in row["candidate_pool"])
        input_tokens.append(row["input_tokens"]); output_tokens.append(row["output_tokens"])
        latency.append(row["latency_seconds"]); calls.append(row["api_calls"]); valid.append(row["valid"])
    return {
        "queries": len(rows), "acc@1": float(np.mean([rank == 1 for rank in ranks])),
        "acc@5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
        "ndcg@5": float(np.mean([
            1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0 for rank in ranks
        ])),
        "mrr": float(np.mean([1.0 / rank if rank else 0.0 for rank in ranks])),
        "candidate_recall": float(np.mean(recall)), "invalid_output_rate": float(np.mean([not value for value in valid])),
        "input_tokens_mean": float(np.mean(input_tokens)), "output_tokens_mean": float(np.mean(output_tokens)),
        "api_calls_mean": float(np.mean(calls)), "latency_mean": float(np.mean(latency)),
        "latency_median": float(np.median(latency)), "latency_p95": float(np.quantile(latency, 0.95)),
    }


def run(args) -> Dict[str, float]:
    from models.llm_api import LLMWrapper
    source_rows = [row for row in read_jsonl(args.test) if "_bundle" not in row]
    if args.agentmove_sample is not None:
        source_rows = _select_one_session_per_user(source_rows, args.agentmove_sample)
    elif args.limit is not None:
        source_rows = source_rows[:args.limit]
    destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / "predictions.jsonl"
    cached = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cached[str(row["query_id"])] = row
    wrapper = LLMWrapper(model_name=args.model_name, platform=args.platform)
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(source_rows, 1):
            query_id = str(row["query_id"])
            if query_id in cached:
                continue
            pool = _pool(row); prompt = _prompt(row, args.prompt_type)
            ranking, raw, elapsed, error, calls = [], "", 0.0, "", 0
            for attempt in range(args.retries + 1):
                started = time.perf_counter(); calls += 1
                try:
                    raw = wrapper.get_response(prompt); elapsed += time.perf_counter() - started
                except Exception as exc:
                    raise LLMServiceUnavailable(f"LLM-only endpoint failed at {query_id}: {exc}") from exc
                try:
                    parsed = _parse(raw)
                    seen = set(); ranking = []
                    for candidate in parsed:
                        if candidate in pool and candidate not in seen:
                            ranking.append(candidate); seen.add(candidate)
                    ranking = ranking[:5]
                    if ranking:
                        break
                    error = "no_valid_candidate_ids"
                except (ValueError, json.JSONDecodeError) as exc:
                    error = str(exc)
            result = {
                "query_id": query_id, "true_id": str(row["true_id"]), "ranking": ranking,
                "candidate_pool": pool, "valid": bool(ranking), "error": error if not ranking else "",
                "input_tokens": max(1, len(prompt) // 4) * calls,
                "output_tokens": max(1, len(raw) // 4), "api_calls": calls,
                "latency_seconds": elapsed, "model": args.model_name,
                "protocol": f"agentmove-{args.prompt_type}-matched-v1",
            }
            cached[query_id] = result
            handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            if index % 10 == 0:
                print(f"LLM-only progress: {len(cached)}/{len(source_rows)}", flush=True)
    ordered = [cached[str(row["query_id"])] for row in source_rows]
    metrics = _summarize(ordered); write_json(destination / "metrics.json", metrics)
    sample_mode = "one-session-per-user" if args.agentmove_sample is not None else "matched-test-prefix"
    selection = (
        "users and trajectories sorted by ID; one session per user; first n users"
        if args.agentmove_sample is not None else
        "first n queries from the supplied test JSONL, matching the Hybrid city runner"
    )
    baseline = "llm-mob" if args.prompt_type == "llmmob" else "llm-zs"
    write_json(destination / "protocol.json", {
        "name": "AgentMove-faithful LLM-Mob" if baseline == "llm-mob" else "AgentMove-faithful LLM-ZS",
        "baseline": baseline, "model": args.model_name,
        "test": str(Path(args.test).resolve()), "queries": len(source_rows),
        "uses_cgm": False, "uses_bbn": False, "candidate_pool": "unique POI IDs in history + recent context",
        "sample_mode": sample_mode, "selection": selection,
        "requested_users": args.agentmove_sample,
        "requested_limit": args.limit,
        "selected_query_ids_sha256": hashlib.sha256(
            "\n".join(str(row["query_id"]) for row in source_rows).encode("utf-8")
        ).hexdigest(),
        "prediction_count": 5, "metrics": ["acc@1", "acc@5", "ndcg@5"],
        "prompt_source": f"AgentMove prompt implementation: {args.prompt_type}",
        "token_accounting": "estimated len(text)//4",
    })
    print(json.dumps(metrics, indent=2)); return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable history-only LLM next-POI baseline")
    parser.add_argument("--test", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", default="Ollama"); parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-type", choices=["llmzs", "llmmob"], default="llmzs")
    parser.add_argument("--retries", type=int, default=2); parser.add_argument("--limit", type=int)
    parser.add_argument("--agentmove-sample", type=int, default=200,
                        help="One session per sorted user, then first N users; set 0 to disable")
    args = parser.parse_args()
    if args.agentmove_sample == 0:
        args.agentmove_sample = None
    try:
        run(args)
    except LLMServiceUnavailable as exc:
        print(f"LLM service unavailable: {exc}", flush=True); raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
