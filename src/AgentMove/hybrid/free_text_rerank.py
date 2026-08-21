from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .evidence import LLMServiceUnavailable
from .io import read_jsonl, write_json
from .memory import EmbeddingMemoryRetriever


def _parse(raw: str) -> List[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("response has no JSON object")
        payload = json.loads(match.group(0))
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("JSON must contain ranking")
    return [str(value) for value in ranking]


def _load_evidence(path: str) -> Dict[tuple[str, str], Dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        result[(str(row["query_id"]), str(row["evidence"]["candidate_id"]))] = row["evidence"]
    return result


def _prompt(row, memory, candidates) -> str:
    return f"""Directly rerank next-location candidates without a Bayesian network.
Use the quantitative prior, personal-memory evidence, semantic/world evidence and rationales.
Return JSON only: {{"ranking":["candidate_id",...]}} containing every supplied ID once.
Do not invent IDs and do not reveal hidden chain-of-thought.
Target time: {row.get('target_time')}
Recent context: {json.dumps(row.get('context', []), ensure_ascii=False)}
Retrieved personal memory: {json.dumps(memory, ensure_ascii=False)}
Candidates and evidence: {json.dumps(candidates, ensure_ascii=False)}
"""


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ranks = [row["true_rank"] for row in rows]
    return {
        "queries": len(rows), "acc@1": float(np.mean([rank == 1 for rank in ranks])),
        "acc@5": float(np.mean([rank <= 5 for rank in ranks])),
        "acc@10": float(np.mean([rank <= 10 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "candidate_recall": float(np.mean([row["true_in_top_k"] for row in rows])),
        "invalid_output_rate": float(np.mean([not row["valid"] for row in rows])),
        "input_tokens_mean": float(np.mean([row["input_tokens"] for row in rows])),
        "output_tokens_mean": float(np.mean([row["output_tokens"] for row in rows])),
        "api_calls_mean": float(np.mean([row["api_calls"] for row in rows])),
        "latency_mean": float(np.mean([row["latency_seconds"] for row in rows])),
        "latency_median": float(np.median([row["latency_seconds"] for row in rows])),
        "latency_p95": float(np.quantile([row["latency_seconds"] for row in rows], 0.95)),
    }


def run(args) -> Dict[str, float]:
    from models.llm_api import LLMWrapper
    rows = list(read_jsonl(args.test)); bundle = rows.pop(0)["_bundle"]
    if args.limit is not None:
        rows = rows[:args.limit]
    logits = np.load(bundle["logits"], mmap_mode="r")
    ids = [str(value) for value in json.loads(Path(bundle["candidate_ids"]).read_text())]
    metadata = json.loads(Path(bundle["candidate_metadata"]).read_text())
    evidence = _load_evidence(args.evidence_cache)
    temperature = float(json.loads(Path(args.calibration).read_text())["temperature"]["temperature"])
    retriever = EmbeddingMemoryRetriever(args.top_m)
    destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / "predictions.jsonl"; cached = {}
    if cache_path.exists():
        cached = {str(item["query_id"]): item for item in read_jsonl(cache_path)}
    wrapper = LLMWrapper(model_name=args.model_name, platform=args.platform)
    with cache_path.open("a", encoding="utf-8") as handle:
        for position, row in enumerate(rows, 1):
            query_id = str(row["query_id"])
            if query_id in cached:
                continue
            scores = np.asarray(logits[int(row["_row_index"])], dtype=float) / temperature
            order = np.argsort(-scores, kind="stable"); top = order[:args.top_k]
            top_ids = [ids[index] for index in top]
            shifted = scores - scores.max(); probabilities = np.exp(shifted); probabilities /= probabilities.sum()
            payload = []
            for index, candidate_id in zip(top, top_ids):
                item = metadata.get(candidate_id, {}); ev = evidence[(query_id, candidate_id)]
                payload.append({
                    "candidate_id": candidate_id, "prior_probability": float(probabilities[index]),
                    "category": item.get("category"), "address": item.get("address"),
                    "habit_score": ev["habit_score"], "semantic_score": ev["semantic_score"],
                    "habit_rationale": ev.get("habit_rationale", ""), "semantic_rationale": ev.get("semantic_rationale", ""),
                })
            memory = retriever.retrieve(row.get("history", []), row.get("context", []))
            prompt = _prompt(row, memory, payload); raw = ""; ranking = []; elapsed = 0.0; calls = 0; error = ""
            for _ in range(args.retries + 1):
                started = time.perf_counter(); calls += 1
                try:
                    raw = wrapper.get_response(prompt); elapsed += time.perf_counter() - started
                except Exception as exc:
                    raise LLMServiceUnavailable(f"free-text reranker endpoint failed at {query_id}: {exc}") from exc
                try:
                    parsed = _parse(raw); seen = set(); ranking = []
                    for value in parsed:
                        if value in top_ids and value not in seen:
                            ranking.append(value); seen.add(value)
                    if ranking:
                        break
                    error = "no_valid_candidate_ids"
                except (ValueError, json.JSONDecodeError) as exc:
                    error = str(exc)
            valid = set(ranking) == set(top_ids) and len(ranking) == len(top_ids)
            ranking += [value for value in top_ids if value not in ranking]
            tail = [ids[index] for index in order[args.top_k:]]
            true_id = str(row["true_id"])
            if true_id in ranking:
                true_rank = ranking.index(true_id) + 1
            else:
                true_rank = args.top_k + tail.index(true_id) + 1
            result = {
                "query_id": query_id, "true_id": true_id, "ranking_top_k": ranking,
                "true_rank": true_rank, "true_in_top_k": true_id in top_ids, "valid": valid,
                "error": "" if valid else error or "incomplete_ranking_filled_from_stage1",
                "input_tokens": max(1, len(prompt) // 4) * calls, "output_tokens": max(1, len(raw) // 4),
                "api_calls": calls, "latency_seconds": elapsed, "model": args.model_name,
                "protocol": "free-text-rerank-v1",
            }
            cached[query_id] = result; handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            if position % 10 == 0:
                print(f"Free-text rerank progress: {len(cached)}/{len(rows)}", flush=True)
    ordered = [cached[str(row["query_id"])] for row in rows]
    metrics = _metrics(ordered); write_json(destination / "metrics.json", metrics)
    write_json(destination / "protocol.json", {
        "name": "w/o BBN (direct free-text reranking)", "test": str(Path(args.test).resolve()),
        "evidence_cache": str(Path(args.evidence_cache).resolve()), "top_k": args.top_k,
        "uses_same_cgm_candidates": True, "uses_same_embedding_memory": True, "uses_bbn": False,
        "uses_same_structured_evidence": True, "uses_osm_candidate_metadata": True,
        "replacement": "Stage 3 BBN replaced by one direct LLM ranking call",
    })
    print(json.dumps(metrics, indent=2)); return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact free-text reranking ablation without BBN")
    parser.add_argument("--test", required=True); parser.add_argument("--evidence-cache", required=True)
    parser.add_argument("--calibration", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True); parser.add_argument("--platform", default="Ollama")
    parser.add_argument("--top-k", type=int, default=10); parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2); parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except LLMServiceUnavailable as exc:
        print(f"LLM service unavailable: {exc}", flush=True); raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
