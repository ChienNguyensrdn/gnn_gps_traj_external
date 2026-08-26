from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from .free_text_rerank import _metrics, _parse
from .io import read_jsonl, write_json


def stable_seed(query_id: str, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def shuffled(values, query_id, seed, namespace):
    result = list(values); random.Random(stable_seed(query_id, seed, namespace)).shuffle(result)
    if len(result) > 1 and result == list(values): result = result[1:] + result[:1]
    return result


def donor_histories(rows):
    by_user = {}
    for row in sorted(rows, key=lambda item: (str(item["user_id"]), str(item["query_id"]))):
        by_user.setdefault(str(row["user_id"]), row.get("history", []))
    users = sorted(by_user)
    if len(users) < 2: return {str(row["query_id"]): [] for row in rows}
    donor = {user: by_user[users[(index + 1) % len(users)]] for index, user in enumerate(users)}
    return {str(row["query_id"]): donor[str(row["user_id"])] for row in rows}


def random_poi_context(context, metadata, query_id, seed):
    ids = sorted(str(value) for value in metadata)
    if not ids: return []
    rng = random.Random(stable_seed(query_id, seed, "random-poi")); result = []
    for item in context:
        original = str(item[2]) if isinstance(item, (list, tuple)) and len(item) > 2 else None
        choices = [candidate_id for candidate_id in ids if candidate_id != original] or ids
        candidate_id = choices[rng.randrange(len(choices))]; candidate = metadata.get(candidate_id, {})
        timestamp = item[0] if isinstance(item, (list, tuple)) and item else None
        result.append([timestamp, candidate.get("category", "Unknown"), candidate_id])
    return result


def perturb(row, variant, donors, metadata, seed):
    history = list(row.get("history", [])); context = list(row.get("context", [])); query_id = str(row["query_id"])
    if variant == "memory-shuffled": history = shuffled(history, query_id, seed, "memory")
    elif variant == "memory-random-user": history = list(donors[query_id])
    elif variant == "memory-none": history = []
    elif variant == "context-shuffled": context = shuffled(context, query_id, seed, "context")
    elif variant == "context-random-poi": context = random_poi_context(context, metadata, query_id, seed)
    elif variant == "context-none": context = []
    elif variant != "memory-true": raise ValueError(f"unknown RQ9 variant: {variant}")
    return history, context


def prompt(row, history, context, candidates):
    return f"""Rerank next-location candidates using only the supplied mobility memory and current context.
Return JSON only: {{"ranking":["candidate_id",...]}} containing every supplied ID once.
Do not invent candidate IDs and do not reveal hidden chain-of-thought.
Target time: {row.get('target_time')}
Personal mobility memory: {json.dumps(history, ensure_ascii=False)}
Current trajectory context: {json.dumps(context, ensure_ascii=False)}
Candidates: {json.dumps(candidates, ensure_ascii=False)}
"""


def resolve_bundle(path: Path, value):
    target = Path(value)
    if target.is_file(): return target
    for candidate in (path.parent / target.name, path.parent.parent / target.name):
        if candidate.is_file(): return candidate
    raise FileNotFoundError(f"missing bundle artifact: {target}")


def run(args):
    from models.llm_api import LLMWrapper
    rows = list(read_jsonl(args.test)); bundle = rows.pop(0)["_bundle"]; rows = rows[:args.limit] if args.limit else rows
    logits = np.load(resolve_bundle(Path(args.test), bundle["logits"]), mmap_mode="r")
    ids = [str(value) for value in json.loads(resolve_bundle(Path(args.test), bundle["candidate_ids"]).read_text())]
    metadata = json.loads(resolve_bundle(Path(args.test), bundle["candidate_metadata"]).read_text())
    temperature = float(json.loads(Path(args.calibration).read_text())["temperature"]["temperature"])
    donors = donor_histories(rows); destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / "predictions.jsonl"; cached = {}
    if cache_path.exists(): cached = {str(row["query_id"]): row for row in read_jsonl(cache_path)}
    wrapper = LLMWrapper(model_name=args.model_name, platform=args.platform)
    with cache_path.open("a", encoding="utf-8") as handle:
        for position, row in enumerate(rows, 1):
            query_id = str(row["query_id"])
            if query_id in cached: continue
            scores = np.asarray(logits[int(row["_row_index"])], dtype=float) / temperature
            order = np.argsort(-scores, kind="stable"); top = order[:args.top_k]; top_ids = [ids[index] for index in top]
            shifted = scores - scores.max(); probabilities = np.exp(shifted); probabilities /= probabilities.sum()
            candidates = [{"candidate_id": candidate_id, "prior_probability": float(probabilities[index]),
                           "category": metadata.get(candidate_id, {}).get("category"),
                           "address": metadata.get(candidate_id, {}).get("address")} for index, candidate_id in zip(top, top_ids)]
            history, context = perturb(row, args.variant, donors, metadata, args.seed)
            text = prompt(row, history, context, candidates); raw = ""; ranking = []; calls = 0; elapsed = 0.0; error = ""
            for _ in range(args.retries + 1):
                started = time.perf_counter(); calls += 1; raw = wrapper.get_response(text); elapsed += time.perf_counter() - started
                try:
                    parsed = _parse(raw); seen = set(); ranking = [value for value in parsed if value in top_ids and not (value in seen or seen.add(value))]
                    if ranking: break
                    error = "no_valid_candidate_ids"
                except (ValueError, json.JSONDecodeError) as exc: error = str(exc)
            valid = set(ranking) == set(top_ids) and len(ranking) == len(top_ids); ranking += [value for value in top_ids if value not in ranking]
            true_id = str(row["true_id"]); tail = [ids[index] for index in order[args.top_k:]]
            true_rank = ranking.index(true_id) + 1 if true_id in ranking else args.top_k + tail.index(true_id) + 1
            result = {"query_id": query_id, "true_id": true_id, "true_rank": true_rank,
                      "true_in_top_k": true_rank <= args.top_k, "ranking_top_k": ranking,
                      "valid": valid, "error": "" if valid else error or "incomplete_ranking_filled",
                      "input_tokens": max(1, len(text) // 4) * calls, "output_tokens": max(1, len(raw) // 4),
                      "api_calls": calls, "latency_seconds": elapsed, "variant": args.variant,
                      "model": args.model_name, "protocol": "rq9-semantic-prompt-v1"}
            cached[query_id] = result; handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            if position % 10 == 0: print(f"RQ9 {args.variant}: {len(cached)}/{len(rows)}", flush=True)
    ordered = [cached[str(row["query_id"])] for row in rows]
    # Backward-compatible migration for caches produced before
    # true_in_top_k was added. No LLM calls need to be repeated.
    for row in ordered: row.setdefault("true_in_top_k", row["true_rank"] <= args.top_k)
    result = _metrics(ordered)
    result.update({"variant": args.variant, "seed": args.seed, "limit": args.limit, "protocol": "rq9-semantic-prompt-v1"})
    write_json(destination / "metrics.json", result); print(json.dumps(result, indent=2)); return result


def main():
    parser = argparse.ArgumentParser(description="RQ9 semantic memory/context verification with resumable LLM caches")
    parser.add_argument("--test", required=True); parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--variant", required=True)
    parser.add_argument("--model-name", required=True); parser.add_argument("--platform", default="Ollama")
    parser.add_argument("--top-k", type=int, default=10); parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--retries", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__": main()
