from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_CITIES = (
    "Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris",
    "Beijing", "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow",
)


def count_source(path: Path, limit: int) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): count += 1
            if count >= limit: break
    return count


def cache_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.is_file():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            try:
                row = json.loads(line); counts[str(row["query_id"])] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def specifications(root: Path, phase2_root: Path, slug: str, limit: int, cities: list[str]):
    for city in cities:
        yield "tist2015", city, root / "data/hybrid/TIST2015" / city, \
            root / f"results/tist2015-hybrid/{slug}/limit-{limit}/no-osm/{city}", \
            phase2_root / "tist2015" / "artifacts/full" / city
    yield "www2019", "Shanghai", root / "data/hybrid/WWW2019/Shanghai", \
        root / f"results/www2019-hybrid/{slug}/limit-{limit}/no-osm/Shanghai", \
        phase2_root / "www2019" / "artifacts/full/Shanghai"
    yield "yjmob100k", "YJMob", root / "data/hybrid/YJMob100K/YJMob", \
        root / f"results/yjmob100k-hybrid/{slug}/limit-{limit}/no-osm/YJMob", \
        phase2_root / "yjmob100k" / "artifacts/full/YJMob"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Phase 2 LLM evidence progress")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--phase2-root", type=Path, default=Path("results/phase2"))
    parser.add_argument("--model-slug", default="qwen2-7b")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    args = parser.parse_args()
    total_queries = complete_queries = total_evidence = 0
    total_rq8 = complete_rq8 = 0
    for dataset, unit, data, cache, output in specifications(
            args.root, args.phase2_root, args.model_slug, args.limit, args.cities):
        validation_expected = count_source(data / "neural_cgm/validation.jsonl", args.limit)
        test_expected = count_source(data / "neural_cgm/test.jsonl", args.limit)
        expected = validation_expected + test_expected
        counts = cache_counts(cache / "evidence_cache.jsonl")
        complete = sum(value >= args.top_k for value in counts.values())
        evidence = sum(counts.values()); percent = 100.0 * complete / expected if expected else 0.0
        rq3 = output / f"rq3/{args.model_slug}/limit-{args.limit}/seed-42/rq3.metrics.json"
        rq8 = output / f"rq8/{args.model_slug}/limit-{args.limit}"
        rq8_validation = count_source(rq8 / "always-cache/validation/predictions.jsonl", validation_expected)
        rq8_test = count_source(rq8 / "always-cache/test/predictions.jsonl", test_expected)
        rq8_complete = rq8_validation + rq8_test
        rq8_runs = len(list(rq8.glob("seed-*/rq8.metrics.json")))
        if rq8_runs >= 50: stage = "ready"
        elif rq8_complete: stage = "rq8-llm"
        elif rq3.is_file(): stage = "rq8-wait"
        elif complete >= expected and expected: stage = "rq3-eval"
        elif evidence: stage = "rq3-llm"
        else: stage = "pending"
        print(f"{dataset:10} {unit:14} {stage:8} "
              f"evidence={complete:4}/{expected:<4} ({percent:6.2f}%) "
              f"rq3={'ready' if rq3.is_file() else 'wait':5} "
              f"rq8-cache={rq8_complete:4}/{expected:<4} rq8-runs={rq8_runs:2}/50")
        total_queries += expected; complete_queries += min(complete, expected); total_evidence += evidence
        total_rq8 += expected; complete_rq8 += min(rq8_complete, expected)
    percent = 100.0 * complete_queries / total_queries if total_queries else 0.0
    rq8_percent = 100.0 * complete_rq8 / total_rq8 if total_rq8 else 0.0
    print(f"TOTAL evidence-queries={complete_queries}/{total_queries} ({percent:.2f}%) "
          f"evidence-rows={total_evidence} rq8-cache={complete_rq8}/{total_rq8} ({rq8_percent:.2f}%)")


if __name__ == "__main__":
    main()
