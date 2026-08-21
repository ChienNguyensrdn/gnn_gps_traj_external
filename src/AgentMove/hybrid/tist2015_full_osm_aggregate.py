from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .io import write_json
from .tist2015_protocol import CITIES


METRICS = ("acc@1", "acc@5", "acc@10", "mrr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate publication-ready full-OSM TIST2015 Hybrid results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=200)
    args = parser.parse_args()

    per_city = {}
    problems = {}
    for city in CITIES:
        path = args.root / city / "full" / "metrics.json"
        if not path.exists():
            problems[city] = "missing full/metrics.json"
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if not all(metric in row for metric in METRICS):
            problems[city] = "missing ranking metric"
            continue
        queries = int(row.get("queries", 0))
        if not 0 < queries <= args.query_limit:
            problems[city] = f"invalid query count: {queries}"
            continue
        per_city[city] = row

    complete = len(per_city) == len(CITIES)
    macro = {
        metric: statistics.fmean(row[metric] for row in per_city.values())
        for metric in METRICS
    } if per_city else {}
    result = {
        "protocol": "TIST2015 12-city macro average, Hybrid full OSM, matched limit",
        "query_limit": args.query_limit,
        "completed_cities": [city for city in CITIES if city in per_city],
        "missing_or_incompatible_cities": problems,
        "is_complete_12_city": complete,
        "publication_ready": complete,
        "queries_total": sum(int(row["queries"]) for row in per_city.values()),
        "macro_average": macro,
        "population_variance_acc1": statistics.pvariance(
            row["acc@1"] for row in per_city.values()
        ) if per_city else None,
        "per_city": per_city,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
