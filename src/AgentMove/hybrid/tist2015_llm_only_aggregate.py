from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .io import write_json
from .tist2015_protocol import CITIES
METRICS = ["acc@1", "acc@5", "mrr"]


def aggregate(root: Path, model: str, query_limit: int, baseline: str = "llm-zs") -> Dict[str, Any]:
    completed: List[str] = []
    city_metrics: Dict[str, Dict[str, float]] = {}
    problems: Dict[str, str] = {}
    expected_mode = "matched-test-prefix"

    for city in CITIES:
        city_dir = root / city
        metrics_path = city_dir / "metrics.json"
        protocol_path = city_dir / "protocol.json"
        if not metrics_path.exists() or not protocol_path.exists():
            problems[city] = "missing metrics.json or protocol.json"
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_baseline = protocol.get("baseline", "llm-zs")
        if protocol_baseline != baseline:
            problems[city] = f"baseline mismatch: {protocol_baseline}"
            continue
        if protocol.get("model") != model:
            problems[city] = f"model mismatch: {protocol.get('model')}"
            continue
        if protocol.get("sample_mode") != expected_mode:
            problems[city] = f"sample mode mismatch: {protocol.get('sample_mode')}"
            continue
        if protocol.get("requested_limit") != query_limit:
            problems[city] = f"query limit mismatch: {protocol.get('requested_limit')}"
            continue
        if not all(metric in metrics for metric in METRICS):
            problems[city] = "required metric missing"
            continue
        completed.append(city)
        city_metrics[city] = metrics

    interim = {
        metric: statistics.fmean(city_metrics[city][metric] for city in completed)
        for metric in METRICS
    } if completed else {}
    variance = (
        statistics.pvariance(city_metrics[city]["acc@1"] for city in completed)
        if completed else None
    )
    summary: Dict[str, Any] = {
        "protocol": f"TIST2015 {baseline} matched test prefix, at most 200 queries per city",
        "baseline": baseline,
        "model": model,
        "query_limit": query_limit,
        "prediction_count": 5,
        "acc@10": None,
        "completed_cities": completed,
        "missing_or_incompatible_cities": problems,
        "is_complete_12_city": len(completed) == len(CITIES),
        "queries_total": int(sum(city_metrics[city]["queries"] for city in completed)),
        "interim_macro_average": interim,
        "population_variance_acc1": variance,
        "per_city": city_metrics,
    }
    file_slug = baseline.replace("-", "_")
    write_json(root / f"tist2015_{file_slug}_summary.json", summary)

    marker = "" if summary["is_complete_12_city"] else r"^{\dagger}"
    def cell(metric: str) -> str:
        return f"${interim[metric]:.4f}{marker}$" if metric in interim else ""
    cells = f"{cell('acc@1')} & {cell('acc@5')} &  & {cell('mrr')}"
    (root / f"tist2015_{file_slug}_table2_cells.tex").write_text(cells + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched TIST2015 LLM-ZS city results")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", choices=["llm-zs", "llm-mob"], default="llm-zs")
    parser.add_argument("--query-limit", type=int, default=200)
    args = parser.parse_args()
    summary = aggregate(args.input_root, args.model, args.query_limit, args.baseline)
    print(json.dumps(summary, indent=2))
    if not summary["is_complete_12_city"]:
        print("WARNING: partial result only; do not label it as a 12-city macro average.")


if __name__ == "__main__":
    main()
