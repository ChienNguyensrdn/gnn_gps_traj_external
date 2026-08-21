from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict

from .io import write_json
from .tist2015_llm_only_aggregate import aggregate as aggregate_llm_zs
from .tist2015_protocol import CITIES


RANKING_METRICS = ("acc@1", "acc@5", "acc@10", "mrr")


def _aggregate_hybrid(root: Path, query_limit: int) -> Dict[str, Any]:
    variants: Dict[str, Dict[str, Dict[str, float]]] = {"full": {}, "stage1_only": {}}
    problems: Dict[str, str] = {}
    for city in CITIES:
        loaded: Dict[str, Dict[str, float]] = {}
        for variant in variants:
            path = root / city / variant / "metrics.json"
            if not path.exists():
                problems[city] = f"missing {variant}/metrics.json"
                break
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if not all(metric in metrics for metric in RANKING_METRICS):
                problems[city] = f"required metric missing in {variant}"
                break
            queries = int(metrics.get("queries", 0))
            if not 0 < queries <= query_limit:
                problems[city] = f"invalid query count in {variant}: {queries}"
                break
            loaded[variant] = metrics
        if city in problems:
            continue
        if int(loaded["full"]["queries"]) != int(loaded["stage1_only"]["queries"]):
            problems[city] = "full/stage1 query-count mismatch"
            continue
        for variant in variants:
            variants[variant][city] = loaded[variant]

    completed = [city for city in CITIES if city in variants["full"]]
    macro = {
        variant: {
            metric: statistics.fmean(variants[variant][city][metric] for city in completed)
            for metric in RANKING_METRICS
        }
        for variant in variants
    } if completed else {variant: {} for variant in variants}
    variance = (
        statistics.pvariance(variants["full"][city]["acc@1"] for city in completed)
        if completed else None
    )
    return {
        "completed_cities": completed,
        "missing_or_incompatible_cities": problems,
        "is_complete_12_city": len(completed) == len(CITIES),
        "queries_total": int(sum(variants["full"][city]["queries"] for city in completed)),
        "interim_macro_average": macro,
        "population_variance_acc1": variance,
        "per_city": variants,
        "world_knowledge_status": "no-osm-ablation",
    }


def aggregate(
    hybrid_root: Path,
    llm_root: Path,
    output_dir: Path,
    model: str,
    query_limit: int,
) -> Dict[str, Any]:
    hybrid = _aggregate_hybrid(hybrid_root, query_limit)
    llm_zs = aggregate_llm_zs(llm_root, model, query_limit)
    summary = {
        "protocol": "TIST2015 matched test-prefix Table II, city macro average",
        "model": model,
        "query_limit": query_limit,
        "hybrid_no_osm": hybrid,
        "llm_zs": llm_zs,
        "publication_ready": bool(
            hybrid["is_complete_12_city"] and llm_zs["is_complete_12_city"]
        ),
        "publication_note": (
            "All 12 cities complete, but Hybrid remains a no-OSM ablation and must not be labeled Ours (full)."
            if hybrid["is_complete_12_city"] and llm_zs["is_complete_12_city"] else
            "Partial result: do not label any macro average as the final 12-city result."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "tist2015_table2_summary.json", summary)

    hybrid_marker = "" if hybrid["is_complete_12_city"] else r"^{\dagger}"
    llm_marker = "" if llm_zs["is_complete_12_city"] else r"^{\ddagger}"
    def cells(values: Dict[str, float], marker: str, include_acc10: bool = True) -> str:
        acc10 = f"${values['acc@10']:.4f}{marker}$" if include_acc10 and values else ""
        return (
            f"${values['acc@1']:.4f}{marker}$ & ${values['acc@5']:.4f}{marker}$ & "
            f"{acc10} & ${values['mrr']:.4f}{marker}$"
        ) if values else " &  &  & "

    rows = [
        "% TIST2015 columns only: Acc@1 & Acc@5 & Acc@10 & MRR",
        "CGM (Stage 1 only) & " + cells(hybrid["interim_macro_average"]["stage1_only"], hybrid_marker),
        "LLM-ZS (Qwen2:7b, history-only) & " + cells(
            llm_zs["interim_macro_average"], llm_marker, include_acc10=False
        ),
        "Ours (TIST no-OSM) & " + cells(hybrid["interim_macro_average"]["full"], hybrid_marker),
    ]
    (output_dir / "tist2015_table2_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Hybrid/LLM-ZS TIST2015 Table II rows")
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--llm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query-limit", type=int, default=200)
    args = parser.parse_args()
    result = aggregate(args.hybrid_root, args.llm_root, args.output_dir, args.model, args.query_limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
