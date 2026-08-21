from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_json


CITIES = [
    "Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris",
    "Beijing", "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow",
]
METRICS = ("acc@1", "acc@5", "acc@10", "mrr")


def stable_descending_rank(scores: np.ndarray, true_index: int) -> int:
    """Rank matching np.argsort(-scores, kind='stable') without sorting."""
    true_score = scores[true_index]
    higher = int(np.count_nonzero(scores > true_score))
    tied_before = int(np.count_nonzero(scores[:true_index] == true_score))
    return higher + tied_before + 1


def evaluate_city(input_path: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    iterator = read_jsonl(input_path)
    first = next(iterator, None)
    if first is None or "_bundle" not in first:
        raise ValueError(f"Expected bundled Markov JSONL: {input_path}")
    bundle = first["_bundle"]
    logits = np.load(bundle["logits"], mmap_mode="r")
    candidate_ids = [
        str(value) for value in json.loads(Path(bundle["candidate_ids"]).read_text(encoding="utf-8"))
    ]
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    ranks: list[int] = []
    rank_rows: list[dict[str, Any]] = []
    for row in iterator:
        if len(ranks) >= limit:
            break
        true_id = str(row["true_id"])
        true_index = candidate_index.get(true_id)
        if true_index is None:
            continue
        row_index = int(row["_row_index"])
        rank = stable_descending_rank(logits[row_index], true_index)
        ranks.append(rank)
        rank_rows.append({"query_id": row["query_id"], "true_id": true_id, "rank": rank})
    if not ranks:
        raise ValueError(f"No evaluable Markov queries: {input_path}")
    values = np.asarray(ranks)
    metrics = {
        "queries": len(ranks),
        "acc@1": float(np.mean(values <= 1)),
        "acc@5": float(np.mean(values <= 5)),
        "acc@10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
        "ndcg@5": float(np.mean([1.0 / math.log2(rank + 1) if rank <= 5 else 0.0 for rank in ranks])),
        "candidate_space": len(candidate_ids),
        "ranking": "full candidate space; stable descending logits",
        "model": "smoothed first-order Markov/Bi-gram",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "ranks.jsonl").open("w", encoding="utf-8") as handle:
        for row in rank_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "protocol.json", {
        "baseline": "markov-bigram",
        "input": str(input_path.resolve()),
        "query_limit": limit,
        "selection": "first test queries in the shared temporal split",
        "metrics": list(METRICS),
    })
    return metrics


def aggregate(input_root: Path, limit: int) -> dict[str, Any]:
    per_city: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}
    for city in CITIES:
        path = input_root / city / "metrics.json"
        if not path.exists():
            missing[city] = "metrics missing"
            continue
        per_city[city] = json.loads(path.read_text(encoding="utf-8"))
    complete = len(per_city) == len(CITIES)
    macro = {
        metric: float(np.mean([per_city[city][metric] for city in CITIES]))
        for metric in METRICS
    } if complete else None
    summary = {
        "protocol": "TIST2015 Markov/Bi-gram matched test prefix, city macro average",
        "query_limit": limit,
        "completed_cities": [city for city in CITIES if city in per_city],
        "missing_cities": missing,
        "is_complete_12_city": complete,
        "queries_total": sum(int(row["queries"]) for row in per_city.values()),
        "macro_average": macro,
        "population_variance_acc1": (
            float(np.var([per_city[city]["acc@1"] for city in CITIES])) if complete else None
        ),
        "per_city": per_city,
    }
    write_json(input_root / "tist2015_markov_summary.json", summary)
    cells = input_root / "tist2015_markov_table2_cells.tex"
    if complete and macro is not None:
        cells.write_text(
            " & ".join(f"{macro[metric]:.4f}" for metric in METRICS) + "\n",
            encoding="utf-8",
        )
    else:
        cells.write_text(" &  &  & \n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate/aggregate TIST2015 Markov/Bi-gram")
    subparsers = parser.add_subparsers(dest="command", required=True)
    city_parser = subparsers.add_parser("city")
    city_parser.add_argument("--input", required=True)
    city_parser.add_argument("--output-dir", required=True)
    city_parser.add_argument("--limit", type=int, default=200)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input-root", required=True)
    aggregate_parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.command == "city":
        result = evaluate_city(Path(args.input), Path(args.output_dir), args.limit)
    else:
        result = aggregate(Path(args.input_root), args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
