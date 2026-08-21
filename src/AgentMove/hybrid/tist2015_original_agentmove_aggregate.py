from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import write_json


CITIES = [
    "Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris",
    "Beijing", "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow",
]
METRICS = ["acc@1", "acc@5", "mrr"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate original AgentMove across 12 TIST2015 cities")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--model", default="qwen2:7b")
    parser.add_argument("--query-limit", type=int, default=200)
    args = parser.parse_args()
    root = Path(args.input_root)
    per_city, missing = {}, {}
    for city in CITIES:
        path = root / city / "agentmove" / args.model / "agent_move_v6" / "metrics.json"
        if not path.exists():
            missing[city] = "metrics missing"
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        per_city[city] = row
    complete = len(per_city) == len(CITIES)
    macro = {
        metric: float(np.mean([per_city[city][metric] for city in CITIES]))
        for metric in METRICS
    } if complete else None
    summary = {
        "protocol": "AgentMove original modules, TIST2015 no-OSM matched, city macro average",
        "model": args.model,
        "query_limit": args.query_limit,
        "prediction_count": 5,
        "acc@10": None,
        "completed_cities": [city for city in CITIES if city in per_city],
        "missing_cities": missing,
        "is_complete_12_city": complete,
        "queries_total": sum(int(row["queries"]) for row in per_city.values()),
        "macro_average": macro,
        "population_variance_acc1": (
            float(np.var([per_city[city]["acc@1"] for city in CITIES])) if complete else None
        ),
        "per_city": per_city,
        "world_knowledge_status": "no-osm-matched; category fallback, not publication full",
    }
    output = root / "tist2015_agentmove_summary.json"
    write_json(output, summary)
    cells = root / "tist2015_agentmove_table2_cells.tex"
    if complete:
        cells.write_text(
            f"{macro['acc@1']:.4f} & {macro['acc@5']:.4f} &  & {macro['mrr']:.4f}\n",
            encoding="utf-8",
        )
    else:
        cells.write_text(" &  &  & \n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
