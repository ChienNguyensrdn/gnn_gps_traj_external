from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .io import write_json
from .tist2015_protocol import CITIES


METRICS = ("acc@1", "acc@5", "acc@10", "mrr")


def load_city(path: Path, city: str, query_limit: int) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing metrics.json"
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    missing = [key for key in METRICS if key not in row]
    if missing:
        return None, f"missing metrics: {', '.join(missing)}"
    count = int(row.get("count", row.get("queries", 0)))
    if count != query_limit:
        return None, f"expected count={query_limit}, found {count}"
    recorded_city = str(row.get("city", city))
    if recorded_city != city:
        return None, f"city mismatch: {recorded_city}"
    if any(not 0.0 <= float(row[key]) <= 1.0 for key in METRICS):
        return None, "ranking metric outside [0,1]"
    return row, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and aggregate GETNext TIST2015 city results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=200)
    args = parser.parse_args()

    rows: dict[str, dict[str, Any]] = {}
    problems: dict[str, str] = {}
    for city in CITIES:
        row, problem = load_city(args.root / city / "metrics.json", city, args.query_limit)
        if problem:
            problems[city] = problem
        else:
            assert row is not None
            rows[city] = row

    protocols = sorted({str(row.get("protocol", "")) for row in rows.values()})
    seeds = sorted({int(row.get("seed", -1)) for row in rows.values()})
    methods = sorted({str(row.get("method", "")) for row in rows.values()})
    compatible = len(protocols) <= 1 and len(seeds) <= 1 and len(methods) <= 1
    complete = len(rows) == len(CITIES) and compatible
    macro = {
        key: statistics.fmean(float(row[key]) for row in rows.values())
        for key in METRICS
    } if rows else {}
    result = {
        "method": "GETNext sparse reproduction",
        "dataset": "TIST2015",
        "aggregation": "unweighted city macro average",
        "query_limit_per_city": args.query_limit,
        "completed_cities": [city for city in CITIES if city in rows],
        "missing_or_incompatible_cities": problems,
        "city_count": len(rows),
        "is_complete_12_city": complete,
        "publication_ready": complete,
        "protocol_compatible": compatible,
        "protocols": protocols,
        "seeds": seeds,
        "methods": methods,
        "queries_total": sum(int(row.get("count", row.get("queries", 0))) for row in rows.values()),
        "interim_macro_average": macro,
        "population_variance_acc1": (
            statistics.pvariance(float(row["acc@1"]) for row in rows.values()) if rows else None
        ),
        "per_city": {
            city: {"count": int(row.get("count", row.get("queries", 0))), **{key: row[key] for key in METRICS}}
            for city, row in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)

    row_path = args.output.with_suffix(".tex")
    if macro:
        marker = "" if complete else r"^{\dagger}"
        label = "GETNext" if complete else "GETNext (interim)"
        cells = " & ".join(f"${macro[key]:.4f}{marker}$" for key in METRICS)
        row_path.write_text(
            "% Acc@1 & Acc@5 & Acc@10 & MRR\n" + f"{label} & {cells} \\\\\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))
    print(f"summary={args.output}")
    if macro:
        print(f"latex_row={row_path}")
    if not complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
