from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from .io import read_jsonl, write_json


def _score(seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).hexdigest()


def sample_jsonl(source: Path, destination: Path, fraction: float, seed: int) -> Dict[str, Any]:
    rows = list(read_jsonl(source))
    bundle = rows.pop(0) if rows and "_bundle" in rows[0] else None
    count = max(1, min(len(rows), math.ceil(len(rows) * fraction)))
    selected = sorted(rows, key=lambda row: (_score(seed, str(row["query_id"])), str(row["query_id"])))[:count]
    # Restore source order so execution remains deterministic and cache files
    # are easy to compare, while membership is independent of source ordering.
    selected_ids = {str(row["query_id"]) for row in selected}
    selected = [row for row in rows if str(row["query_id"]) in selected_ids]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        if bundle is not None:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "source_queries": len(rows),
        "selected_queries": len(selected),
        "fraction_requested": fraction,
        "fraction_actual": len(selected) / len(rows) if rows else 0.0,
        "seed": seed,
        "selection": "lowest sha256(seed:query_id), then source order",
        "query_ids_sha256": hashlib.sha256("\n".join(sorted(selected_ids)).encode("utf-8")).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create reproducible Hybrid JSONL subsets")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")
    destination = Path(args.output_dir)
    manifest = {
        "protocol": "shanghai-deterministic-subset-v1",
        "validation": sample_jsonl(Path(args.validation), destination / "validation.jsonl", args.fraction, args.seed),
        "test": sample_jsonl(Path(args.test), destination / "test.jsonl", args.fraction, args.seed),
    }
    write_json(destination / "sample_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
