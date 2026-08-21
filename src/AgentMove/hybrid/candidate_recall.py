from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .io import read_jsonl, write_json


def evaluate(path: str | Path) -> Dict[str, float]:
    rows = list(read_jsonl(path))
    if not rows or "_bundle" not in rows[0]:
        raise ValueError("candidate recall evaluator requires bundled Hybrid JSONL")
    bundle = rows.pop(0)["_bundle"]
    logits = np.load(bundle["logits"], mmap_mode="r")
    candidates = [str(value) for value in json.loads(Path(bundle["candidate_ids"]).read_text(encoding="utf-8"))]
    candidate_index = {value: index for index, value in enumerate(candidates)}
    correct = {1: 0, 5: 0, 10: 0, 20: 0}
    ranks = []
    missing = 0
    for row in rows:
        target = candidate_index.get(str(row["true_id"]))
        if target is None:
            missing += 1; continue
        scores = np.asarray(logits[int(row["_row_index"])])
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == target)[0]) + 1
        ranks.append(rank)
        for k in correct:
            correct[k] += int(rank <= k)
    total = len(rows)
    return {
        "queries": total,
        **{f"recall@{k}": value / total for k, value in correct.items()},
        "mrr_full_candidate_space": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "ground_truth_missing_from_candidates": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 candidate recall")
    parser.add_argument("--input", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {}
    for value in args.input:
        name, separator, path = value.partition("=")
        if not separator:
            raise SystemExit("--input must use NAME=JSONL")
        results[name] = evaluate(path)
    write_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
