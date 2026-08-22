from __future__ import annotations

import argparse
import json
from pathlib import Path

from .beliefmove_results import dataset_hash, write_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an experiment metrics file into the BeliefMove-Evo raw schema")
    parser.add_argument("--metrics", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rq", required=True); parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--repository", type=Path, default=Path("../.."))
    parser.add_argument("--dataset-files", type=Path, nargs="*", default=[])
    parser.add_argument("--evaluation-split", choices=["train", "validation", "test"], default="validation")
    args = parser.parse_args(); payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics = payload.get("best", payload.get("metrics", payload))
    metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    extra = {"source_metrics": str(args.metrics.resolve()), "evaluation_split": args.evaluation_split}
    if args.dataset_files:
        missing = [str(path) for path in args.dataset_files if not path.exists()]
        if missing: raise FileNotFoundError(f"dataset provenance files missing: {missing}")
        extra["dataset_hash"] = dataset_hash(args.dataset_files)
    result = write_raw(args.output, args.rq, args.experiment, args.seed, args.dataset, args.config, metrics, args.repository.resolve(), extra)
    print(json.dumps({"output": str(args.output), "metrics": sorted(result["metrics"])}))


if __name__ == "__main__":
    main()
