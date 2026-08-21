from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def export_queries(logits_path: str, metadata_path: str, candidate_ids_path: str, output_path: str,
                   candidate_metadata_path: str | None = None) -> None:
    """Convert CGM arrays and query metadata to the hybrid JSONL contract.

    `logits.npy` is shaped [queries, locations]. `candidate_ids.json` is a
    location-id list aligned to its second axis. Metadata is JSONL and must
    contain query_id, user_id, city, true_id and may include history/context,
    target_time, backbone and candidate_metadata keyed by candidate id.
    """
    logits = np.load(logits_path)
    candidate_ids = [str(value) for value in json.loads(Path(candidate_ids_path).read_text(encoding="utf-8"))]
    metadata = [json.loads(line) for line in Path(metadata_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    shared_candidate_metadata: Dict[str, Dict[str, Any]] = {}
    if candidate_metadata_path:
        shared_candidate_metadata = json.loads(Path(candidate_metadata_path).read_text(encoding="utf-8"))
    if logits.ndim != 2:
        raise ValueError("logits.npy must have shape [queries, locations]")
    if logits.shape != (len(metadata), len(candidate_ids)):
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, metadata={len(metadata)}, candidates={len(candidate_ids)}"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_bundle": {
            "logits": str(Path(logits_path).resolve()),
            "candidate_ids": str(Path(candidate_ids_path).resolve()),
            "candidate_metadata": str(Path(candidate_metadata_path).resolve()) if candidate_metadata_path else None,
        }}, ensure_ascii=False) + "\n")
        for row_index, row in enumerate(metadata):
            row["query_id"] = str(row["query_id"])
            row["user_id"] = str(row["user_id"])
            row["true_id"] = str(row["true_id"])
            row.pop("candidate_metadata", None)
            row["_row_index"] = row_index
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GETNext/CGM logits to hybrid query JSONL")
    parser.add_argument("--logits", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-metadata", help="Optional shared candidate metadata JSON")
    args = parser.parse_args()
    export_queries(args.logits, args.metadata, args.candidate_ids, args.output, args.candidate_metadata)


if __name__ == "__main__":
    main()
