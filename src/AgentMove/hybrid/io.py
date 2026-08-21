from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from .schemas import Prediction, Query


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Hybrid JSONL input file not found: {input_path}. "
            "This CLI expects a schema-conformant JSONL file for the validation/test split. "
            "Provide the file as --validation or --test, or generate it from CGM metadata/logits via "
            "python -m hybrid.cgm_adapter."
        )
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Hybrid JSONL input path is not a regular file: {input_path}. "
            "Provide a JSONL file for --validation or --test."
        )

    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def load_queries(
    path: str | Path, limit: int | None = None, candidate_limit: int | None = None
) -> List[Query]:
    """Load queries without materialising a bundled city-wide candidate matrix.

    Bundled CGM files can contain tens of thousands of POIs.  Keep the logits as
    mmap-backed row views and instantiate Candidate objects only for the leading
    candidates needed by Stage 2.  The complete row remains available to
    temperature calibration and Stage-1 probability normalisation.
    """
    iterator = read_jsonl(path)
    first = next(iterator, None)
    if first is None:
        return []
    if "_bundle" in first:
        bundle = first["_bundle"]
        logits = np.load(bundle["logits"], mmap_mode="r")
        candidate_ids = [str(value) for value in json.loads(Path(bundle["candidate_ids"]).read_text(encoding="utf-8"))]
        candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
        metadata = {}
        if bundle.get("candidate_metadata"):
            metadata = json.loads(Path(bundle["candidate_metadata"]).read_text(encoding="utf-8"))
        expanded = []
        for row in iterator:
            if limit is not None and len(expanded) >= limit:
                break
            row_index = int(row.pop("_row_index"))
            full_logits = logits[row_index]
            keep = len(candidate_ids) if candidate_limit is None else min(candidate_limit, len(candidate_ids))
            if keep < len(candidate_ids):
                indices = np.argpartition(full_logits, -keep)[-keep:]
                indices = indices[np.argsort(-full_logits[indices], kind="stable")]
            else:
                indices = np.argsort(-full_logits, kind="stable")
            row["candidates"] = []
            for column_index in indices:
                candidate_id = candidate_ids[int(column_index)]
                extra = dict(metadata.get(candidate_id, {}))
                row["candidates"].append({
                    "candidate_id": candidate_id,
                    "logit": float(logits[row_index, column_index]),
                    "latitude": extra.pop("latitude", None),
                    "longitude": extra.pop("longitude", None),
                    "category": extra.pop("category", None),
                    "address": extra.pop("address", None),
                    "metadata": extra,
                })
            query_metadata = dict(row.get("metadata", {}))
            query_metadata.update({
                "_bundle_full_logits": full_logits,
                "_bundle_candidate_ids": candidate_ids,
                "_bundle_true_index": candidate_index.get(str(row["true_id"]), -1),
            })
            row["metadata"] = query_metadata
            expanded.append(Query.from_dict(row))
        return expanded
    rows = [first]
    for row in iterator:
        if limit is not None and len(rows) >= limit:
            break
        rows.append(row)
    return [Query.from_dict(row) for row in rows]


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
