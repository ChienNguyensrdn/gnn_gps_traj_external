from __future__ import annotations

"""Leakage-safe orchestration for the faithful BeliefMove experiment.

Only the inexpensive preparation and offline-teacher pilot stages live here.
The confirmatory training/evaluation commands are intentionally registered as
``NOT_IMPLEMENTED`` until their artifacts and gates have been reviewed.  This
prevents a shell driver from silently substituting an older Phase-2 protocol.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .ontology import (
    ACTIVITY_ONTOLOGY,
    DESTINATION_ONTOLOGY,
    INTENT_ONTOLOGY,
    category_to_activity,
    category_to_destination,
    derive_intent,
    ontology_contract,
    ontology_hash,
)
from .regions import (
    UNKNOWN_REGION,
    AnonymousCellRegionVocabulary,
    MetricSpatialRegionVocabulary,
)
from .schema import SCHEMA_HASH, StructuredBelief, hash_text
from .pilot import PilotEvaluationError, evaluate_pilot

OTHER_REGION = "__OTHER__"
PREPARE_VERSION = "beliefmove.prepare.v1"
TEACHER_VERSION = "beliefmove.teacher-cache.v2"
PROMPT_VERSION = "beliefmove.structured-teacher.v2"
RESPONSE_SCHEMA_VERSION = "beliefmove.teacher-response-schema.v2"
NORMALIZATION_VERSION = "beliefmove.teacher-l1-weights.v1"
FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "true_id",
        "target",
        "target_id",
        "target_poi_id",
        "target_region",
        "raw_target_region",
        "target_time",
        "label",
    }
)
PILOT_UNITS = {
    ("tist2015", "Tokyo"),
    ("www2019", "Shanghai"),
    ("yjmob100k", "YJMob"),
}
PILOT_BASE_REQUESTED = {"train": 500, "validation": 200}
PILOT_REPEAT_REQUESTED = {"train": 0, "validation": 50}
PILOT_REPEAT_COUNT = 3
SMOKE_REQUESTED = {"train": 20, "validation": 20}
FUTURE_COMMANDS = {
    "fit",
    "quantitative-teacher",
    "train",
    "test-bank",
    "evaluate",
    "controls",
    "aggregate",
    "report",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable(path: Path, text: str) -> None:
    """Create an artifact or verify byte-for-byte identity on resume."""

    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"immutable artifact conflict; use a new output root: {path}")
        return
    _atomic_text(path, text)


def _write_json_immutable(path: Path, payload: Any) -> None:
    _write_immutable(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl_immutable(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(_canonical_json(row) + "\n" for row in rows)
    _write_immutable(path, text)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _index_unique(rows: Iterable[Mapping[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"duplicate query_id={query_id} in {path}")
        result[query_id] = dict(row)
    return result


def _resolve_csv(data_base: Path, split: str) -> Path:
    name = "val.csv" if split == "validation" else f"{split}.csv"
    return data_base / "getnext" / name


def _column(fieldnames: Sequence[str], choices: Sequence[str], *, required: bool = True) -> str | None:
    for value in choices:
        if value in fieldnames:
            return value
    if required:
        raise ValueError(f"CSV is missing one of required columns: {list(choices)}")
    return None


def _candidate_id(
    raw: object,
    candidate_ids: Sequence[str],
    candidate_lookup: set[str] | frozenset[str] | None = None,
) -> str:
    value = str(raw).strip()
    lookup = candidate_lookup if candidate_lookup is not None else frozenset(candidate_ids)
    if value in lookup:
        return value
    try:
        index = int(float(value))
    except (TypeError, ValueError):
        return value
    return candidate_ids[index] if 0 <= index < len(candidate_ids) else value


def _parse_timestamp(raw: object, *, path: Path, row_number: int, column: str) -> tuple[str, int]:
    """Return canonical UTC text and an exact chronological key.

    Prepared datasets normally use ``YYYY-mm-dd HH:MM:SS`` without an offset;
    that representation is defined as UTC by the preprocessing contract.
    ISO-8601 offsets and ``Z`` are also accepted and normalized to UTC.  The
    parser deliberately rejects ambiguous/non-date values instead of letting
    source order or lexicographic ordering determine trajectory chronology.
    """

    text = str(raw if raw is not None else "").strip()
    if not text:
        raise ValueError(f"missing timestamp at {path}:{row_number} column={column}")
    candidate = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(
            f"unparseable timestamp at {path}:{row_number} column={column}: {text!r}"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    normalized = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = normalized - epoch
    key = ((delta.days * 86_400 + delta.seconds) * 1_000_000) + delta.microseconds
    canonical = normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return canonical, key


def _read_split(path: Path, candidate_ids: Sequence[str]) -> list[dict[str, Any]]:
    candidate_lookup = frozenset(candidate_ids)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        user_col = _column(fieldnames, ("user_id", "user"))
        trajectory_col = _column(fieldnames, ("trajectory_id", "traj_id"))
        poi_col = _column(fieldnames, ("POI_id", "venue_id", "poi_id"))
        # ``timezone`` is an AgentMove compatibility field and is used only
        # when no actual UTC/datetime column exists.
        time_col = _column(fieldnames, ("UTC_time", "utc_time", "datetime", "timezone"))
        latitude_col = _column(fieldnames, ("latitude", "Latitude", "lat"), required=False)
        longitude_col = _column(fieldnames, ("longitude", "Longitude", "lon", "lng"), required=False)
        category_col = _column(
            fieldnames,
            ("POI_catname", "venue_category_name", "poi_category", "category"),
            required=False,
        )
        rows: list[dict[str, Any]] = []
        for source_position, row in enumerate(reader):
            poi_id = _candidate_id(row.get(poi_col, ""), candidate_ids, candidate_lookup)
            observed_time, chronological_key = _parse_timestamp(
                row.get(time_col, ""),
                path=path,
                row_number=source_position + 2,
                column=time_col,
            )
            rows.append(
                {
                    "user_id": str(row.get(user_col, "")),
                    "trajectory_id": str(row.get(trajectory_col, "")),
                    "poi_id": poi_id,
                    "observed_time": observed_time,
                    "latitude": row.get(latitude_col) if latitude_col else None,
                    "longitude": row.get(longitude_col) if longitude_col else None,
                    "activity_hint": str(row.get(category_col, "") or "") if category_col else "",
                    "_source_position": source_position,
                    "_chronological_key": chronological_key,
                }
            )
            if len(rows) % 100_000 == 0:
                print(
                    json.dumps(
                        {
                            "action": "prepare-progress",
                            "stage": "read-split",
                            "path": str(path),
                            "rows": len(rows),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    return rows


def _valid_coordinate(latitude: object, longitude: object) -> bool:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def _fit_representation(
    dataset: str,
    train_rows: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    region_cell_meters: float,
) -> tuple[dict[str, Any], dict[str, str], Counter[str]]:
    candidate_lookup = frozenset(candidate_ids)
    by_poi: dict[str, list[tuple[float, float]]] = defaultdict(list)
    poi_counts: Counter[str] = Counter()
    for row in train_rows:
        poi_id = str(row["poi_id"])
        if poi_id not in candidate_lookup:
            continue
        poi_counts[poi_id] += 1
        if _valid_coordinate(row.get("latitude"), row.get("longitude")):
            by_poi[poi_id].append((float(row["latitude"]), float(row["longitude"])))

    if dataset == "yjmob100k":
        # YJMob locations are anonymous grid cells, not GPS coordinates.
        vocabulary = AnonymousCellRegionVocabulary.fit(
            (row["poi_id"] for row in train_rows if str(row["poi_id"]) in candidate_lookup),
            split="train",
        )
        poi_region = {
            poi_id: vocabulary.assign(poi_id)
            for poi_id in candidate_ids
            if vocabulary.assign(poi_id) != UNKNOWN_REGION
        }
        region_counts = Counter(
            {region: vocabulary.count(region) for region in vocabulary.region_ids}
        )
        representation = vocabulary.to_dict()
        representation.update({"kind": "anonymous-500m-cell", "cell_size_meters": 500})
        return representation, poi_region, region_counts

    points = [point for values in by_poi.values() for point in values]
    if not points:
        raise ValueError("training split has no valid latitude/longitude for region fitting")
    vocabulary = MetricSpatialRegionVocabulary.fit(
        points,
        split="train",
        cell_size_meters=region_cell_meters,
    )
    poi_region: dict[str, str] = {}
    for poi_id, values in sorted(by_poi.items()):
        latitude = sum(item[0] for item in values) / len(values)
        longitude = sum(item[1] for item in values) / len(values)
        poi_region[poi_id] = vocabulary.assign(latitude, longitude)
    region_counts = Counter()
    for poi_id, count in poi_counts.items():
        region_counts[poi_region.get(poi_id, UNKNOWN_REGION)] += count
    representation = vocabulary.to_dict()
    representation.update({"kind": "local-equirectangular-metric-grid"})
    return representation, poi_region, region_counts


def _prefix_queries(
    dataset: str,
    unit: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    poi_region: Mapping[str, str],
    candidate_regions: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["user_id"]), str(row["trajectory_id"]))].append(row)
    queries: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for (user_id, trajectory_id), trajectory in sorted(grouped.items()):
        ordered = sorted(
            trajectory,
            key=lambda item: (int(item["_chronological_key"]), int(item["_source_position"])),
        )
        for stop in range(1, len(ordered)):
            observed = ordered[:stop]
            target = ordered[stop]
            prefix = [
                {
                    "observed_time": str(item["observed_time"]),
                    "region": poi_region.get(str(item["poi_id"]), UNKNOWN_REGION),
                    "activity_hint": str(item.get("activity_hint", "")),
                }
                for item in observed
            ]
            identity = {
                "dataset": dataset,
                "unit": unit,
                "split": split,
                "user_id": user_id,
                "trajectory_id": trajectory_id,
                "prefix_length": stop,
                "prefix": prefix,
                # AgentMove defines the next-event timestamp as a known query
                # covariate.  The next POI/category/region remain hidden.
                "query_time": str(target["observed_time"]),
            }
            query_id = f"{split}:{hash_text(_canonical_json(identity))[:24]}"
            query = {
                "query_id": query_id,
                "dataset": dataset,
                "unit": unit,
                "split": split,
                "user_id": user_id,
                "trajectory_id": trajectory_id,
                "prefix_length": stop,
                "observed_prefix": prefix,
                "query_time": str(target["observed_time"]),
                "candidate_regions": list(candidate_regions),
                "source_hash": _canonical_hash(identity),
            }
            # The target is deliberately kept in a different artifact.  Never
            # merge this row into ``query`` or a teacher prompt.
            raw_target_region = poi_region.get(str(target["poi_id"]), UNKNOWN_REGION)
            target_region = (
                raw_target_region if raw_target_region in candidate_regions else OTHER_REGION
            )
            prefix_poi_ids = [str(item["poi_id"]) for item in observed]
            prefix_region_ids = [
                poi_region.get(str(item["poi_id"]), UNKNOWN_REGION) for item in observed
            ]
            target_row = {
                "query_id": query_id,
                "split": split,
                "target_poi_id": str(target["poi_id"]),
                "target_region": target_region,
                "raw_target_region": raw_target_region,
                "target_activity": category_to_activity(target.get("activity_hint")),
                "target_destination": category_to_destination(target.get("activity_hint")),
                "target_intent": derive_intent(
                    prefix_poi_ids,
                    prefix_region_ids,
                    target_poi_id=target["poi_id"],
                    target_region_id=raw_target_region,
                ),
            }
            queries.append(query)
            targets.append(target_row)
    return queries, targets


def audit(args: argparse.Namespace) -> dict[str, Any]:
    data_base = Path(args.data_base)
    required = [data_base / "candidate_ids.json"] + [
        _resolve_csv(data_base, split) for split in ("train", "validation", "test")
    ]
    checkpoint = data_base / "neural_cgm" / "best.pt"
    required.append(checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    payload = {
        "action": "audit",
        "dataset": args.dataset,
        "unit": args.unit,
        "gate": "ready" if not missing else "incomplete",
        "missing": missing,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return payload


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_base, output_root = Path(args.data_base), Path(args.output_root)
    base_audit = audit(args)
    if base_audit["missing"]:
        raise FileNotFoundError(f"missing required source artifacts: {base_audit['missing']}")
    candidate_path = data_base / "candidate_ids.json"
    candidate_ids = [str(value) for value in json.loads(candidate_path.read_text(encoding="utf-8"))]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids.json must be a non-empty unique list")
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        source_rows[split] = _read_split(_resolve_csv(data_base, split), candidate_ids)
        print(
            json.dumps(
                {
                    "action": "prepare-progress",
                    "dataset": args.dataset,
                    "unit": args.unit,
                    "stage": "split-read",
                    "split": split,
                    "rows": len(source_rows[split]),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    representation, poi_region, region_counts = _fit_representation(
        args.dataset, source_rows["train"], candidate_ids, args.region_cell_meters
    )
    top_regions = [region for region, _ in sorted(region_counts.items(), key=lambda item: (-item[1], item[0]))[:20]]
    candidate_regions = list(dict.fromkeys(top_regions + [UNKNOWN_REGION, OTHER_REGION]))
    queries_by_split: dict[str, list[dict[str, Any]]] = {}
    targets_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        queries_by_split[split], targets_by_split[split] = _prefix_queries(
            args.dataset, args.unit, split, source_rows[split], poi_region, candidate_regions
        )
        print(
            json.dumps(
                {
                    "action": "prepare-progress",
                    "dataset": args.dataset,
                    "unit": args.unit,
                    "stage": "prefixes-built",
                    "split": split,
                    "queries": len(queries_by_split[split]),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    _write_json_immutable(output_root / "representation" / "regions.json", representation)
    _write_json_immutable(
        output_root / "representation" / "ontology.json", ontology_contract()
    )
    _write_json_immutable(
        output_root / "representation" / "poi_region_map.json",
        {
            "fit_split": "train",
            "mapping": dict(sorted(poi_region.items())),
            "mapping_hash": _canonical_hash(dict(sorted(poi_region.items()))),
        },
    )
    for split in ("train", "validation", "test"):
        _write_jsonl_immutable(output_root / "queries" / f"{split}.jsonl", queries_by_split[split])
        _write_jsonl_immutable(output_root / "targets" / f"{split}.jsonl", targets_by_split[split])

    source_hashes = {
        "candidate_ids": _file_hash(candidate_path),
        "neural_checkpoint": _file_hash(data_base / "neural_cgm" / "best.pt"),
        **{split: _file_hash(_resolve_csv(data_base, split)) for split in ("train", "validation", "test")},
    }
    split_hashes = {
        split: {
            "queries": _file_hash(output_root / "queries" / f"{split}.jsonl"),
            "targets": _file_hash(output_root / "targets" / f"{split}.jsonl"),
            "query_count": len(queries_by_split[split]),
        }
        for split in ("train", "validation", "test")
    }
    manifest = {
        "version": PREPARE_VERSION,
        "dataset": args.dataset,
        "unit": args.unit,
        "fit_split": "train",
        "region_cell_meters": float(args.region_cell_meters),
        "activity_ontology": list(ACTIVITY_ONTOLOGY),
        "intent_ontology": list(INTENT_ONTOLOGY),
        "destination_ontology": list(DESTINATION_ONTOLOGY),
        "ontology_hash": ontology_hash(),
        "candidate_regions": candidate_regions,
        "representation_hash": _canonical_hash(representation),
        "source_hashes": source_hashes,
        "dataset_hash": _canonical_hash(source_hashes),
        "split_hashes": split_hashes,
        "leakage_contract": {
            "teacher_allowed_splits": ["train", "validation"],
            "target_artifacts_separate": True,
            "target_fields_in_query": False,
        },
    }
    _write_json_immutable(output_root / "manifests" / "prepare_manifest.json", manifest)
    payload = {
        "action": "prepare",
        "dataset": args.dataset,
        "unit": args.unit,
        "gate": "ready",
        "queries": {split: len(rows) for split, rows in queries_by_split.items()},
        "output": str(output_root),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return payload


def _stable_sample(rows: Sequence[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("teacher sample limit must be non-negative")
    ordered = sorted(
        rows,
        key=lambda row: (hash_text(f"{seed}:{row['query_id']}"), str(row["query_id"])),
    )
    return ordered[: min(limit, len(ordered))]


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for item in value.values():
            result.update(_nested_keys(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_nested_keys(item))
        return result
    return set()


def build_teacher_prompt(query: Mapping[str, Any]) -> str:
    overlap = FORBIDDEN_TARGET_KEYS & _nested_keys(query)
    if overlap:
        raise ValueError(f"teacher query contains forbidden target fields: {sorted(overlap)}")
    support = [str(value) for value in query["candidate_regions"]]
    schema = {
        "activity": {label: "non-negative relative weight" for label in ACTIVITY_ONTOLOGY},
        "intent": {label: "non-negative relative weight" for label in INTENT_ONTOLOGY},
        "destination": {
            label: "non-negative relative weight" for label in DESTINATION_ONTOLOGY
        },
        "next_region": {label: "non-negative relative weight" for label in support},
        "confidence": "number in [0,1]",
    }
    observed = {
        "dataset": query["dataset"],
        "unit": query["unit"],
        "user_id": query["user_id"],
        "trajectory_id": query["trajectory_id"],
        "prefix_length": query.get("prefix_length", len(query["observed_prefix"])),
        "query_time": query["query_time"],
        # Preserve causality while bounding the prompt explicitly.  Relying on
        # LLMWrapper's suffix truncation could remove the schema silently.
        "observed_prefix_tail": list(query["observed_prefix"])[-20:],
        "candidate_regions": support,
    }
    return (
        "You are an offline mobility-belief teacher. Infer uncertainty from observed history only.\n"
        "Do not invent labels or regions. Return exactly one JSON object, no markdown and no explanation.\n"
        "Every distribution must contain exactly the shown keys and finite non-negative relative weights.\n"
        "Weights do not need to sum to 1, but each distribution must have positive total mass. "
        "The collector deterministically L1-normalizes them.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Fixed output schema: {_canonical_json(schema)}\n"
        f"Observed causal input: {_canonical_json(observed)}\n"
    )


def _teacher_weight_schema(keys: Sequence[str]) -> dict[str, Any]:
    labels = [str(key) for key in keys]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("teacher response support must be non-empty and unique")
    return {
        "type": "object",
        "properties": {
            key: {"type": "number", "minimum": 0.0} for key in labels
        },
        "required": labels,
        "additionalProperties": False,
    }


def build_teacher_response_format(support: Sequence[str]) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "activity": _teacher_weight_schema(ACTIVITY_ONTOLOGY),
            "intent": _teacher_weight_schema(INTENT_ONTOLOGY),
            "destination": _teacher_weight_schema(DESTINATION_ONTOLOGY),
            "next_region": _teacher_weight_schema(support),
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["activity", "intent", "destination", "next_region", "confidence"],
        "additionalProperties": False,
    }
    schema_hash = _canonical_hash(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"beliefmove_teacher_v2_{schema_hash[:12]}",
            "strict": True,
            "schema": schema,
        },
    }


def _normalizable_weight_map(
    value: Any,
    keys: Sequence[str],
    name: str,
) -> tuple[dict[str, float], float]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly the fixed support")
    result: dict[str, float] = {}
    for key in keys:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name}.{key} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name}.{key} must be finite and non-negative")
        result[key] = number
    total = math.fsum(result.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return {key: result[key] / total for key in keys}, total


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_teacher_response(raw: str, support: Sequence[str]) -> dict[str, Any]:
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    expected = {"activity", "intent", "destination", "next_region", "confidence"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("teacher JSON fields do not match the fixed schema")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0,1]")
    activity, activity_sum = _normalizable_weight_map(
        payload["activity"], ACTIVITY_ONTOLOGY, "activity"
    )
    intent, intent_sum = _normalizable_weight_map(
        payload["intent"], INTENT_ONTOLOGY, "intent"
    )
    destination, destination_sum = _normalizable_weight_map(
        payload["destination"], DESTINATION_ONTOLOGY, "destination"
    )
    next_region, next_region_sum = _normalizable_weight_map(
        payload["next_region"], support, "next_region"
    )
    raw_sums = {
        "activity": activity_sum,
        "intent": intent_sum,
        "destination": destination_sum,
        "next_region": next_region_sum,
    }
    return {
        "activity": activity,
        "intent": intent,
        "destination": destination,
        "next_region": next_region,
        "confidence": confidence,
        "_normalization": {
            "version": NORMALIZATION_VERSION,
            "raw_sums": raw_sums,
            "applied": any(
                not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12)
                for total in raw_sums.values()
            ),
            "canonical_probability_error": max(
                abs(math.fsum(values.values()) - 1.0)
                for values in (activity, intent, destination, next_region)
            ),
        },
    }


def _fallback_payload(support: Sequence[str]) -> dict[str, Any]:
    uniform = {key: 1.0 / len(support) for key in support}
    return {
        "activity": {key: 1.0 if key == "other" else 0.0 for key in ACTIVITY_ONTOLOGY},
        "intent": {key: 1.0 if key == "new-zone" else 0.0 for key in INTENT_ONTOLOGY},
        "destination": {
            key: 1.0 if key == "other" else 0.0 for key in DESTINATION_ONTOLOGY
        },
        "next_region": uniform,
        "confidence": 0.0,
    }


def _make_llm(model: str, platform: str) -> Any:
    from models.llm_api import LLMWrapper

    return LLMWrapper(model_name=model, platform=platform)


def _usage_from_wrapper(llm: Any, started: float, prompt: str, raw: str) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    try:
        stats = llm.get_call_stats()
        row = stats[-1] if stats else {}
    except (AttributeError, IndexError, TypeError):
        row = getattr(llm, "last_call_stats", {}) or {}
    return {
        "latency_seconds": float(row.get("latency_seconds", elapsed)),
        "input_tokens": int(row.get("input_tokens", max(1, len(prompt) // 4))),
        "output_tokens": int(row.get("output_tokens", max(1, len(raw) // 4) if raw else 0)),
        "token_source": str(row.get("token_source", "estimated")),
    }


def _load_teacher_cache(cache_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts = _index_unique(_read_jsonl(cache_root / "attempts.jsonl"), cache_root / "attempts.jsonl")
    beliefs = _index_unique(_read_jsonl(cache_root / "beliefs.jsonl"), cache_root / "beliefs.jsonl")
    return attempts, beliefs


def _teacher_interface_policy(cache_tag: str) -> dict[str, float]:
    """Return the predeclared teacher-interface thresholds for one cache.

    Collection completeness and semantic interface validity are deliberately
    separate gates.  Smoke/repeat caches are small diagnostic samples and
    therefore admit no fallback.  The larger pilot base retains the protocol's
    predeclared 95% valid-output and 1% transport-error thresholds.
    """

    if cache_tag == "smoke" or cache_tag.startswith("pilot-repeat-"):
        return {
            "minimum_valid_output_rate": 1.0,
            "maximum_fallback_rate": 0.0,
            "maximum_transport_error_rate": 0.0,
        }
    return {
        "minimum_valid_output_rate": 0.95,
        "maximum_fallback_rate": 0.05,
        "maximum_transport_error_rate": 0.01,
    }


def teacher_cache(
    args: argparse.Namespace,
    *,
    llm_factory: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    if args.train_limit < 0 or args.validation_limit < 0:
        raise ValueError("teacher-cache limits must be non-negative")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(args.model_digest)):
        raise ValueError(
            "teacher cache requires a 64-hex SHA-256 --model-digest; obtain it from "
            "the Ollama /api/tags response"
        )
    output_root = Path(args.output_root)
    prepare_manifest_path = output_root / "manifests" / "prepare_manifest.json"
    if not prepare_manifest_path.is_file():
        raise FileNotFoundError("run prepare before teacher-cache")
    prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    if prepare_manifest.get("dataset") != args.dataset or prepare_manifest.get("unit") != args.unit:
        raise ValueError("prepared manifest dataset/unit mismatch")

    query_root = Path(getattr(args, "query_root", None) or output_root / "queries")
    query_manifest_path = query_root / "manifest.json"
    if query_root != output_root / "queries":
        if not query_manifest_path.is_file():
            raise FileNotFoundError(f"derived teacher query root has no manifest: {query_manifest_path}")
        query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
        required = {
            "dataset": args.dataset,
            "unit": args.unit,
            "dataset_hash": prepare_manifest["dataset_hash"],
            "ontology_hash": prepare_manifest["ontology_hash"],
            "target_artifacts_read": False,
        }
        for key, expected in required.items():
            if query_manifest.get(key) != expected:
                raise ValueError(f"derived teacher query manifest mismatch for {key}")
        if query_manifest.get("allowed_splits") != ["train", "validation"]:
            raise ValueError("derived teacher query manifest must be train/validation-only")
        query_source_contract = {
            "kind": "quantitative-bound",
            "manifest_hash": _file_hash(query_manifest_path),
            "checkpoint_hash": query_manifest.get("checkpoint_hash"),
        }
    else:
        query_source_contract = {
            "kind": "prepared-global-support",
            "manifest_hash": _file_hash(prepare_manifest_path),
        }

    selected: dict[str, list[dict[str, Any]]] = {}
    requested_by_split = {
        "train": int(args.train_limit),
        "validation": int(args.validation_limit),
    }
    available_by_split: dict[str, int] = {}
    for split, limit in (("train", args.train_limit), ("validation", args.validation_limit)):
        # Deliberately never construct or read the test path in this command.
        query_path = query_root / f"{split}.jsonl"
        rows = _read_jsonl(query_path)
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"query split provenance mismatch: {query_path}")
        if query_root != output_root / "queries":
            expected_hash = query_manifest.get("split_hashes", {}).get(split, {}).get("bound_query_hash")
            if expected_hash != _file_hash(query_path):
                raise ValueError(f"derived teacher query hash mismatch for split={split}")
        available_by_split[split] = len(rows)
        selected[split] = _stable_sample(rows, limit, args.seed)

    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model).strip("-")
    digest_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model_digest).strip("-")
    teacher_contract = {
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "response_format": "json_schema.strict",
        "temperature": 0.0,
        "sdk_retries": 0,
        "schema_hash": SCHEMA_HASH,
        "activity": ACTIVITY_ONTOLOGY,
        "intent": INTENT_ONTOLOGY,
        "destination": DESTINATION_ONTOLOGY,
        "ontology_hash": prepare_manifest["ontology_hash"],
        "query_source_contract": query_source_contract,
    }
    prompt_contract_hash = _canonical_hash(teacher_contract)
    selection_payload = {
        "version": TEACHER_VERSION,
        "dataset": args.dataset,
        "unit": args.unit,
        "cache_tag": args.cache_tag,
        "seed": args.seed,
        "model": args.model,
        "platform": args.platform,
        "model_digest": args.model_digest,
        "prompt_contract_hash": prompt_contract_hash,
        "dataset_hash": prepare_manifest["dataset_hash"],
        "ontology_hash": prepare_manifest["ontology_hash"],
        "query_source_contract": query_source_contract,
        "requested_by_split": requested_by_split,
        "available_by_split": available_by_split,
        "selected_by_split": {split: len(rows) for split, rows in selected.items()},
        "query_ids": {split: [str(row["query_id"]) for row in rows] for split, rows in selected.items()},
        "query_hashes": {split: _canonical_hash(rows) for split, rows in selected.items()},
        "test_queries_read": False,
    }
    selection_contract_hash = _canonical_hash(selection_payload)
    selection_manifest = {
        **selection_payload,
        "selection_contract_hash": selection_contract_hash,
    }
    cache_root = (
        output_root
        / "teacher"
        / model_slug
        / digest_slug
        / prompt_contract_hash
        / args.cache_tag
        / f"seed-{args.seed}-{selection_contract_hash[:16]}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    collector_lock = (cache_root / ".collector.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(collector_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        collector_lock.close()
        raise RuntimeError(f"teacher cache already has an active collector: {cache_root}") from exc
    _write_json_immutable(cache_root / "selection_manifest.json", selection_manifest)

    selected_ids = {
        str(row["query_id"])
        for rows in selected.values()
        for row in rows
    }
    if len(selected_ids) != sum(len(rows) for rows in selected.values()):
        fcntl.flock(collector_lock.fileno(), fcntl.LOCK_UN)
        collector_lock.close()
        raise ValueError("teacher selection contains duplicate query IDs")
    attempts, beliefs = _load_teacher_cache(cache_root)
    unexpected_attempts = sorted(set(attempts) - selected_ids)
    unexpected_beliefs = sorted(set(beliefs) - selected_ids)
    if unexpected_attempts or unexpected_beliefs:
        fcntl.flock(collector_lock.fileno(), fcntl.LOCK_UN)
        collector_lock.close()
        raise RuntimeError(
            "teacher cache journals contain query IDs outside the immutable selection; "
            f"attempts={unexpected_attempts[:5]}, beliefs={unexpected_beliefs[:5]}"
        )
    dangling = sorted(set(attempts) - set(beliefs))
    if dangling:
        fcntl.flock(collector_lock.fileno(), fcntl.LOCK_UN)
        collector_lock.close()
        raise RuntimeError(
            "strict one-request journal has attempts without beliefs; no automatic retry is allowed: "
            + ", ".join(dangling[:5])
        )
    llm = None
    factory = llm_factory or _make_llm
    model_identity = f"{args.platform}:{args.model}@{args.model_digest}"
    new_requests = 0
    for split in ("train", "validation"):
        for query in selected[split]:
            query_id = str(query["query_id"])
            prompt = build_teacher_prompt(query)
            response_format = build_teacher_response_format(query["candidate_regions"])
            source_hash = _canonical_hash(query)
            prompt_hash = hash_text(prompt)
            expected = {
                "query_id": query_id,
                "split": split,
                "source_hash": source_hash,
                "prompt_hash": prompt_hash,
                "response_format_hash": _canonical_hash(response_format),
                "model_hash": hash_text(model_identity),
                "dataset_hash": prepare_manifest["dataset_hash"],
                "ontology_hash": prepare_manifest["ontology_hash"],
                "physical_request": 1,
            }
            if query_id in beliefs:
                existing = beliefs[query_id]
                existing_attempt = attempts.get(query_id)
                try:
                    if existing_attempt is None:
                        raise RuntimeError(
                            f"query={query_id} has a belief without a matching physical-request attempt"
                        )
                    for key in (
                        "query_id",
                        "split",
                        "source_hash",
                        "prompt_hash",
                        "response_format_hash",
                        "model_hash",
                        "dataset_hash",
                        "ontology_hash",
                        "physical_request",
                    ):
                        if existing_attempt.get(key) != expected[key]:
                            raise ValueError(
                                f"immutable teacher attempt conflict for query={query_id} field={key}"
                            )
                        if existing.get(key) != expected[key]:
                            raise ValueError(
                                f"immutable teacher belief conflict for query={query_id} field={key}"
                            )
                    StructuredBelief.from_dict(existing["belief"])
                except Exception:
                    fcntl.flock(collector_lock.fileno(), fcntl.LOCK_UN)
                    collector_lock.close()
                    raise
                continue
            if query_id in attempts:
                raise RuntimeError(f"query={query_id} was attempted previously; refusing a second request")

            if llm is None:
                llm = factory(args.model, args.platform)
            _append_jsonl(cache_root / "attempts.jsonl", expected)
            attempts[query_id] = expected
            raw = ""
            started = time.perf_counter()
            try:
                reset = getattr(llm, "reset_call_stats", None)
                if callable(reset):
                    reset()
                raw = str(
                    llm.get_response_once(prompt, response_format=response_format)
                )
                usage = _usage_from_wrapper(llm, started, prompt, raw)
                parsed = parse_teacher_response(raw, query["candidate_regions"])
                normalization = parsed.pop("_normalization")
                raw_valid, fallback_reason, transport_error = True, "", ""
            except Exception as exc:  # one request means invalid/error output must fall back, never retry
                usage = _usage_from_wrapper(llm, started, prompt, raw)
                parsed = _fallback_payload(query["candidate_regions"])
                normalization = {
                    "version": NORMALIZATION_VERSION,
                    "raw_sums": {},
                    "applied": False,
                    "canonical_probability_error": 0.0,
                }
                raw_valid = False
                transport_error = "" if isinstance(exc, (ValueError, json.JSONDecodeError)) else f"{type(exc).__name__}: {exc}"
                fallback_reason = f"{type(exc).__name__}: {exc}"
            belief = StructuredBelief.create(
                query=_canonical_json(query),
                prompt=prompt,
                model_identity=model_identity,
                activity=parsed["activity"],
                intent=parsed["intent"],
                destination=parsed["destination"],
                next_region=parsed["next_region"],
                confidence=parsed["confidence"],
            )
            record = {
                **expected,
                "schema_hash": SCHEMA_HASH,
                "prompt": prompt,
                "raw_response": raw,
                "raw_response_hash": hash_text(raw),
                "raw_valid": raw_valid,
                "usable_output": raw_valid,
                "fallback": not raw_valid,
                "fallback_reason": fallback_reason,
                "transport_error": transport_error,
                "normalization": normalization,
                **usage,
                "belief": belief.to_dict(),
            }
            _append_jsonl(cache_root / "beliefs.jsonl", record)
            beliefs[query_id] = record
            new_requests += 1
            del started, raw

    selected_records = [beliefs[query_id] for query_id in selected_ids]
    completed_queries = len(selected_records)
    valid_outputs = sum(
        bool(row.get("usable_output", row.get("raw_valid")))
        for row in selected_records
    )
    fallbacks = sum(bool(row.get("fallback")) for row in selected_records)
    transport_errors = sum(bool(row.get("transport_error")) for row in selected_records)
    prompt_target_leakage = sum(
        bool(FORBIDDEN_TARGET_KEYS & _nested_keys(query))
        for rows in selected.values()
        for query in rows
    )
    valid_output_rate = valid_outputs / completed_queries if completed_queries else 0.0
    fallback_rate = fallbacks / completed_queries if completed_queries else 0.0
    transport_error_rate = transport_errors / completed_queries if completed_queries else 0.0
    collection_ready = (
        bool(selected_ids)
        and completed_queries == len(selected_ids)
        and set(attempts) == selected_ids
        and set(beliefs) == selected_ids
    )
    interface_policy = _teacher_interface_policy(args.cache_tag)
    interface_ready = (
        collection_ready
        and valid_output_rate >= interface_policy["minimum_valid_output_rate"]
        and fallback_rate <= interface_policy["maximum_fallback_rate"]
        and transport_error_rate <= interface_policy["maximum_transport_error_rate"]
        and prompt_target_leakage == 0
    )
    manifest = {
        **selection_manifest,
        "selected_queries": len(selected_ids),
        "completed_queries": completed_queries,
        "selected_by_split": {split: len(rows) for split, rows in selected.items()},
        "completed_by_split": {
            split: sum(record["split"] == split for record in selected_records)
            for split in ("train", "validation")
        },
        "physical_requests": sum(int(row["physical_request"]) for row in selected_records),
        "raw_valid": valid_outputs,
        "fallbacks": fallbacks,
        "transport_errors": transport_errors,
        "valid_output_rate": valid_output_rate,
        "fallback_rate": fallback_rate,
        "transport_error_rate": transport_error_rate,
        "prompt_target_leakage": prompt_target_leakage,
        "interface_policy": interface_policy,
        "collection_gate": "ready" if collection_ready else "incomplete",
        "interface_gate": "ready" if interface_ready else "incomplete",
        "gate": "ready" if collection_ready and interface_ready else "incomplete",
    }
    _write_json_immutable(cache_root / "manifest.json", manifest)
    payload = {
        "action": "teacher-cache",
        "dataset": args.dataset,
        "unit": args.unit,
        "gate": manifest["gate"],
        "collection_gate": manifest["collection_gate"],
        "interface_gate": manifest["interface_gate"],
        "valid_output_rate": valid_output_rate,
        "fallback_rate": fallback_rate,
        "transport_error_rate": transport_error_rate,
        "selected": len(selected_ids),
        "completed": len(selected_records),
        "new_requests": new_requests,
        "cache": str(cache_root),
    }
    fcntl.flock(collector_lock.fileno(), fcntl.LOCK_UN)
    collector_lock.close()
    print(json.dumps(payload, ensure_ascii=False))
    if getattr(args, "require_gate", False) and manifest["gate"] != "ready":
        raise SystemExit(3)
    return payload


def _teacher_cache_status(cache_root: Path) -> dict[str, Any]:
    attempts, beliefs = _load_teacher_cache(cache_root)
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return {
        "path": str(cache_root),
        "attempts": len(attempts),
        "beliefs": len(beliefs),
        "dangling_attempts": len(set(attempts) - set(beliefs)),
        "raw_valid": sum(bool(row.get("raw_valid")) for row in beliefs.values()),
        "fallbacks": sum(bool(row.get("fallback")) for row in beliefs.values()),
        "valid_output_rate": manifest.get("valid_output_rate"),
        "fallback_rate": manifest.get("fallback_rate"),
        "transport_error_rate": manifest.get("transport_error_rate"),
        "collection_gate": manifest.get("collection_gate", "incomplete"),
        "interface_gate": manifest.get("interface_gate", "incomplete"),
        "gate": manifest.get("gate", "incomplete"),
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    prepared = [
        output_root / "manifests" / "prepare_manifest.json",
        output_root / "representation" / "regions.json",
        output_root / "representation" / "poi_region_map.json",
        output_root / "representation" / "ontology.json",
    ] + [
        output_root / artifact / f"{split}.jsonl"
        for artifact in ("queries", "targets")
        for split in ("train", "validation", "test")
    ]
    missing = [str(path) for path in prepared if not path.is_file()]
    caches = [
        _teacher_cache_status(path.parent)
        for path in sorted((output_root / "teacher").rglob("manifest.json"))
    ] if (output_root / "teacher").is_dir() else []
    if missing:
        stage, gate = "incomplete", "incomplete"
    elif not caches:
        stage, gate = "prepared", "prepared"
    elif all(item["gate"] == "ready" for item in caches):
        stage, gate = "teacher-cached", "ready"
    else:
        stage, gate = "teacher-incomplete", "incomplete"
    payload = {
        "action": "status",
        "dataset": args.dataset,
        "unit": args.unit,
        "prepared": not missing,
        "missing": missing,
        "teacher_caches": caches,
        "stage": stage,
        "gate": gate,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return payload


def _smoke_cache_roots(
    results_root: Path,
    *,
    model: str = "",
    model_digest: str = "",
) -> dict[tuple[str, str], list[Path]]:
    """Index immutable smoke caches for the three predeclared pilot units."""

    indexed: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for selection_path in sorted(results_root.rglob("selection_manifest.json")):
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        key = (str(selection.get("dataset", "")), str(selection.get("unit", "")))
        if (
            key not in PILOT_UNITS
            or selection.get("cache_tag") != "smoke"
            or (model and selection.get("model") != model)
            or (model_digest and selection.get("model_digest") != model_digest)
        ):
            continue
        # Count every immutable selection, including an interrupted cache that
        # never reached its derived manifest.  Otherwise one complete cache
        # plus one abandoned duplicate could be mistaken for an unambiguous
        # smoke run.
        indexed[key].append(selection_path.parent)
    return indexed


def _audit_teacher_journals(
    cache_root: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a teacher-interface decision from immutable raw journals.

    Derived manifest counters are intentionally ignored.  The selection is
    the sole ID/split contract, while attempts and beliefs must each contain
    that exact set once.  Content and response-format provenance is checked on
    both journal rows before any response is counted as usable.
    """

    query_ids = selection.get("query_ids")
    expected_by_id: dict[str, str] = {}
    selection_ids_valid = (
        isinstance(query_ids, Mapping)
        and set(query_ids) == {"train", "validation"}
    )
    if selection_ids_valid:
        for split in ("train", "validation"):
            rows = query_ids.get(split)
            if not isinstance(rows, list):
                selection_ids_valid = False
                continue
            for raw_query_id in rows:
                query_id = str(raw_query_id)
                if not query_id or query_id in expected_by_id:
                    selection_ids_valid = False
                else:
                    expected_by_id[query_id] = split
    selected_counts = selection.get("selected_by_split")
    selection_ids_valid = selection_ids_valid and isinstance(selected_counts, Mapping)
    if selection_ids_valid:
        selection_ids_valid = all(
            selected_counts.get(split) == sum(
                expected_split == split for expected_split in expected_by_id.values()
            )
            for split in ("train", "validation")
        )

    journal_error = ""
    try:
        attempts, beliefs = _load_teacher_cache(cache_root)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        attempts, beliefs = {}, {}
        journal_error = f"{type(exc).__name__}: {exc}"

    expected_ids = set(expected_by_id)
    attempt_ids = set(attempts)
    belief_ids = set(beliefs)
    missing_attempt_ids = sorted(expected_ids - attempt_ids)
    unexpected_attempt_ids = sorted(attempt_ids - expected_ids)
    missing_belief_ids = sorted(expected_ids - belief_ids)
    unexpected_belief_ids = sorted(belief_ids - expected_ids)
    dangling_attempt_ids = sorted(attempt_ids - belief_ids)
    orphan_belief_ids = sorted(belief_ids - attempt_ids)

    schema_valid = True
    ontology_valid = True
    content_provenance_valid = True
    response_format_provenance_valid = True
    normalization_valid = True
    record_flags_valid = True
    prompt_target_leakage = 0
    usable_outputs = 0
    fallbacks = 0
    transport_errors = 0
    v2_cache = selection.get("version") == TEACHER_VERSION
    hash_fields = (
        "source_hash",
        "prompt_hash",
        "response_format_hash",
        "model_hash",
        "dataset_hash",
        "ontology_hash",
    )
    for query_id in sorted(expected_ids & attempt_ids & belief_ids):
        attempt = attempts[query_id]
        record = beliefs[query_id]
        expected_split = expected_by_id[query_id]
        if (
            attempt.get("query_id") != query_id
            or record.get("query_id") != query_id
            or attempt.get("split") != expected_split
            or record.get("split") != expected_split
            or attempt.get("physical_request") != 1
            or record.get("physical_request") != 1
        ):
            content_provenance_valid = False
        for field in hash_fields:
            attempt_value = attempt.get(field)
            record_value = record.get(field)
            if (
                not isinstance(attempt_value, str)
                or re.fullmatch(r"[0-9a-f]{64}", attempt_value) is None
                or record_value != attempt_value
            ):
                content_provenance_valid = False
                if field == "response_format_hash":
                    response_format_provenance_valid = False
        if (
            record.get("dataset_hash") != selection.get("dataset_hash")
            or record.get("ontology_hash") != selection.get("ontology_hash")
        ):
            content_provenance_valid = False
        prompt = record.get("prompt")
        raw_response = record.get("raw_response")
        if (
            not isinstance(prompt, str)
            or hash_text(prompt) != record.get("prompt_hash")
            or not isinstance(raw_response, str)
            or hash_text(raw_response) != record.get("raw_response_hash")
        ):
            content_provenance_valid = False
        if isinstance(prompt, str) and any(
            f'"{key}"' in prompt for key in FORBIDDEN_TARGET_KEYS
        ):
            prompt_target_leakage += 1
        try:
            belief = StructuredBelief.from_dict(record["belief"])
            if (
                belief.query_hash != record.get("source_hash")
                or belief.prompt_hash != record.get("prompt_hash")
                or belief.model_hash != record.get("model_hash")
            ):
                content_provenance_valid = False
            ontology_valid = ontology_valid and (
                set(belief.activity) == set(ACTIVITY_ONTOLOGY)
                and set(belief.intent) == set(INTENT_ONTOLOGY)
                and set(belief.destination) == set(DESTINATION_ONTOLOGY)
            )
        except (KeyError, TypeError, ValueError):
            schema_valid = False

        if v2_cache:
            normalization = record.get("normalization")
            error = normalization.get("canonical_probability_error") if isinstance(normalization, Mapping) else None
            normalization_valid = normalization_valid and (
                isinstance(normalization, Mapping)
                and normalization.get("version") == NORMALIZATION_VERSION
                and isinstance(error, (int, float))
                and not isinstance(error, bool)
                and math.isfinite(float(error))
                and 0.0 <= float(error) <= 1e-6
            )

        raw_valid = record.get("raw_valid")
        usable = record.get("usable_output", raw_valid)
        fallback = record.get("fallback")
        transport_error = record.get("transport_error")
        if (
            not isinstance(raw_valid, bool)
            or not isinstance(usable, bool)
            or not isinstance(fallback, bool)
            or not isinstance(transport_error, str)
            or ("usable_output" in record and usable != raw_valid)
            or fallback == usable
            or (usable and bool(transport_error))
        ):
            record_flags_valid = False
        usable_outputs += int(usable is True)
        fallbacks += int(fallback is True)
        transport_errors += int(bool(transport_error))

    completed = len(beliefs)
    checks = {
        "manifest_present": (cache_root / "manifest.json").is_file(),
        "teacher_contract_version": v2_cache,
        "selection_ids_valid": bool(expected_ids) and selection_ids_valid,
        "attempt_ids_exact": attempt_ids == expected_ids,
        "belief_ids_exact": belief_ids == expected_ids,
        "no_dangling_ids": attempt_ids == belief_ids,
        "journals_parse": not journal_error,
        "belief_schema_valid": schema_valid and completed > 0,
        "fixed_ontology_valid": ontology_valid and completed > 0,
        "content_provenance_valid": content_provenance_valid and completed > 0,
        "response_format_provenance_valid": response_format_provenance_valid and completed > 0,
        "normalization_valid": normalization_valid and completed > 0,
        "record_flags_valid": record_flags_valid and completed > 0,
        "all_outputs_valid": completed > 0 and usable_outputs == completed,
        "zero_fallbacks": fallbacks == 0,
        "zero_transport_errors": transport_errors == 0,
        "no_prompt_target_leakage": (
            prompt_target_leakage == 0 and selection.get("test_queries_read") is False
        ),
    }
    return {
        "selected": len(expected_ids),
        "completed": completed,
        "raw_valid": usable_outputs,
        "fallbacks": fallbacks,
        "transport_errors": transport_errors,
        "prompt_target_leakage": prompt_target_leakage,
        "journal_error": journal_error or None,
        "missing_attempt_ids": missing_attempt_ids,
        "unexpected_attempt_ids": unexpected_attempt_ids,
        "missing_belief_ids": missing_belief_ids,
        "unexpected_belief_ids": unexpected_belief_ids,
        "dangling_attempt_ids": dangling_attempt_ids,
        "orphan_belief_ids": orphan_belief_ids,
        "checks": checks,
    }


def build_smoke_report(
    results_root: Path,
    *,
    model: str = "",
    model_digest: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Build the fail-fast teacher-interface gate before pilot collection.

    A unit is ready only when one unambiguous smoke cache has complete
    train/validation journals and every response is valid without fallback,
    transport error, or target leakage.  The report reads no test artifacts.
    """

    indexed = _smoke_cache_roots(
        results_root,
        model=model,
        model_digest=model_digest,
    )
    units: list[dict[str, Any]] = []
    for dataset, unit in sorted(PILOT_UNITS):
        candidates = indexed.get((dataset, unit), [])
        if len(candidates) != 1:
            units.append(
                {
                    "dataset": dataset,
                    "unit": unit,
                    "cache": None,
                    "candidate_caches": [str(path) for path in candidates],
                    "checks": {"unique_cache": False},
                    "gate": "incomplete",
                }
            )
            continue
        cache_root = candidates[0]
        selection = json.loads(
            (cache_root / "selection_manifest.json").read_text(encoding="utf-8")
        )
        requested = selection.get("requested_by_split", {})
        available = selection.get("available_by_split", {})
        selected = selection.get("selected_by_split", {})
        budget_valid = requested == SMOKE_REQUESTED and all(
            isinstance(available.get(split), int)
            and int(available[split]) >= 0
            and int(selected.get(split, -1))
            == min(SMOKE_REQUESTED[split], int(available[split]))
            for split in ("train", "validation")
        )
        audit = _audit_teacher_journals(cache_root, selection)
        completed = int(audit["completed"])
        valid = int(audit["raw_valid"])
        checks = {
            "unique_cache": True,
            "smoke_budget_contract": budget_valid,
            **audit["checks"],
        }
        units.append(
            {
                "dataset": dataset,
                "unit": unit,
                "cache": str(cache_root),
                "selected": int(audit["selected"]),
                "completed": completed,
                "valid_output_rate": valid / completed if completed else None,
                "fallback_rate": (
                    int(audit["fallbacks"]) / completed if completed else None
                ),
                "transport_error_rate": (
                    int(audit["transport_errors"]) / completed if completed else None
                ),
                "journal_diagnostics": {
                    key: audit[key]
                    for key in (
                        "journal_error",
                        "missing_attempt_ids",
                        "unexpected_attempt_ids",
                        "missing_belief_ids",
                        "unexpected_belief_ids",
                        "dangling_attempt_ids",
                        "orphan_belief_ids",
                    )
                },
                "checks": checks,
                "gate": "ready" if all(checks.values()) else "incomplete",
            }
        )
    all_units_present = all(len(indexed.get(key, [])) == 1 for key in PILOT_UNITS)
    report = {
        "version": "beliefmove.smoke-report.v2",
        "results_root": str(results_root),
        "model": model or None,
        "model_digest": model_digest or None,
        "required_units": [f"{dataset}:{unit}" for dataset, unit in sorted(PILOT_UNITS)],
        "all_units_present": all_units_present,
        "units": units,
        "test_artifacts_read": False,
        "gate": (
            "ready"
            if all_units_present and len(units) == len(PILOT_UNITS)
            and all(item["gate"] == "ready" for item in units)
            else "incomplete"
        ),
    }
    report["report_hash"] = _canonical_hash(report)
    if write:
        output = results_root / "aggregated" / "smoke_report.json"
        _atomic_text(output, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return report


def smoke_report(args: argparse.Namespace) -> dict[str, Any]:
    report = build_smoke_report(
        Path(args.results_root),
        model=str(getattr(args, "model", "") or ""),
        model_digest=str(getattr(args, "model_digest", "") or ""),
        write=True,
    )
    print(json.dumps(report, ensure_ascii=False))
    if getattr(args, "require_ready", False) and report["gate"] != "ready":
        raise SystemExit(3)
    return report


def _pilot_cache_roots(results_root: Path) -> list[Path]:
    roots: list[Path] = []
    for selection_path in sorted(results_root.rglob("selection_manifest.json")):
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if selection.get("cache_tag") == "pilot":
            # Include interrupted bases without a derived manifest so they
            # still count toward the uniqueness gate.
            roots.append(selection_path.parent)
    return roots


def _related_pilot_cache(cache_root: Path, tag: str) -> Path | None:
    """Resolve one content-addressed cache sharing the base prompt contract."""

    legacy = cache_root.parent / tag
    if (legacy / "manifest.json").is_file() and (legacy / "selection_manifest.json").is_file():
        return legacy
    prompt_root = cache_root.parent.parent
    base_selection = json.loads((cache_root / "selection_manifest.json").read_text(encoding="utf-8"))
    candidates: list[Path] = []
    tag_root = prompt_root / tag
    if tag_root.is_dir():
        for manifest_path in sorted(tag_root.rglob("manifest.json")):
            selection_path = manifest_path.parent / "selection_manifest.json"
            if not selection_path.is_file():
                continue
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if (
                selection.get("cache_tag") == tag
                and selection.get("model") == base_selection.get("model")
                and selection.get("model_digest") == base_selection.get("model_digest")
                and selection.get("prompt_contract_hash") == base_selection.get("prompt_contract_hash")
                and selection.get("seed") == base_selection.get("seed")
            ):
                candidates.append(manifest_path.parent)
    return candidates[0] if len(candidates) == 1 else None


def _unit_root_for_cache(cache_root: Path) -> Path:
    for parent in cache_root.parents:
        if (parent / "manifests" / "prepare_manifest.json").is_file():
            return parent
    # Synthetic unit tests construct only teacher artifacts.  Their dataset
    # root is still the first parent above ``teacher``.
    for parent in cache_root.parents:
        if parent.name == "teacher":
            return parent.parent
    raise ValueError(f"cannot resolve dataset unit root for cache: {cache_root}")


def _pilot_evaluation(
    unit_root: Path,
    cache_root: Path,
    belief_hash: str,
) -> dict[str, Any] | None:
    """Load a pilot decision only after reproducing it from immutable raw inputs.

    A hand-written or stale summary must never open the protocol lock.  The
    inexpensive evaluator is therefore rerun from the prepared train/
    validation artifacts and the four teacher caches on every report/lock.
    """

    path = unit_root / "manifests" / "pilot_evaluation.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    embedded_hash = payload.get("evaluation_hash")
    unhashed = {key: value for key, value in payload.items() if key != "evaluation_hash"}
    if (
        payload.get("fit_splits") != ["train", "validation"]
        or payload.get("test_artifacts_read") is not False
        or payload.get("teacher_cache_hash") != belief_hash
        or embedded_hash != _canonical_hash(unhashed)
    ):
        return None
    repeats = [
        _related_pilot_cache(cache_root, f"pilot-repeat-{index}")
        for index in range(1, PILOT_REPEAT_COUNT + 1)
    ]
    if any(path is None for path in repeats):
        return None
    try:
        recomputed = evaluate_pilot(
            unit_root,
            cache_root,
            [path for path in repeats if path is not None],
            expected_repeat_count=PILOT_REPEAT_COUNT,
            write=False,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, PilotEvaluationError):
        return None
    if _canonical_hash(recomputed) != _canonical_hash(payload):
        return None
    return payload


def build_pilot_report(results_root: Path, *, write: bool = True) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for cache_root in _pilot_cache_roots(results_root):
        selection_path = cache_root / "selection_manifest.json"
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        key = (str(selection.get("dataset", "")), str(selection.get("unit", "")))
        if key in PILOT_UNITS:
            indexed[key].append(cache_root)

    for dataset, unit in sorted(PILOT_UNITS):
        candidates = indexed.get((dataset, unit), [])
        if len(candidates) != 1:
            units.append(
                {
                    "dataset": dataset,
                    "unit": unit,
                    "cache": None,
                    "candidate_caches": [str(path) for path in candidates],
                    "pilot_evaluation_present": False,
                    "checks": {"unique_base_cache": False},
                    "gate": "incomplete",
                }
            )
            continue
        cache_root = candidates[0]
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.is_file():
            units.append(
                {
                    "dataset": dataset,
                    "unit": unit,
                    "cache": str(cache_root),
                    "candidate_caches": [str(cache_root)],
                    "pilot_evaluation_present": False,
                    "checks": {
                        "unique_base_cache": True,
                        "manifest_present": False,
                    },
                    "gate": "incomplete",
                }
            )
            continue
        unit_root = _unit_root_for_cache(cache_root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection_path = cache_root / "selection_manifest.json"
        attempts_path = cache_root / "attempts.jsonl"
        beliefs_path = cache_root / "beliefs.jsonl"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
        attempts, beliefs = _load_teacher_cache(cache_root)
        expected_ids = {
            str(query_id)
            for split in ("train", "validation")
            for query_id in selection.get("query_ids", {}).get(split, [])
        }
        schema_valid = True
        ontology_valid = True
        provenance_valid = True
        for record in beliefs.values():
            try:
                belief = StructuredBelief.from_dict(record["belief"])
                ontology_valid = ontology_valid and (
                    set(belief.activity) == set(ACTIVITY_ONTOLOGY)
                    and set(belief.intent) == set(INTENT_ONTOLOGY)
                    and set(belief.destination) == set(DESTINATION_ONTOLOGY)
                )
                provenance_valid = provenance_valid and (
                    record.get("dataset_hash") == selection.get("dataset_hash")
                    and record.get("ontology_hash") == selection.get("ontology_hash")
                )
            except (KeyError, TypeError, ValueError):
                schema_valid = False
                break
        completed = len(beliefs)
        valid = sum(bool(record.get("raw_valid")) for record in beliefs.values())
        transport = sum(bool(record.get("transport_error")) for record in beliefs.values())
        belief_hash = _file_hash(beliefs_path) if beliefs_path.is_file() else ""
        evaluation = _pilot_evaluation(unit_root, cache_root, belief_hash) if belief_hash else None
        requested = selection.get("requested_by_split", {})
        available = selection.get("available_by_split", {})
        selected_counts = selection.get("selected_by_split", {})
        pilot_budget_valid = requested == PILOT_BASE_REQUESTED and all(
            isinstance(available.get(split), int)
            and int(available[split]) >= 0
            and int(selected_counts.get(split, -1))
            == min(PILOT_BASE_REQUESTED[split], int(available[split]))
            for split in ("train", "validation")
        )
        raw_cache_complete = (
            bool(expected_ids)
            and set(attempts) == expected_ids
            and set(beliefs) == expected_ids
            and {record.get("split") for record in beliefs.values()} <= {"train", "validation"}
            and all(selection.get("query_ids", {}).get(split) for split in ("train", "validation"))
        )
        checks = {
            "unique_base_cache": True,
            "cache_complete": raw_cache_complete,
            "belief_schema_valid": schema_valid and completed > 0,
            "fixed_ontology_valid": ontology_valid and completed > 0,
            "content_provenance_valid": provenance_valid and completed > 0,
            "pilot_budget_contract": pilot_budget_valid,
            "valid_output_rate": completed > 0 and valid / completed >= 0.95,
            "transport_error_rate": completed > 0 and transport / completed <= 0.01,
            "no_prompt_target_leakage": (
                int(manifest.get("prompt_target_leakage", -1)) == 0
                and selection.get("test_queries_read") is False
            ),
            "teacher_activity_gain": bool(
                evaluation and evaluation.get("gates", {}).get("teacher_activity_gain") is True
            ),
            "teacher_destination_gain": bool(
                evaluation and evaluation.get("gates", {}).get("teacher_destination_gain") is True
            ),
            "m2_heldout_nll_improves": bool(
                evaluation
                and evaluation.get("gates", {}).get("m2_heldout_nll_improves") is True
            ),
            "repeat_jsd": bool(
                evaluation and evaluation.get("gates", {}).get("repeat_jsd") is True
            ),
        }
        units.append(
            {
                "dataset": dataset,
                "unit": unit,
                "cache": str(cache_root),
                "selected": int(manifest.get("selected_queries", 0)),
                "completed": completed,
                "valid_output_rate": valid / completed if completed else None,
                "transport_error_rate": transport / completed if completed else None,
                "pilot_evaluation_present": evaluation is not None,
                "checks": checks,
                "gate": "ready" if all(checks.values()) else "incomplete",
            }
        )
    all_units_present = all(len(indexed.get(key, [])) == 1 for key in PILOT_UNITS)
    report = {
        "version": "beliefmove.pilot-report.v2",
        "results_root": str(results_root),
        "required_units": [f"{dataset}:{unit}" for dataset, unit in sorted(PILOT_UNITS)],
        "all_units_present": all_units_present,
        "units": units,
        "test_artifacts_read": False,
        "gate": (
            "ready"
            if all_units_present
            and len(units) == len(PILOT_UNITS)
            and all(item["gate"] == "ready" for item in units)
            else "incomplete"
        ),
    }
    report["report_hash"] = _canonical_hash(report)
    if write:
        output = results_root / "aggregated" / "pilot_report.json"
        # Aggregates are derived views and may legitimately move from
        # incomplete to ready while immutable raw caches remain unchanged.
        _atomic_text(output, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return report


def pilot_report(args: argparse.Namespace) -> dict[str, Any]:
    report = build_pilot_report(Path(args.results_root), write=True)
    print(json.dumps(report, ensure_ascii=False))
    return report


def pilot_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Derive the train/validation pilot gate from immutable raw caches."""

    unit_root = Path(args.output_root)
    candidates: list[Path] = []
    for manifest_path in sorted((unit_root / "teacher").rglob("manifest.json")):
        selection_path = manifest_path.parent / "selection_manifest.json"
        if not selection_path.is_file():
            continue
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            payload.get("cache_tag") == "pilot"
            and payload.get("model") == args.model
            and payload.get("model_digest") == args.model_digest
        ):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            "pilot-evaluate requires exactly one matching base cache for "
            f"model={args.model!r} digest={args.model_digest!r}; observed={len(candidates)}"
        )
    base = candidates[0]
    resolved = [
        _related_pilot_cache(base, f"pilot-repeat-{index}")
        for index in range(1, args.repeat_count + 1)
    ]
    missing = [f"pilot-repeat-{index}" for index, path in enumerate(resolved, start=1) if path is None]
    if missing:
        raise FileNotFoundError(f"pilot repeat caches are incomplete: {missing}")
    repeats = [path for path in resolved if path is not None]
    result = evaluate_pilot(
        unit_root,
        base,
        repeats,
        expected_repeat_count=args.repeat_count,
    )
    response = {
        "action": "pilot-evaluate",
        "dataset": args.dataset,
        "unit": args.unit,
        "gate": result["gate"],
        "output": str(unit_root / "manifests" / "pilot_evaluation.json"),
    }
    print(json.dumps(response, ensure_ascii=False))
    return response


def lock(args: argparse.Namespace) -> dict[str, Any]:
    results_root, protocol_path = Path(args.results_root), Path(args.protocol)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # Recompute only from pilot train/validation artifacts.  This function
    # deliberately has no code path to queries/targets/test.*.
    report = build_pilot_report(results_root, write=False)
    if report["gate"] != "ready" or report.get("test_artifacts_read") is not False:
        failed = [
            f"{item['dataset']}:{item['unit']}"
            for item in report["units"]
            if item["gate"] != "ready"
        ]
        raise RuntimeError(f"pilot gate is not ready; protocol remains unlocked; failed={failed}")
    payload = {
        "version": "beliefmove.protocol-lock.v2",
        "protocol_hash": _canonical_hash(protocol),
        "pilot_report_hash": report["report_hash"],
        "pilot_gate": "ready",
        "test_artifacts_read": False,
    }
    lock_path = results_root / "locks" / "protocol_lock.json"
    _write_json_immutable(lock_path, payload)
    response = {"action": "lock", "gate": "ready", "lock": str(lock_path), **payload}
    print(json.dumps(response, ensure_ascii=False))
    return response


def _not_implemented(args: argparse.Namespace) -> None:
    raise SystemExit(
        f"NOT_IMPLEMENTED: '{args.command}' is reserved by the faithful BeliefMove protocol; "
        "it must not be replaced by a legacy Phase-2 experiment"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Faithful BeliefMove experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def unit_parser(name: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset", required=True, choices=("tist2015", "www2019", "yjmob100k"))
        sub.add_argument("--unit", required=True)
        sub.add_argument("--data-base", required=True)
        sub.add_argument("--output-root", required=True)
        return sub

    audit_parser = unit_parser("audit")
    audit_parser.set_defaults(handler=audit)
    prepare_parser = unit_parser("prepare")
    prepare_parser.add_argument("--region-cell-meters", type=float, default=500.0)
    prepare_parser.set_defaults(handler=prepare)
    teacher_parser = unit_parser("teacher-cache")
    teacher_parser.add_argument("--model", required=True)
    teacher_parser.add_argument("--platform", default="Ollama")
    teacher_parser.add_argument("--model-digest", default="unresolved")
    teacher_parser.add_argument("--cache-tag", required=True)
    teacher_parser.add_argument("--train-limit", type=int, required=True)
    teacher_parser.add_argument("--validation-limit", type=int, required=True)
    teacher_parser.add_argument("--seed", type=int, default=42)
    teacher_parser.add_argument("--query-root")
    teacher_parser.add_argument("--require-gate", action="store_true")
    teacher_parser.set_defaults(handler=teacher_cache)
    pilot_parser = unit_parser("pilot-evaluate")
    pilot_parser.add_argument("--model", required=True)
    pilot_parser.add_argument("--model-digest", required=True)
    pilot_parser.add_argument("--repeat-count", type=int, default=3)
    pilot_parser.set_defaults(handler=pilot_evaluate)
    status_parser = unit_parser("status")
    status_parser.add_argument("--model", default="")
    status_parser.set_defaults(handler=status)

    report_parser = subparsers.add_parser("pilot-report")
    report_parser.add_argument("--results-root", required=True)
    report_parser.set_defaults(handler=pilot_report)
    smoke_report_parser = subparsers.add_parser("smoke-report")
    smoke_report_parser.add_argument("--results-root", required=True)
    smoke_report_parser.add_argument("--model", default="")
    smoke_report_parser.add_argument("--model-digest", default="")
    smoke_report_parser.add_argument("--require-ready", action="store_true")
    smoke_report_parser.set_defaults(handler=smoke_report)
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--results-root", required=True)
    lock_parser.add_argument("--protocol", required=True)
    lock_parser.set_defaults(handler=lock)
    for name in sorted(FUTURE_COMMANDS):
        future = subparsers.add_parser(name)
        future.add_argument("--results-root")
        future.add_argument("--dataset")
        future.add_argument("--unit")
        future.add_argument("--data-base")
        future.add_argument("--output-root")
        future.add_argument("--model")
        future.add_argument("--seeds", nargs="*")
        future.add_argument("--limit")
        future.add_argument("--protocol")
        future.add_argument("--iterations")
        future.set_defaults(handler=_not_implemented)
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
