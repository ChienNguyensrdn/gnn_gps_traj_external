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
TEACHER_VERSION = "beliefmove.teacher-cache.v1"
PROMPT_VERSION = "beliefmove.structured-teacher.v1"
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


def _candidate_id(raw: object, candidate_ids: Sequence[str]) -> str:
    value = str(raw).strip()
    if value in candidate_ids:
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
            poi_id = _candidate_id(row.get(poi_col, ""), candidate_ids)
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
    by_poi: dict[str, list[tuple[float, float]]] = defaultdict(list)
    poi_counts: Counter[str] = Counter()
    for row in train_rows:
        poi_id = str(row["poi_id"])
        if poi_id not in candidate_ids:
            continue
        poi_counts[poi_id] += 1
        if _valid_coordinate(row.get("latitude"), row.get("longitude")):
            by_poi[poi_id].append((float(row["latitude"]), float(row["longitude"])))

    if dataset == "yjmob100k":
        # YJMob locations are anonymous grid cells, not GPS coordinates.
        vocabulary = AnonymousCellRegionVocabulary.fit(
            (row["poi_id"] for row in train_rows if str(row["poi_id"]) in candidate_ids),
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
    source_rows = {
        split: _read_split(_resolve_csv(data_base, split), candidate_ids)
        for split in ("train", "validation", "test")
    }
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
        "activity": {label: "probability" for label in ACTIVITY_ONTOLOGY},
        "intent": {label: "probability" for label in INTENT_ONTOLOGY},
        "destination": {label: "probability" for label in DESTINATION_ONTOLOGY},
        "next_region": {label: "probability" for label in support},
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
        "Every distribution must contain exactly the shown keys, use finite non-negative numbers, and sum to 1.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Fixed output schema: {_canonical_json(schema)}\n"
        f"Observed causal input: {_canonical_json(observed)}\n"
    )


def _strict_probability_map(value: Any, keys: Sequence[str], name: str) -> dict[str, float]:
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
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to one")
    return result


def parse_teacher_response(raw: str, support: Sequence[str]) -> dict[str, Any]:
    payload = json.loads(raw)
    expected = {"activity", "intent", "destination", "next_region", "confidence"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("teacher JSON fields do not match the fixed schema")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0,1]")
    return {
        "activity": _strict_probability_map(payload["activity"], ACTIVITY_ONTOLOGY, "activity"),
        "intent": _strict_probability_map(payload["intent"], INTENT_ONTOLOGY, "intent"),
        "destination": _strict_probability_map(payload["destination"], DESTINATION_ONTOLOGY, "destination"),
        "next_region": _strict_probability_map(payload["next_region"], support, "next_region"),
        "confidence": confidence,
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


def teacher_cache(
    args: argparse.Namespace,
    *,
    llm_factory: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    if args.train_limit < 0 or args.validation_limit < 0:
        raise ValueError("teacher-cache limits must be non-negative")
    if args.cache_tag != "smoke" and not re.fullmatch(r"[0-9a-fA-F]{64}", str(args.model_digest)):
        raise ValueError(
            "non-smoke teacher cache requires a 64-hex SHA-256 --model-digest; obtain it from "
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

    attempts, beliefs = _load_teacher_cache(cache_root)
    dangling = sorted(set(attempts) - set(beliefs))
    if dangling:
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
            source_hash = _canonical_hash(query)
            prompt_hash = hash_text(prompt)
            expected = {
                "query_id": query_id,
                "split": split,
                "source_hash": source_hash,
                "prompt_hash": prompt_hash,
                "model_hash": hash_text(model_identity),
                "dataset_hash": prepare_manifest["dataset_hash"],
                "ontology_hash": prepare_manifest["ontology_hash"],
                "physical_request": 1,
            }
            if query_id in beliefs:
                existing = beliefs[query_id]
                for key in (
                    "split",
                    "source_hash",
                    "prompt_hash",
                    "model_hash",
                    "dataset_hash",
                    "ontology_hash",
                ):
                    if existing.get(key) != expected[key]:
                        raise ValueError(f"immutable teacher belief conflict for query={query_id} field={key}")
                StructuredBelief.from_dict(existing["belief"])
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
                raw = str(llm.get_response_once(prompt))
                usage = _usage_from_wrapper(llm, started, prompt, raw)
                parsed = parse_teacher_response(raw, query["candidate_regions"])
                raw_valid, fallback_reason, transport_error = True, "", ""
            except Exception as exc:  # one request means invalid/error output must fall back, never retry
                usage = _usage_from_wrapper(llm, started, prompt, raw)
                parsed = _fallback_payload(query["candidate_regions"])
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
                "fallback": not raw_valid,
                "fallback_reason": fallback_reason,
                "transport_error": transport_error,
                **usage,
                "belief": belief.to_dict(),
            }
            _append_jsonl(cache_root / "beliefs.jsonl", record)
            beliefs[query_id] = record
            new_requests += 1
            del started, raw

    selected_ids = {str(row["query_id"]) for rows in selected.values() for row in rows}
    selected_records = [beliefs[query_id] for query_id in selected_ids]
    manifest = {
        **selection_manifest,
        "selected_queries": len(selected_ids),
        "completed_queries": len(selected_records),
        "selected_by_split": {split: len(rows) for split, rows in selected.items()},
        "completed_by_split": {
            split: sum(record["split"] == split for record in selected_records)
            for split in ("train", "validation")
        },
        "physical_requests": sum(int(row["physical_request"]) for row in selected_records),
        "raw_valid": sum(bool(row["raw_valid"]) for row in selected_records),
        "fallbacks": sum(bool(row["fallback"]) for row in selected_records),
        "transport_errors": sum(bool(row["transport_error"]) for row in selected_records),
        "prompt_target_leakage": sum(
            bool(FORBIDDEN_TARGET_KEYS & _nested_keys(query))
            for rows in selected.values()
            for query in rows
        ),
        "gate": "ready" if selected_ids and len(selected_records) == len(selected_ids) else "incomplete",
    }
    _write_json_immutable(cache_root / "manifest.json", manifest)
    payload = {
        "action": "teacher-cache",
        "dataset": args.dataset,
        "unit": args.unit,
        "gate": manifest["gate"],
        "selected": len(selected_ids),
        "completed": len(selected_records),
        "new_requests": new_requests,
        "cache": str(cache_root),
    }
    fcntl.flock(collector_lock.fileno(), fcntl.LOCK_UN)
    collector_lock.close()
    print(json.dumps(payload, ensure_ascii=False))
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
        "gate": manifest.get("gate", "incomplete"),
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    prepared = [
        output_root / "manifests" / "prepare_manifest.json",
        output_root / "representation" / "regions.json",
        output_root / "representation" / "poi_region_map.json",
        output_root / "representation" / "ontology.json",
    ] + [output_root / "queries" / f"{split}.jsonl" for split in ("train", "validation", "test")]
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


def _pilot_cache_roots(results_root: Path) -> list[Path]:
    roots: list[Path] = []
    for path in sorted(results_root.rglob("manifest.json")):
        selection_path = path.parent / "selection_manifest.json"
        if not selection_path.is_file():
            continue
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if selection.get("cache_tag") == "pilot":
            roots.append(path.parent)
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
    seen: set[tuple[str, str]] = set()
    for cache_root in _pilot_cache_roots(results_root):
        unit_root = _unit_root_for_cache(cache_root)
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        dataset, unit = str(manifest["dataset"]), str(manifest["unit"])
        seen.add((dataset, unit))
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
    all_units_present = PILOT_UNITS <= seen
    report = {
        "version": "beliefmove.pilot-report.v1",
        "results_root": str(results_root),
        "required_units": [f"{dataset}:{unit}" for dataset, unit in sorted(PILOT_UNITS)],
        "all_units_present": all_units_present,
        "units": units,
        "test_artifacts_read": False,
        "gate": "ready" if all_units_present and units and all(item["gate"] == "ready" for item in units) else "incomplete",
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
        "version": "beliefmove.protocol-lock.v1",
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
