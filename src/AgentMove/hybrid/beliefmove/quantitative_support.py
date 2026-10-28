from __future__ import annotations

"""Frozen quantitative-teacher support on the faithful region vocabulary.

This module is deliberately separate from :mod:`hybrid.beliefmove.experiment`.
It consumes the immutable artifacts produced by ``prepare`` and exports a
per-query probability distribution over the *train-fitted* region support.
The held-out target artifact is never opened.  The only next-event field used
by the frozen Neural-CGM is the protocol-declared prediction timestamp already
stored in the prepared query; the next POI/category/region is neither emitted
nor passed to the model.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .regions import UNKNOWN_REGION


EXPORT_VERSION = "beliefmove.quantitative-region-support.v1"
BIND_VERSION = "beliefmove.teacher-query-binding.v1"
OTHER_REGION = "__OTHER__"


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable quantitative artifact conflict: {path}")
        return
    _atomic_bytes(path, payload)


def _write_npz_immutable(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write NPZ once; compare arrays rather than ZIP metadata on resume."""

    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise ValueError(f"immutable quantitative artifact fields changed: {path}")
            for key, expected in arrays.items():
                observed = existing[key]
                if observed.dtype != expected.dtype or observed.shape != expected.shape or not np.array_equal(observed, expected):
                    raise ValueError(f"immutable quantitative artifact array changed: {path}:{key}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "query_id" not in value:
            raise ValueError(f"invalid query row at {path}:{line_number}")
        query_id = str(value["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate query_id={query_id} in {path}")
        seen.add(query_id)
        rows.append(value)
    return rows


def _column(fieldnames: Sequence[str], choices: Sequence[str]) -> str:
    for value in choices:
        if value in fieldnames:
            return value
    raise ValueError(f"CSV is missing one of required columns: {list(choices)}")


def _candidate_id(raw: object, candidate_ids: Sequence[str]) -> str:
    value = str(raw).strip()
    if value in candidate_ids:
        return value
    try:
        index = int(float(value))
    except (TypeError, ValueError):
        return value
    return candidate_ids[index] if 0 <= index < len(candidate_ids) else value


def _source_trajectories(path: Path, candidate_ids: Sequence[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Read source trajectories without materialising a target label artifact.

    Every event can be observed history for a later all-prefix query, so the
    source CSV necessarily contains POI IDs.  ``build_quantitative_inputs``
    enforces the causal boundary per query: only rows strictly before ``stop``
    provide POI IDs.  The prediction timestamp comes from the hash-bound
    prepared query contract.
    """

    # Share the exact timestamp normalisation contract used by faithful
    # ``prepare`` so query/source provenance is byte-stable across timezones.
    from .experiment import _parse_timestamp

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        user_col = _column(fields, ("user_id", "user"))
        trajectory_col = _column(fields, ("trajectory_id", "traj_id"))
        poi_col = _column(fields, ("POI_id", "venue_id", "poi_id"))
        time_col = _column(fields, ("UTC_time", "utc_time", "timezone", "datetime"))
        for position, row in enumerate(reader):
            key = (str(row.get(user_col, "")), str(row.get(trajectory_col, "")))
            observed_time, chronological_key = _parse_timestamp(
                row.get(time_col, ""), path=path, row_number=position + 2, column=time_col
            )
            grouped[key].append(
                {
                    "poi_id": _candidate_id(row.get(poi_col, ""), candidate_ids),
                    "observed_time": observed_time,
                    "chronological_key": chronological_key,
                    "source_position": position,
                }
            )
    for key in grouped:
        grouped[key].sort(key=lambda row: (row["chronological_key"], row["source_position"]))
    return dict(grouped)


def build_quantitative_inputs(
    query_rows: Sequence[Mapping[str, Any]],
    source_csv: Path,
    candidate_ids: Sequence[str],
    poi_region: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Join faithful query IDs to causal Neural-CGM inputs.

    Returned dictionaries intentionally have no target POI, target category or
    target region field.  ``query_time`` is the declared prediction timestamp
    required by the frozen model's time embedding.
    """

    trajectories = _source_trajectories(source_csv, candidate_ids)
    inputs: list[dict[str, Any]] = []
    for query in query_rows:
        query_id = str(query["query_id"])
        key = (str(query["user_id"]), str(query["trajectory_id"]))
        if key not in trajectories:
            raise ValueError(f"query trajectory missing from source CSV: {query_id} key={key}")
        stop = int(query.get("prefix_length", -1))
        trajectory = trajectories[key]
        if stop < 1 or stop >= len(trajectory):
            raise ValueError(f"invalid all-prefix stop for query_id={query_id}: {stop}")
        observed = trajectory[:stop]
        query_prefix = list(query.get("observed_prefix", []))
        if len(query_prefix) != stop:
            raise ValueError(f"prepared prefix length mismatch for query_id={query_id}")
        for position, (prepared, source) in enumerate(zip(query_prefix, observed)):
            if str(prepared.get("observed_time", "")) != source["observed_time"]:
                raise ValueError(
                    f"prepared/source timestamp mismatch for query_id={query_id} position={position}"
                )
            expected_region = poi_region.get(source["poi_id"], UNKNOWN_REGION)
            if str(prepared.get("region")) != expected_region:
                raise ValueError(
                    f"prepared/source region mismatch for query_id={query_id} position={position}"
                )
        query_time = str(query.get("query_time", ""))
        if not query_time:
            raise ValueError(f"prepared query is missing query_time for query_id={query_id}")
        # This equality is a provenance check only.  No target identity,
        # category or region is materialised or passed to the model.
        if query_time != trajectory[stop]["observed_time"]:
            raise ValueError(f"prepared/source query_time mismatch for query_id={query_id}")
        inputs.append(
            {
                "query_id": query_id,
                "user_id": key[0],
                "trajectory_id": key[1],
                "prefix_length": stop,
                "context_poi_ids": [row["poi_id"] for row in observed],
                "context_times": [row["observed_time"] for row in observed],
                # Do not access trajectory[stop]["poi_id"] here.
                "query_time": query_time,
            }
        )
    if len({row["query_id"] for row in inputs}) != len(inputs):
        raise ValueError("quantitative inputs contain duplicate query IDs")
    return inputs


def region_assignment(
    checkpoint_candidate_ids: Sequence[str],
    poi_region: Mapping[str, str],
    region_ids: Sequence[str],
) -> np.ndarray:
    if len(region_ids) != len(set(region_ids)) or not region_ids:
        raise ValueError("region support must be non-empty and unique")
    if UNKNOWN_REGION not in region_ids:
        raise ValueError(f"region support must include {UNKNOWN_REGION}")
    index = {region: position for position, region in enumerate(region_ids)}
    assignment = np.empty(len(checkpoint_candidate_ids), dtype=np.int64)
    for position, poi_id in enumerate(checkpoint_candidate_ids):
        raw_region = poi_region.get(str(poi_id), UNKNOWN_REGION)
        if raw_region not in index:
            raise ValueError(f"POI-region mapping references a region outside the train vocabulary: {raw_region}")
        assignment[position] = index[raw_region]
    return assignment


def aggregate_poi_probabilities(
    poi_probabilities: np.ndarray,
    assignment: np.ndarray,
    number_of_regions: int,
) -> np.ndarray:
    """Pure NumPy reference used by tests and artifact audits."""

    values = np.asarray(poi_probabilities, dtype=np.float64)
    mapping = np.asarray(assignment, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(mapping):
        raise ValueError("POI probabilities and assignment shape mismatch")
    if number_of_regions <= 0 or np.any(mapping < 0) or np.any(mapping >= number_of_regions):
        raise ValueError("invalid POI-to-region assignment")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("POI probabilities must be finite and non-negative")
    output = np.zeros((values.shape[0], number_of_regions), dtype=np.float64)
    for region in range(number_of_regions):
        output[:, region] = values[:, mapping == region].sum(axis=1)
    denominator = output.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("each POI probability row must have positive mass")
    return output / denominator


def _load_contract(data_base: Path, prepared_root: Path, checkpoint_path: Path) -> dict[str, Any]:
    required = {
        "candidate_ids": data_base / "candidate_ids.json",
        "checkpoint": checkpoint_path,
        "prepare_manifest": prepared_root / "manifests" / "prepare_manifest.json",
        "poi_region_map": prepared_root / "representation" / "poi_region_map.json",
        "region_vocabulary": prepared_root / "representation" / "regions.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing quantitative-support inputs: {missing}")
    candidate_ids = [str(value) for value in json.loads(required["candidate_ids"].read_text(encoding="utf-8"))]
    prepare_manifest = json.loads(required["prepare_manifest"].read_text(encoding="utf-8"))
    mapping_payload = json.loads(required["poi_region_map"].read_text(encoding="utf-8"))
    vocabulary_payload = json.loads(required["region_vocabulary"].read_text(encoding="utf-8"))
    poi_region = {str(key): str(value) for key, value in mapping_payload.get("mapping", {}).items()}
    mapping_hash = hashlib.sha256(_canonical_json(dict(sorted(poi_region.items()))).encode("utf-8")).hexdigest()
    if mapping_payload.get("fit_split") != "train" or mapping_payload.get("mapping_hash") != mapping_hash:
        raise ValueError("POI-region map is not a valid train-only immutable mapping")
    source_hashes = prepare_manifest.get("source_hashes", {})
    if source_hashes.get("candidate_ids") != _file_hash(required["candidate_ids"]):
        raise ValueError("candidate_ids changed after faithful prepare")
    if source_hashes.get("neural_checkpoint") != _file_hash(checkpoint_path):
        raise ValueError("frozen Neural-CGM checkpoint changed after faithful prepare")
    if vocabulary_payload.get("fit_split") != "train" or vocabulary_payload.get("unknown_region") != UNKNOWN_REGION:
        raise ValueError("region vocabulary is not train-only")
    if "centroids" in vocabulary_payload:
        fitted_regions = {str(value) for value in vocabulary_payload["centroids"]}
    elif "counts" in vocabulary_payload:
        fitted_regions = {str(value) for value in vocabulary_payload["counts"]}
    else:
        fitted_regions = {str(value) for value in vocabulary_payload.get("regions", [])}
    mapped_regions = set(poi_region.values())
    if not mapped_regions.issubset(fitted_regions):
        raise ValueError("POI-region map references cells outside the fitted train vocabulary")
    # The quantitative teacher ranks the complete train-only vocabulary per
    # query.  It must not inherit prepare's temporary global top-20 prompt
    # support; doing so would discard the very per-query support we need.
    region_ids = sorted(fitted_regions - {UNKNOWN_REGION}) + [UNKNOWN_REGION]
    return {
        "candidate_ids": candidate_ids,
        "prepare_manifest": prepare_manifest,
        "poi_region": poi_region,
        "region_ids": region_ids,
        "checkpoint_hash": _file_hash(checkpoint_path),
        "mapping_hash": mapping_hash,
    }


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - repository requires torch for export
        raise RuntimeError("PyTorch is required for quantitative-support export") from exc


def _checkpoint_contract(checkpoint_path: Path, source_candidate_ids: Sequence[str]) -> tuple[dict[str, Any], Any]:
    torch = _torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_ids = [str(value) for value in checkpoint.get("candidate_ids", [])]
    if checkpoint_ids != list(source_candidate_ids):
        raise ValueError("checkpoint candidate_ids do not exactly match the prepared candidate order")
    config = checkpoint.get("config", {})
    if int(config.get("num_pois", -1)) != len(checkpoint_ids):
        raise ValueError("checkpoint num_pois/candidate_ids mismatch")
    if not isinstance(checkpoint.get("user_map"), dict):
        raise ValueError("checkpoint is missing user_map")
    from ..checkpoint_models import build_checkpoint_model

    model, _ = build_checkpoint_model(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, model


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _find_protocol_lock(
    prepared_root: Path,
    output_root: Path | None,
    explicit_results_root: str | None,
) -> tuple[Path, Path]:
    """Resolve exactly one results-root lock without consulting test data."""

    if explicit_results_root:
        results_root = Path(explicit_results_root).resolve()
        try:
            prepared_root.resolve().relative_to(results_root)
        except ValueError as exc:
            raise ValueError(
                f"prepared root is outside --results-root: prepared={prepared_root} root={results_root}"
            ) from exc
        lock_path = results_root / "locks" / "protocol_lock.json"
        if not lock_path.is_file():
            raise FileNotFoundError(f"test export requires protocol lock: {lock_path}")
        return results_root, lock_path

    candidates: dict[Path, Path] = {}
    starts = [prepared_root.resolve()]
    if output_root is not None:
        starts.append(output_root.resolve())
    for start in starts:
        for root in (start, *start.parents):
            lock_path = root / "locks" / "protocol_lock.json"
            if lock_path.is_file():
                candidates[root] = lock_path
                break
    if not candidates:
        raise FileNotFoundError(
            "test export requires locks/protocol_lock.json in an ancestor of the prepared/output root; "
            "pass --results-root explicitly if artifacts are stored elsewhere"
        )
    if len(candidates) != 1:
        rendered = [str(path) for path in sorted(candidates.values())]
        raise ValueError(f"ambiguous protocol locks for test export: {rendered}")
    return next(iter(candidates.items()))


def validate_test_protocol_lock(
    prepared_root: Path,
    output_root: Path | None,
    *,
    results_root: str | None,
    protocol_path: str | None,
) -> dict[str, Any]:
    """Validate the pilot lock before any held-out query/source is opened."""

    resolved_root, lock_path = _find_protocol_lock(prepared_root, output_root, results_root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ValueError(f"invalid protocol lock object: {lock_path}")
    if lock.get("version") != "beliefmove.protocol-lock.v1":
        raise ValueError(f"unsupported protocol lock version: {lock.get('version')!r}")
    if lock.get("pilot_gate") != "ready":
        raise RuntimeError("test export is blocked because pilot_gate is not ready")
    if lock.get("test_artifacts_read") is not False:
        raise ValueError("protocol lock does not prove target-free pilot provenance")
    if not _is_sha256(lock.get("protocol_hash")) or not _is_sha256(lock.get("pilot_report_hash")):
        raise ValueError("protocol lock contains an invalid protocol/pilot SHA-256")

    selected_protocol = Path(protocol_path or "phase3_beliefmove/protocol.json")
    if not selected_protocol.is_file():
        raise FileNotFoundError(
            f"test export requires the locked protocol document for identity verification: {selected_protocol}"
        )
    protocol = json.loads(selected_protocol.read_text(encoding="utf-8"))
    observed_hash = _canonical_hash(protocol)
    if observed_hash != lock["protocol_hash"]:
        raise ValueError(
            "protocol hash does not match protocol_lock.json; do not export test under a modified protocol"
        )
    return {
        "results_root": str(resolved_root),
        "lock_path": str(lock_path),
        "lock_hash": _file_hash(lock_path),
        "protocol_path": str(selected_protocol.resolve()),
        "protocol_hash": observed_hash,
        "pilot_report_hash": lock["pilot_report_hash"],
        "pilot_gate": "ready",
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    data_base = Path(args.data_base)
    prepared_root = Path(args.prepared_root)
    checkpoint_path = Path(args.checkpoint or data_base / "neural_cgm" / "best.pt")
    errors: list[str] = []
    contract: dict[str, Any] | None = None
    protocol_lock: dict[str, Any] | None = None
    try:
        if "test" in args.splits:
            protocol_lock = validate_test_protocol_lock(
                prepared_root,
                None,
                results_root=getattr(args, "results_root", None),
                protocol_path=getattr(args, "protocol", None),
            )
        contract = _load_contract(data_base, prepared_root, checkpoint_path)
        _checkpoint_contract(checkpoint_path, contract["candidate_ids"])
        splits = tuple(args.splits)
        for split in splits:
            query_path = prepared_root / "queries" / f"{split}.jsonl"
            source_path = data_base / "getnext" / ("val.csv" if split == "validation" else f"{split}.csv")
            if not query_path.is_file() or not source_path.is_file():
                raise FileNotFoundError(f"missing split inputs: {query_path}, {source_path}")
            rows = _read_jsonl(query_path)
            split_manifest = contract["prepare_manifest"].get("split_hashes", {}).get(split, {})
            if split_manifest.get("queries") != _file_hash(query_path):
                raise ValueError(f"prepared query hash mismatch for split={split}")
            if split_manifest.get("query_count") != len(rows):
                raise ValueError(f"prepared query count mismatch for split={split}")
            if contract["prepare_manifest"].get("source_hashes", {}).get(split) != _file_hash(source_path):
                raise ValueError(f"source CSV hash mismatch for split={split}")
            build_quantitative_inputs(rows, source_path, contract["candidate_ids"], contract["poi_region"])
    except Exception as exc:  # audit reports all failures through one explicit gate
        errors.append(f"{type(exc).__name__}: {exc}")
    result = {
        "action": "quantitative-support-audit",
        "gate": "ready" if not errors else "incomplete",
        "errors": errors,
        "target_artifacts_read": False,
        "checkpoint_hash": contract.get("checkpoint_hash") if contract else None,
        "protocol_lock": protocol_lock,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def _device(torch: Any, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _slot(value: str) -> int:
    from ..neural_cgm import _timestamp

    stamp = _timestamp(value)
    if getattr(stamp, "hour", None) is None or bool(getattr(stamp, "isna", lambda: False)()):
        return 0
    try:
        missing = bool(np.asarray(stamp != stamp).item())
    except Exception:
        missing = False
    return 0 if missing else int(stamp.hour * 2 + stamp.minute // 30)


def export(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    data_base = Path(args.data_base)
    prepared_root = Path(args.prepared_root)
    checkpoint_path = Path(args.checkpoint or data_base / "neural_cgm" / "best.pt")
    requested_output = Path(args.output_root) if args.output_root else None
    protocol_lock = None
    if "test" in args.splits:
        # Check the pilot/protocol lock before opening any test query or source
        # file and before creating a partial held-out artifact.
        protocol_lock = validate_test_protocol_lock(
            prepared_root,
            requested_output,
            results_root=getattr(args, "results_root", None),
            protocol_path=getattr(args, "protocol", None),
        )
    contract = _load_contract(data_base, prepared_root, checkpoint_path)
    checkpoint, model = _checkpoint_contract(checkpoint_path, contract["candidate_ids"])
    assignment = region_assignment(
        checkpoint["candidate_ids"], contract["poi_region"], contract["region_ids"]
    )
    assignment_tensor = torch.tensor(assignment, dtype=torch.long)
    device = _device(torch, args.device)
    model.to(device).eval()
    assignment_tensor = assignment_tensor.to(device)
    candidate_index = {str(value): position for position, value in enumerate(checkpoint["candidate_ids"])}
    output_base = Path(args.output_root or prepared_root / "quantitative") / contract["checkpoint_hash"][:16]
    summaries: dict[str, Any] = {}

    for split in args.splits:
        query_path = prepared_root / "queries" / f"{split}.jsonl"
        source_path = data_base / "getnext" / ("val.csv" if split == "validation" else f"{split}.csv")
        rows = _read_jsonl(query_path)
        inputs = build_quantitative_inputs(
            rows, source_path, contract["candidate_ids"], contract["poi_region"]
        )
        region_probability_rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(inputs), args.batch_size):
                batch = inputs[start : start + args.batch_size]
                lengths = torch.tensor([len(row["context_poi_ids"]) for row in batch], dtype=torch.long)
                width = int(lengths.max())
                poi = torch.zeros((len(batch), width), dtype=torch.long)
                slots = torch.zeros((len(batch), width), dtype=torch.long)
                for row_index, row in enumerate(batch):
                    indices = [candidate_index[str(value)] for value in row["context_poi_ids"]]
                    times = [_slot(value) for value in row["context_times"]]
                    poi[row_index, : len(indices)] = torch.tensor(indices, dtype=torch.long)
                    slots[row_index, : len(times)] = torch.tensor(times, dtype=torch.long)
                users = torch.tensor(
                    [checkpoint["user_map"].get(str(row["user_id"]), len(checkpoint["user_map"])) for row in batch],
                    dtype=torch.long,
                )
                query_slots = torch.tensor([_slot(str(row["query_time"])) for row in batch], dtype=torch.long)
                logits = model(
                    poi.to(device), slots.to(device), lengths.to(device), users.to(device), query_slots.to(device)
                )
                poi_probability = torch.softmax(logits, dim=1)
                region_probability = torch.zeros(
                    (len(batch), len(contract["region_ids"])), dtype=poi_probability.dtype, device=device
                )
                region_probability.scatter_add_(
                    1, assignment_tensor.unsqueeze(0).expand(len(batch), -1), poi_probability
                )
                region_probability_rows.append(region_probability.cpu().numpy().astype(np.float32))
        probabilities = np.concatenate(region_probability_rows) if region_probability_rows else np.empty(
            (0, len(contract["region_ids"])), dtype=np.float32
        )
        if len(probabilities) != len(inputs) or (len(probabilities) and not np.allclose(probabilities.sum(1), 1.0, atol=1e-5)):
            raise ValueError(f"invalid region probabilities for split={split}")
        top_k = min(int(args.top_k), len(contract["region_ids"]))
        order = np.argsort(-probabilities, axis=1, kind="stable")[:, :top_k]
        top_probability = np.take_along_axis(probabilities, order, axis=1)
        clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=1).astype(np.float32)
        sorted_probability = np.sort(probabilities, axis=1)
        margin = (
            sorted_probability[:, -1] - sorted_probability[:, -2]
            if probabilities.shape[1] >= 2
            else sorted_probability[:, -1]
        ).astype(np.float32)
        json_rows = []
        for index, row in enumerate(inputs):
            json_rows.append(
                {
                    "query_id": row["query_id"],
                    "top_regions": [contract["region_ids"][value] for value in order[index]],
                    "top_probabilities": [float(value) for value in top_probability[index]],
                    "entropy": float(entropy[index]),
                    "margin": float(margin[index]),
                }
            )
        split_root = output_base / split
        _write_npz_immutable(
            split_root / "region_support.npz",
            {
                "query_ids": np.asarray([row["query_id"] for row in inputs]),
                "region_ids": np.asarray(contract["region_ids"]),
                "probabilities": probabilities,
                "top_region_indices": order.astype(np.int32),
                "top_region_probabilities": top_probability.astype(np.float32),
                "entropy": entropy,
                "margin": margin,
            },
        )
        _write_immutable(split_root / "topk.jsonl", _jsonl_bytes(json_rows))
        manifest = {
            "version": EXPORT_VERSION,
            "split": split,
            "queries": len(inputs),
            "query_hash": _file_hash(query_path),
            "source_csv_hash": _file_hash(source_path),
            "checkpoint_hash": contract["checkpoint_hash"],
            "poi_region_mapping_hash": contract["mapping_hash"],
            "region_ids": contract["region_ids"],
            "region_support_source": "complete train-only region vocabulary",
            "top_k": top_k,
            "aggregation": "sum of frozen POI softmax mass per train-only region",
            "all_prefix": True,
            "target_artifacts_read": False,
            "target_label_fields_emitted": False,
            "known_query_covariates": ["prediction_timestamp"],
            "protocol_lock": protocol_lock if split == "test" else None,
            "device": str(device),
        }
        _write_immutable(split_root / "manifest.json", _json_bytes(manifest))
        summaries[split] = {"queries": len(inputs), "output": str(split_root)}

    result = {
        "action": "quantitative-support-export",
        "gate": "ready",
        "checkpoint_hash": contract["checkpoint_hash"],
        "target_artifacts_read": False,
        "protocol_lock": protocol_lock,
        "splits": summaries,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def bind_teacher_queries(args: argparse.Namespace) -> dict[str, Any]:
    """Bind per-query quantitative top-k regions to immutable teacher inputs.

    The original causal query remains untouched.  This derived view contains
    only the same observed prefix/query time plus a per-query candidate-region
    support.  It never opens a target artifact.
    """

    data_base = Path(args.data_base)
    prepared_root = Path(args.prepared_root)
    checkpoint_path = Path(args.checkpoint or data_base / "neural_cgm" / "best.pt")
    contract = _load_contract(data_base, prepared_root, checkpoint_path)
    checkpoint_slug = contract["checkpoint_hash"][:16]
    quantitative_root = Path(args.output_root or prepared_root / "quantitative") / checkpoint_slug
    destination_root = prepared_root / "teacher_queries" / checkpoint_slug
    split_results: dict[str, Any] = {}
    split_hashes: dict[str, Any] = {}

    for split in args.splits:
        if split == "test":
            raise ValueError("teacher-query binding before protocol lock is restricted to train/validation")
        query_path = prepared_root / "queries" / f"{split}.jsonl"
        topk_path = quantitative_root / split / "topk.jsonl"
        support_manifest_path = quantitative_root / split / "manifest.json"
        queries = _read_jsonl(query_path)
        topk_rows = _read_jsonl(topk_path)
        topk = {str(row["query_id"]): row for row in topk_rows}
        if len(topk) != len(topk_rows) or set(topk) != {str(row["query_id"]) for row in queries}:
            raise ValueError(f"quantitative top-k/query coverage mismatch for split={split}")
        support_manifest = json.loads(support_manifest_path.read_text(encoding="utf-8"))
        if (
            support_manifest.get("checkpoint_hash") != contract["checkpoint_hash"]
            or support_manifest.get("query_hash") != _file_hash(query_path)
            or support_manifest.get("target_artifacts_read") is not False
        ):
            raise ValueError(f"stale or unsafe quantitative support for split={split}")

        bound: list[dict[str, Any]] = []
        allowed_regions = set(contract["region_ids"])
        for query in queries:
            query_id = str(query["query_id"])
            regions = [str(value) for value in topk[query_id].get("top_regions", [])]
            if not regions or len(regions) != len(set(regions)) or not set(regions) <= allowed_regions:
                raise ValueError(f"invalid quantitative region support for query={query_id}")
            candidate_regions = list(dict.fromkeys(regions + [UNKNOWN_REGION, OTHER_REGION]))
            derived = dict(query)
            derived["candidate_regions"] = candidate_regions
            derived["candidate_support_provenance"] = {
                "version": BIND_VERSION,
                "checkpoint_hash": contract["checkpoint_hash"],
                "topk_hash": _file_hash(topk_path),
            }
            bound.append(derived)

        destination = destination_root / f"{split}.jsonl"
        _write_immutable(destination, _jsonl_bytes(bound))
        split_hashes[split] = {
            "source_query_hash": _file_hash(query_path),
            "quantitative_manifest_hash": _file_hash(support_manifest_path),
            "topk_hash": _file_hash(topk_path),
            "bound_query_hash": _file_hash(destination),
            "queries": len(bound),
        }
        split_results[split] = {"queries": len(bound), "output": str(destination)}

    manifest = {
        "version": BIND_VERSION,
        "dataset": contract["prepare_manifest"]["dataset"],
        "unit": contract["prepare_manifest"]["unit"],
        "checkpoint_hash": contract["checkpoint_hash"],
        "dataset_hash": contract["prepare_manifest"]["dataset_hash"],
        "ontology_hash": contract["prepare_manifest"]["ontology_hash"],
        "allowed_splits": ["train", "validation"],
        "target_artifacts_read": False,
        "split_hashes": split_hashes,
    }
    _write_immutable(destination_root / "manifest.json", _json_bytes(manifest))
    result = {
        "action": "quantitative-support-bind",
        "gate": "ready",
        "checkpoint_hash": contract["checkpoint_hash"],
        "target_artifacts_read": False,
        "output": str(destination_root),
        "splits": split_results,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export frozen Neural-CGM support on train-only regions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "export", "bind"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data-base", required=True)
        sub.add_argument("--prepared-root", required=True)
        sub.add_argument("--checkpoint")
        sub.add_argument("--results-root", help="Results root containing locks/protocol_lock.json")
        sub.add_argument(
            "--protocol",
            default="phase3_beliefmove/protocol.json",
            help="Protocol JSON whose canonical hash must match the lock for test export",
        )
        sub.add_argument(
            "--splits", nargs="+", choices=("train", "validation", "test"), default=("train", "validation")
        )
        if command in ("export", "bind"):
            sub.add_argument("--output-root")
        if command == "export":
            sub.add_argument("--batch-size", type=int, default=256)
            sub.add_argument("--top-k", type=int, default=20)
            sub.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        result = audit(args)
        if result["gate"] != "ready":
            raise SystemExit(3)
    elif args.command == "export":
        if args.batch_size <= 0 or args.top_k <= 0:
            raise ValueError("batch-size and top-k must be positive")
        export(args)
    else:
        bind_teacher_queries(args)


if __name__ == "__main__":
    main()
