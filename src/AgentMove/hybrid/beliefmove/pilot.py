from __future__ import annotations

"""Train/validation-only evaluator for the BeliefMove teacher pilot.

The evaluator deliberately has no test-split argument.  It consumes the raw
prepared train/validation query and target artifacts, a base ``pilot`` teacher
cache, and independent repeat caches.  The resulting
``manifests/pilot_evaluation.json`` is therefore a derived, hash-bound pilot
artifact rather than a hand-written gate override.

Ontology labels are not guessed here.  They must either already be present in
the prepared target rows as ``target_activity``/``target_destination`` or be
provided by immutable ``ontology_labels/{train,validation}.jsonl`` artifacts.
Missing or stale ontology labels fail closed.
"""

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cpt import BlendedCPT, DataCPT, TeacherSoftCountCPT
from .ontology import (
    ACTIVITY_ONTOLOGY,
    DESTINATION_ONTOLOGY,
    INTENT_ONTOLOGY,
    ontology_hash,
)
from .schema import StructuredBelief, hash_text


PILOT_EVALUATION_VERSION = "beliefmove.pilot-evaluation.v2"
TEACHER_CACHE_VERSION = "beliefmove.teacher-cache.v2"
TEACHER_NORMALIZATION_VERSION = "beliefmove.teacher-l1-weights.v1"
DANGLING_RECOVERY_VERSION = "beliefmove.dangling-attempt-recovery.v1"
DANGLING_RECOVERY_REASON = (
    "IndeterminateOutcome: collector stopped after durable attempt and before durable belief"
)
ONTOLOGY_LABEL_VERSION = "beliefmove.ontology-labels.v1"
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PILOT_SMOOTHING = 1.0
REPEAT_JSD_LIMIT = 0.10
_GLOBAL_CONTEXT = ("__GLOBAL__",)
_DISTRIBUTION_FIELDS = ("activity", "intent", "destination", "next_region")
_VALIDATION_PARTITION_VERSION = "beliefmove.pilot-validation-partition.v1"
_BASE_REQUESTED_BY_SPLIT = {"train": 500, "validation": 200}
_REPEAT_REQUESTED_BY_SPLIT = {"train": 0, "validation": 50}
_CANONICAL_PROBABILITY_TOLERANCE = 1e-12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PilotEvaluationError(ValueError):
    """Raised when a pilot input violates the frozen evaluation contract."""


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotEvaluationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotEvaluationError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _index(rows: Sequence[Mapping[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "query_id" not in row:
            raise PilotEvaluationError(f"missing query_id in {path}")
        query_id = str(row["query_id"])
        if not query_id or query_id in result:
            raise PilotEvaluationError(f"duplicate/empty query_id={query_id!r} in {path}")
        result[query_id] = dict(row)
    return result


def _strict_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotEvaluationError(f"{name} must be a non-negative integer")
    return value


def _expected_recovery_belief(
    query: Mapping[str, Any],
    prompt: str,
    selection: Mapping[str, Any],
) -> StructuredBelief:
    """Recreate the only belief payload allowed for dangling recovery."""

    regions = [str(value) for value in query.get("candidate_regions", [])]
    if not regions:
        raise PilotEvaluationError("dangling recovery query has empty region support")
    model_identity = (
        f"{selection.get('platform')}:{selection.get('model')}@"
        f"{selection.get('model_digest')}"
    )
    return StructuredBelief.create(
        query=_canonical_json(query),
        prompt=prompt,
        model_identity=model_identity,
        activity={key: 1.0 if key == "other" else 0.0 for key in ACTIVITY_ONTOLOGY},
        intent={key: 1.0 if key == "new-zone" else 0.0 for key in INTENT_ONTOLOGY},
        destination={
            key: 1.0 if key == "other" else 0.0
            for key in DESTINATION_ONTOLOGY
        },
        next_region={key: 1.0 / len(regions) for key in regions},
        confidence=0.0,
    )


def _teacher_record_contract(
    record: Mapping[str, Any],
    belief: StructuredBelief,
    *,
    role: str,
    query_id: str,
) -> None:
    """Validate the v2 structured-output and canonicalization provenance."""

    response_format_hash = record.get("response_format_hash")
    if not isinstance(response_format_hash, str) or _SHA256_RE.fullmatch(response_format_hash) is None:
        raise PilotEvaluationError(
            f"{role} belief has invalid response_format_hash for query={query_id}"
        )

    raw_valid = record.get("raw_valid")
    usable_output = record.get("usable_output")
    fallback = record.get("fallback")
    transport_error = record.get("transport_error")
    if type(raw_valid) is not bool or type(usable_output) is not bool or type(fallback) is not bool:
        raise PilotEvaluationError(
            f"{role} belief has non-boolean validity provenance for query={query_id}"
        )
    if usable_output is not raw_valid:
        raise PilotEvaluationError(
            f"{role} belief usable_output/raw_valid mismatch for query={query_id}"
        )
    if not isinstance(transport_error, str):
        raise PilotEvaluationError(
            f"{role} belief has invalid transport provenance for query={query_id}"
        )
    if raw_valid and (fallback or transport_error):
        raise PilotEvaluationError(
            f"{role} usable belief is marked fallback/transport-error for query={query_id}"
        )
    if not raw_valid and not fallback:
        raise PilotEvaluationError(
            f"{role} unusable belief is not marked fallback for query={query_id}"
        )

    normalization = record.get("normalization")
    if not isinstance(normalization, Mapping):
        raise PilotEvaluationError(
            f"{role} belief is missing normalization provenance for query={query_id}"
        )
    if normalization.get("version") != TEACHER_NORMALIZATION_VERSION:
        raise PilotEvaluationError(
            f"{role} belief normalization version mismatch for query={query_id}"
        )
    if type(normalization.get("applied")) is not bool:
        raise PilotEvaluationError(
            f"{role} belief normalization.applied must be boolean for query={query_id}"
        )
    raw_sums = normalization.get("raw_sums")
    if not isinstance(raw_sums, Mapping):
        raise PilotEvaluationError(
            f"{role} belief normalization.raw_sums is invalid for query={query_id}"
        )
    if raw_valid:
        if set(raw_sums) != set(_DISTRIBUTION_FIELDS):
            raise PilotEvaluationError(
                f"{role} belief normalization.raw_sums support mismatch for query={query_id}"
            )
        parsed_sums: list[float] = []
        for field in _DISTRIBUTION_FIELDS:
            raw_sum = raw_sums[field]
            if isinstance(raw_sum, bool) or not isinstance(raw_sum, (int, float)):
                raise PilotEvaluationError(
                    f"{role} belief normalization raw sum is non-numeric for query={query_id}"
                )
            value = float(raw_sum)
            if not math.isfinite(value) or value <= 0.0:
                raise PilotEvaluationError(
                    f"{role} belief normalization raw sum is not positive finite for query={query_id}"
                )
            parsed_sums.append(value)
        expected_applied = any(
            not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in parsed_sums
        )
        if normalization.get("applied") is not expected_applied:
            raise PilotEvaluationError(
                f"{role} belief normalization.applied is inconsistent for query={query_id}"
            )
    elif raw_sums or normalization.get("applied") is not False:
        raise PilotEvaluationError(
            f"{role} fallback belief has invalid normalization provenance for query={query_id}"
        )

    declared_error = normalization.get("canonical_probability_error")
    if isinstance(declared_error, bool) or not isinstance(declared_error, (int, float)):
        raise PilotEvaluationError(
            f"{role} belief canonical probability error is non-numeric for query={query_id}"
        )
    declared_error = float(declared_error)
    observed_error = max(
        abs(math.fsum(getattr(belief, field).values()) - 1.0)
        for field in _DISTRIBUTION_FIELDS
    )
    if (
        not math.isfinite(declared_error)
        or declared_error < 0.0
        or declared_error > _CANONICAL_PROBABILITY_TOLERANCE
        or observed_error > _CANONICAL_PROBABILITY_TOLERANCE
    ):
        raise PilotEvaluationError(
            f"{role} belief canonical probability error exceeds tolerance for query={query_id}"
        )


def _validation_partition(query_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Create a label-blind deterministic selection/evaluation partition."""

    unique = [str(query_id) for query_id in query_ids]
    if len(unique) != len(set(unique)):
        raise PilotEvaluationError("validation partition contains duplicate query IDs")
    if len(unique) < 2:
        raise PilotEvaluationError(
            "pilot validation requires at least two queries for alpha selection and held-out evaluation"
        )
    ordered = sorted(
        unique,
        key=lambda query_id: (
            hashlib.sha256(
                f"{_VALIDATION_PARTITION_VERSION}:{query_id}".encode("utf-8")
            ).hexdigest(),
            query_id,
        ),
    )
    boundary = max(1, len(ordered) // 2)
    selection, heldout = ordered[:boundary], ordered[boundary:]
    if not selection or not heldout:  # defensive guard for future partition changes
        raise PilotEvaluationError("alpha-selection and held-out validation partitions must be non-empty")
    return selection, heldout


def _validate_alpha_grid(grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in grid)
    if values != ALPHA_GRID:
        raise PilotEvaluationError(
            f"pilot alpha grid is frozen as {list(ALPHA_GRID)}, observed {list(values)}"
        )
    return values


def _previous_region(query: Mapping[str, Any]) -> tuple[str, str]:
    prefix = query.get("observed_prefix")
    if not isinstance(prefix, list) or not prefix or not isinstance(prefix[-1], Mapping):
        raise PilotEvaluationError(f"query={query.get('query_id')} has no causal previous region")
    region = str(prefix[-1].get("region", ""))
    if not region:
        raise PilotEvaluationError(f"query={query.get('query_id')} has empty previous region")
    return "previous_region", region


def _nll(probabilities: Mapping[str, float], label: str) -> float:
    if label not in probabilities:
        raise PilotEvaluationError(f"label={label!r} is outside the frozen support")
    return -math.log(max(float(probabilities[label]), 1e-12))


def _mean_nll(rows: Sequence[tuple[Mapping[str, float], str]]) -> float:
    if not rows:
        raise PilotEvaluationError("cannot compute NLL from an empty validation set")
    return float(np.mean([_nll(probabilities, label) for probabilities, label in rows]))


def _empirical_prior(labels: Sequence[str], observations: Sequence[str]) -> dict[str, float]:
    cpt = DataCPT(labels, smoothing=PILOT_SMOOTHING).fit(
        [(_GLOBAL_CONTEXT, label) for label in observations], split="train"
    )
    return cpt.distribution(_GLOBAL_CONTEXT)


def _jsd(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if set(left) != set(right):
        raise PilotEvaluationError("repeat beliefs use different probability supports")
    keys = sorted(left)
    first = np.asarray([float(left[key]) for key in keys], dtype=np.float64)
    second = np.asarray([float(right[key]) for key in keys], dtype=np.float64)
    if (
        np.any(~np.isfinite(first))
        or np.any(~np.isfinite(second))
        or np.any(first < 0)
        or np.any(second < 0)
        or not np.isclose(first.sum(), 1.0)
        or not np.isclose(second.sum(), 1.0)
    ):
        raise PilotEvaluationError("repeat belief is not a normalized finite distribution")
    middle = 0.5 * (first + second)

    def divergence(values: np.ndarray) -> float:
        positive = values > 0
        return float(np.sum(values[positive] * np.log(values[positive] / middle[positive])))

    return 0.5 * divergence(first) + 0.5 * divergence(second)


def _prepared_split(
    unit_root: Path,
    prepare_manifest: Mapping[str, Any],
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    query_path = unit_root / "queries" / f"{split}.jsonl"
    target_path = unit_root / "targets" / f"{split}.jsonl"
    queries = _index(_read_jsonl(query_path), query_path)
    targets = _index(_read_jsonl(target_path), target_path)
    if set(queries) != set(targets):
        raise PilotEvaluationError(f"prepared query/target alignment mismatch for split={split}")
    if any(row.get("split") != split for row in queries.values()) or any(
        row.get("split") != split for row in targets.values()
    ):
        raise PilotEvaluationError(f"prepared split provenance mismatch for split={split}")
    expected = prepare_manifest.get("split_hashes", {}).get(split, {})
    observed_hashes = {"queries": _file_hash(query_path), "targets": _file_hash(target_path)}
    if expected.get("queries") != observed_hashes["queries"] or expected.get("targets") != observed_hashes["targets"]:
        raise PilotEvaluationError(f"prepared split hash mismatch for split={split}")
    if int(expected.get("query_count", -1)) != len(queries):
        raise PilotEvaluationError(f"prepared query count mismatch for split={split}")
    return queries, targets, observed_hashes


def _ontology_labels(
    unit_root: Path,
    prepare_manifest: Mapping[str, Any],
    prepared: Mapping[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]],
    label_root: Path | None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    expected_ontology_hash = ontology_hash()
    if prepare_manifest.get("ontology_hash") != expected_ontology_hash:
        raise PilotEvaluationError(
            "prepared manifest is not bound to the frozen category-to-ontology contract"
        )

    embedded = all(
        all("target_activity" in row and "target_destination" in row for row in prepared[split][1].values())
        for split in ("train", "validation")
    )
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    if embedded:
        for split in ("train", "validation"):
            labels[split] = {
                query_id: {
                    "query_id": query_id,
                    "split": split,
                    "source_hash": query["source_hash"],
                    "activity": target["target_activity"],
                    "destination": target["target_destination"],
                }
                for query_id, query in prepared[split][0].items()
                for target in (prepared[split][1][query_id],)
            }
        provenance = {
            "source": "prepared-targets",
            "ontology_hash": expected_ontology_hash,
            "split_hashes": {
                split: prepared[split][2]["targets"] for split in ("train", "validation")
            },
        }
    else:
        root = label_root or unit_root / "ontology_labels"
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        required = {
            "version": ONTOLOGY_LABEL_VERSION,
            "dataset": prepare_manifest.get("dataset"),
            "unit": prepare_manifest.get("unit"),
            "fit_split": "train",
            "ontology_hash": expected_ontology_hash,
            "test_artifacts_read": False,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise PilotEvaluationError(f"ontology-label manifest mismatch for {key}")
        mapping_hash = str(manifest.get("mapping_hash", ""))
        if len(mapping_hash) != 64 or any(character not in "0123456789abcdef" for character in mapping_hash):
            raise PilotEvaluationError("ontology-label manifest is missing a SHA-256 mapping_hash")
        labels = {}
        observed: dict[str, str] = {}
        for split in ("train", "validation"):
            path = root / f"{split}.jsonl"
            rows = _index(_read_jsonl(path), path)
            queries = prepared[split][0]
            if set(rows) != set(queries):
                raise PilotEvaluationError(f"ontology-label coverage mismatch for split={split}")
            for query_id, row in rows.items():
                if (
                    row.get("split") != split
                    or row.get("source_hash") != queries[query_id].get("source_hash")
                    or row.get("mapping_hash") != mapping_hash
                ):
                    raise PilotEvaluationError(
                        f"ontology-label provenance mismatch for query={query_id}"
                    )
            labels[split] = rows
            observed[split] = _file_hash(path)
            if manifest.get("split_hashes", {}).get(split) != observed[split]:
                raise PilotEvaluationError(f"ontology-label hash mismatch for split={split}")
        provenance = {
            "source": "ontology-label-artifacts",
            "manifest_hash": _file_hash(manifest_path),
            "mapping_hash": mapping_hash,
            "ontology_hash": expected_ontology_hash,
            "split_hashes": observed,
        }

    for split in ("train", "validation"):
        for query_id, row in labels[split].items():
            if str(row.get("activity")) not in ACTIVITY_ONTOLOGY:
                raise PilotEvaluationError(f"unknown activity ontology label for query={query_id}")
            if str(row.get("destination")) not in DESTINATION_ONTOLOGY:
                raise PilotEvaluationError(f"unknown destination ontology label for query={query_id}")
    return labels, provenance


def _cache(
    cache_root: Path,
    *,
    role: str,
    prepare_manifest: Mapping[str, Any],
    prepared_queries: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str]]:
    selection_path = cache_root / "selection_manifest.json"
    manifest_path = cache_root / "manifest.json"
    attempts_path = cache_root / "attempts.jsonl"
    beliefs_path = cache_root / "beliefs.jsonl"
    selection = _read_json(selection_path)
    manifest = _read_json(manifest_path)
    attempts = _index(_read_jsonl(attempts_path), attempts_path)
    beliefs = _index(_read_jsonl(beliefs_path), beliefs_path)

    if selection.get("version") != TEACHER_CACHE_VERSION or manifest.get("version") != TEACHER_CACHE_VERSION:
        raise PilotEvaluationError(f"{role} cache is not a {TEACHER_CACHE_VERSION} artifact")

    for key in ("dataset", "unit", "dataset_hash", "ontology_hash"):
        expected = prepare_manifest.get(key)
        if selection.get(key) != expected:
            raise PilotEvaluationError(f"{role} cache selection mismatch for {key}")
    if selection.get("test_queries_read") is not False or manifest.get("gate") != "ready":
        raise PilotEvaluationError(f"{role} cache is not a ready train/validation-only cache")
    if manifest.get("prompt_contract_hash") != selection.get("prompt_contract_hash"):
        raise PilotEvaluationError(f"{role} cache prompt contract mismatch")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(selection.get("model_digest", ""))):
        raise PilotEvaluationError(f"{role} cache model_digest is not a pinned SHA-256 digest")

    selected: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        ids = [str(value) for value in selection.get("query_ids", {}).get(split, [])]
        if len(ids) != len(set(ids)) or any(query_id not in prepared_queries[split] for query_id in ids):
            raise PilotEvaluationError(f"{role} cache has invalid selected IDs for split={split}")
        rows = [prepared_queries[split][query_id] for query_id in ids]
        if selection.get("query_hashes", {}).get(split) != _canonical_hash(rows):
            raise PilotEvaluationError(f"{role} cache selected-query hash mismatch for split={split}")
        selected[split] = ids

    expected_requested = (
        _BASE_REQUESTED_BY_SPLIT if role == "base" else _REPEAT_REQUESTED_BY_SPLIT
    )
    requested_payload = selection.get("requested_by_split")
    available_payload = selection.get("available_by_split")
    if not isinstance(requested_payload, Mapping) or not isinstance(available_payload, Mapping):
        raise PilotEvaluationError(
            f"{role} cache selection is missing requested_by_split/available_by_split budget provenance"
        )
    for split in ("train", "validation"):
        requested = _strict_count(
            requested_payload.get(split), name=f"{role}.requested_by_split.{split}"
        )
        available = _strict_count(
            available_payload.get(split), name=f"{role}.available_by_split.{split}"
        )
        if requested != expected_requested[split]:
            raise PilotEvaluationError(
                f"{role} cache requested budget mismatch for split={split}: "
                f"expected={expected_requested[split]} observed={requested}"
            )
        if available != len(prepared_queries[split]):
            raise PilotEvaluationError(
                f"{role} cache available-query count mismatch for split={split}: "
                f"expected={len(prepared_queries[split])} observed={available}"
            )
        expected_selected = min(requested, available)
        if len(selected[split]) != expected_selected:
            raise PilotEvaluationError(
                f"{role} cache selected-query count violates the budget contract for split={split}: "
                f"expected={expected_selected} observed={len(selected[split])}"
            )
        if manifest.get("requested_by_split", {}).get(split) != requested or manifest.get(
            "available_by_split", {}
        ).get(split) != available:
            raise PilotEvaluationError(f"{role} cache manifest budget provenance mismatch for split={split}")
    expected_ids = set(selected["train"]) | set(selected["validation"])
    if not expected_ids or set(attempts) != expected_ids or set(beliefs) != expected_ids:
        raise PilotEvaluationError(f"{role} cache is incomplete or contains unexpected queries")
    if role == "base" and (not selected["train"] or not selected["validation"]):
        raise PilotEvaluationError("base pilot cache must include train and validation queries")
    if role.startswith("repeat") and (selected["train"] or not selected["validation"]):
        raise PilotEvaluationError("repeat cache must contain validation queries only")
    if role == "base" and selection.get("cache_tag") != "pilot":
        raise PilotEvaluationError("base cache_tag must be 'pilot'")
    if role.startswith("repeat") and not str(selection.get("cache_tag", "")).startswith("pilot-repeat-"):
        raise PilotEvaluationError("repeat cache_tag must start with 'pilot-repeat-'")
    if int(manifest.get("selected_queries", -1)) != len(expected_ids) or int(
        manifest.get("completed_queries", -1)
    ) != len(expected_ids):
        raise PilotEvaluationError(f"{role} cache manifest count mismatch")

    ontology = str(prepare_manifest["ontology_hash"])
    dataset = str(prepare_manifest["dataset_hash"])
    for query_id, record in beliefs.items():
        split = str(record.get("split", ""))
        if split not in ("train", "validation") or query_id not in selected[split]:
            raise PilotEvaluationError(f"{role} belief split mismatch for query={query_id}")
        query = prepared_queries[split][query_id]
        prompt = record.get("prompt")
        raw_response = record.get("raw_response")
        if (
            record.get("source_hash") != _canonical_hash(query)
            or record.get("dataset_hash") != dataset
            or record.get("ontology_hash") != ontology
            or int(record.get("physical_request", -1)) != 1
            or not isinstance(prompt, str)
            or record.get("prompt_hash") != hash_text(prompt)
            or not isinstance(raw_response, str)
            or record.get("raw_response_hash") != hash_text(raw_response)
        ):
            raise PilotEvaluationError(f"{role} belief provenance mismatch for query={query_id}")
        belief = StructuredBelief.from_dict(record.get("belief", {}))
        if (
            set(belief.activity) != set(ACTIVITY_ONTOLOGY)
            or set(belief.destination) != set(DESTINATION_ONTOLOGY)
            or set(belief.next_region) != set(query.get("candidate_regions", []))
            or belief.query_hash != hash_text(_canonical_json(query))
            or belief.prompt_hash != record.get("prompt_hash")
            or belief.model_hash != record.get("model_hash")
        ):
            raise PilotEvaluationError(f"{role} structured belief contract mismatch for query={query_id}")
        attempt = attempts[query_id]
        for key in (
            "split",
            "source_hash",
            "prompt_hash",
            "response_format_hash",
            "model_hash",
            "dataset_hash",
            "ontology_hash",
            "physical_request",
        ):
            if attempt.get(key) != record.get(key):
                raise PilotEvaluationError(
                    f"{role} attempt/belief provenance mismatch for query={query_id} field={key}"
                )
        recovery = record.get("recovery")
        if recovery is not None:
            expected_recovery = {
                "version": DANGLING_RECOVERY_VERSION,
                "kind": "indeterminate-outcome-fallback",
                "reason": DANGLING_RECOVERY_REASON,
                "request_outcome": "indeterminate",
                "attempt_hash": _canonical_hash(attempt),
                "selection_contract_hash": selection.get("selection_contract_hash"),
                "request_budget_consumed": True,
                "response_available": False,
                "usage_known": False,
            }
            if role != "base" or recovery != expected_recovery:
                raise PilotEvaluationError(
                    f"{role} belief has invalid dangling-recovery provenance for query={query_id}"
                )
            expected_recovery_fields = {
                "raw_response": "",
                "raw_response_hash": hash_text(""),
                "raw_valid": False,
                "usable_output": False,
                "fallback": True,
                "fallback_reason": DANGLING_RECOVERY_REASON,
                # Indeterminate process outcome is not a confirmed transport error.
                "transport_error": "",
                "latency_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "token_source": "unknown-indeterminate-outcome",
            }
            if any(
                record.get(field) != expected
                for field, expected in expected_recovery_fields.items()
            ):
                raise PilotEvaluationError(
                    f"{role} belief has invalid dangling-recovery fields for query={query_id}"
                )
            expected_belief = _expected_recovery_belief(query, prompt, selection)
            if belief.to_dict() != expected_belief.to_dict():
                raise PilotEvaluationError(
                    f"{role} belief is not the exact deterministic dangling-recovery "
                    f"fallback for query={query_id}"
                )
        _teacher_record_contract(record, belief, role=role, query_id=query_id)
        if bool(record.get("fallback")) and belief.confidence != 0.0:
            raise PilotEvaluationError(f"{role} fallback belief must have zero confidence")
        if role.startswith("repeat") and (
            record.get("raw_valid") is not True
            or bool(record.get("fallback"))
            or bool(record.get("transport_error"))
        ):
            raise PilotEvaluationError(
                f"{role} cache contains an invalid/fallback/transport-error belief for query={query_id}"
            )

    hashes = {
        "selection_manifest": _file_hash(selection_path),
        "manifest": _file_hash(manifest_path),
        "attempts": _file_hash(attempts_path),
        "beliefs": _file_hash(beliefs_path),
    }
    return beliefs, {**selection, "selected": selected}, hashes


def _teacher_prior_gate(
    field: str,
    labels: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_beliefs: Mapping[str, Mapping[str, Any]],
    validation_ids: Sequence[str],
) -> dict[str, float]:
    support = ACTIVITY_ONTOLOGY if field == "activity" else DESTINATION_ONTOLOGY
    empirical = _empirical_prior(
        support, [str(row[field]) for row in labels["train"].values()]
    )
    prior_rows: list[tuple[Mapping[str, float], str]] = []
    teacher_rows: list[tuple[Mapping[str, float], str]] = []
    for query_id in validation_ids:
        label = str(labels["validation"][query_id][field])
        belief = StructuredBelief.from_dict(base_beliefs[query_id]["belief"])
        prior_rows.append((empirical, label))
        teacher_rows.append((getattr(belief, field), label))
    prior_nll = _mean_nll(prior_rows)
    teacher_nll = _mean_nll(teacher_rows)
    return {
        "empirical_prior_nll": prior_nll,
        "teacher_nll": teacher_nll,
        "gain": prior_nll - teacher_nll,
        "validation_queries": len(validation_ids),
    }


def _m1_m2_gate(
    prepared: Mapping[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]],
    base_beliefs: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, Sequence[str]],
    region_support: Sequence[str],
    alpha_grid: Sequence[float],
) -> dict[str, Any]:
    data = DataCPT(region_support, smoothing=PILOT_SMOOTHING).fit(
        [
            (_previous_region(query), str(prepared["train"][1][query_id]["target_region"]))
            for query_id, query in prepared["train"][0].items()
        ],
        split="train",
    )
    teacher_observations = []
    positive_weight = 0.0
    for query_id in selected["train"]:
        query = prepared["train"][0][query_id]
        record = base_beliefs[query_id]
        belief = StructuredBelief.from_dict(record["belief"])
        weight = belief.confidence if bool(record.get("raw_valid")) and not bool(record.get("fallback")) else 0.0
        positive_weight += weight
        teacher_observations.append((_previous_region(query), belief.next_region, weight))
    if positive_weight <= 0.0:
        raise PilotEvaluationError("base pilot cache has no positive teacher soft-count weight")
    teacher = TeacherSoftCountCPT(region_support, smoothing=PILOT_SMOOTHING).fit(
        teacher_observations, split="train"
    )
    blended = BlendedCPT(data, teacher)
    alpha_selection_ids, heldout_ids = _validation_partition(selected["validation"])
    alpha_selection = [
        (
            _previous_region(prepared["validation"][0][query_id]),
            str(prepared["validation"][1][query_id]["target_region"]),
        )
        for query_id in alpha_selection_ids
    ]
    heldout = [
        (
            _previous_region(prepared["validation"][0][query_id]),
            str(prepared["validation"][1][query_id]["target_region"]),
        )
        for query_id in heldout_ids
    ]
    selection = blended.select_alpha(alpha_selection, split="validation", grid=alpha_grid)
    m1_nll = float(
        np.mean(
            [
                _nll(data.distribution(context), label)
                for context, label in heldout
            ]
        )
    )
    m2_nll = float(
        np.mean(
            [
                _nll(blended.distribution(context), label)
                for context, label in heldout
            ]
        )
    )
    return {
        "context": "previous observed region",
        "smoothing": PILOT_SMOOTHING,
        "train_queries": len(prepared["train"][0]),
        "teacher_train_queries": len(selected["train"]),
        "validation_queries": len(selected["validation"]),
        "validation_partition": {
            "version": _VALIDATION_PARTITION_VERSION,
            "alpha_selection_queries": len(alpha_selection_ids),
            "heldout_evaluation_queries": len(heldout_ids),
            "alpha_selection_query_hash": _canonical_hash(alpha_selection_ids),
            "heldout_evaluation_query_hash": _canonical_hash(heldout_ids),
            "disjoint": set(alpha_selection_ids).isdisjoint(heldout_ids),
        },
        "teacher_soft_count_weight": positive_weight,
        "alpha_grid": list(alpha_grid),
        "selected_alpha": selection.alpha,
        "alpha_nll": {str(alpha): value for alpha, value in selection.scores.items()},
        "alpha_selection_nll": selection.mean_nll,
        "m1_validation_nll": m1_nll,
        "m2_validation_nll": m2_nll,
        "strict_heldout_improvement": m2_nll < m1_nll,
    }


def _repeat_gate(
    base_beliefs: Mapping[str, Mapping[str, Any]],
    repeat_beliefs: Sequence[Mapping[str, Mapping[str, Any]]],
    repeat_ids: Sequence[str],
) -> dict[str, Any]:
    values: dict[str, list[float]] = {field: [] for field in _DISTRIBUTION_FIELDS}
    for query_id in repeat_ids:
        records = [base_beliefs[query_id], *(rows[query_id] for rows in repeat_beliefs)]
        beliefs = [StructuredBelief.from_dict(record["belief"]) for record in records]
        for left, right in combinations(beliefs, 2):
            for field in _DISTRIBUTION_FIELDS:
                values[field].append(_jsd(getattr(left, field), getattr(right, field)))
    combined = [value for field in _DISTRIBUTION_FIELDS for value in values[field]]
    if not combined:
        raise PilotEvaluationError("repeat caches do not provide any matched JSD comparisons")
    return {
        "metric": "Jensen-Shannon divergence (natural log)",
        "queries": len(repeat_ids),
        "cache_repetitions_per_query": 1 + len(repeat_beliefs),
        "pairwise_comparisons": len(combined),
        "field_medians": {
            field: float(statistics.median(field_values)) for field, field_values in values.items()
        },
        "median": float(statistics.median(combined)),
    }


def evaluate_pilot(
    unit_root: str | Path,
    base_cache_root: str | Path,
    repeat_cache_roots: Sequence[str | Path],
    *,
    ontology_labels_root: str | Path | None = None,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    expected_repeat_count: int = 3,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Compute and optionally persist the pilot gate from raw artifacts only.

    All paths are explicit except the optional ontology-label root.  There is
    intentionally no test-split argument and the implementation constructs
    only ``train`` and ``validation`` paths.
    """

    root = Path(unit_root)
    base_root = Path(base_cache_root)
    repeats = tuple(Path(path) for path in repeat_cache_roots)
    if len(repeats) != int(expected_repeat_count) or len({path.resolve() for path in repeats}) != len(repeats):
        raise PilotEvaluationError(
            f"expected {expected_repeat_count} distinct repeat caches, observed {len(repeats)}"
        )
    if base_root.resolve() in {path.resolve() for path in repeats}:
        raise PilotEvaluationError("base and repeat cache roots must be distinct")
    grid = _validate_alpha_grid(alpha_grid)

    prepare_path = root / "manifests" / "prepare_manifest.json"
    prepare_manifest = _read_json(prepare_path)
    if not prepare_manifest.get("dataset") or not prepare_manifest.get("unit"):
        raise PilotEvaluationError("prepare manifest is missing dataset/unit identity")
    prepared = {
        split: _prepared_split(root, prepare_manifest, split)
        for split in ("train", "validation")
    }
    support = tuple(str(value) for value in prepare_manifest.get("candidate_regions", []))
    if not support or len(support) != len(set(support)):
        raise PilotEvaluationError("prepare manifest has no unique candidate-region support")
    for split in ("train", "validation"):
        for query_id, query in prepared[split][0].items():
            if list(map(str, query.get("candidate_regions", []))) != list(support):
                raise PilotEvaluationError(f"candidate-region support mismatch for query={query_id}")
            if str(prepared[split][1][query_id].get("target_region")) not in support:
                raise PilotEvaluationError(f"target region outside frozen support for query={query_id}")

    labels, label_provenance = _ontology_labels(
        root,
        prepare_manifest,
        prepared,
        Path(ontology_labels_root) if ontology_labels_root is not None else None,
    )
    prepared_queries = {split: prepared[split][0] for split in ("train", "validation")}
    base_beliefs, base_selection, base_hashes = _cache(
        base_root,
        role="base",
        prepare_manifest=prepare_manifest,
        prepared_queries=prepared_queries,
    )

    repeat_payloads = []
    for index, cache_root in enumerate(repeats, start=1):
        beliefs, selection, hashes = _cache(
            cache_root,
            role=f"repeat-{index}",
            prepare_manifest=prepare_manifest,
            prepared_queries=prepared_queries,
        )
        for key in ("model", "model_digest", "prompt_contract_hash", "seed"):
            if selection.get(key) != base_selection.get(key):
                raise PilotEvaluationError(f"repeat-{index} cache differs from base for {key}")
        repeat_payloads.append((beliefs, selection, hashes, cache_root))

    repeat_ids = list(repeat_payloads[0][1]["selected"]["validation"])
    if not repeat_ids or any(
        list(payload[1]["selected"]["validation"]) != repeat_ids for payload in repeat_payloads[1:]
    ):
        raise PilotEvaluationError("repeat caches do not use the same validation query IDs/order")
    if any(query_id not in base_selection["selected"]["validation"] for query_id in repeat_ids):
        raise PilotEvaluationError("repeat validation queries are not a subset of the base pilot cache")
    repeat_tags = [str(payload[1].get("cache_tag", "")) for payload in repeat_payloads]
    if len(set(repeat_tags)) != len(repeat_tags):
        raise PilotEvaluationError("repeat caches must use distinct cache tags")
    for index, payload in enumerate(repeat_payloads, start=1):
        beliefs = payload[0]
        for query_id in repeat_ids:
            base_record = base_beliefs[query_id]
            repeat_record = beliefs[query_id]
            for key in (
                "source_hash",
                "prompt_hash",
                "response_format_hash",
                "model_hash",
                "dataset_hash",
                "ontology_hash",
            ):
                if repeat_record.get(key) != base_record.get(key):
                    raise PilotEvaluationError(
                        f"repeat-{index} belief differs from base for query={query_id} field={key}"
                    )

    validation_ids = list(base_selection["selected"]["validation"])
    activity = _teacher_prior_gate("activity", labels, base_beliefs, validation_ids)
    destination = _teacher_prior_gate("destination", labels, base_beliefs, validation_ids)
    m1_m2 = _m1_m2_gate(prepared, base_beliefs, base_selection["selected"], support, grid)
    repeatability = _repeat_gate(
        base_beliefs,
        [payload[0] for payload in repeat_payloads],
        repeat_ids,
    )

    gates = {
        "teacher_activity_gain": activity["gain"] > 0.0,
        "teacher_destination_gain": destination["gain"] > 0.0,
        "m2_heldout_nll_improves": m1_m2["m2_validation_nll"] < m1_m2["m1_validation_nll"],
        "repeat_jsd": repeatability["median"] <= REPEAT_JSD_LIMIT,
    }
    hashes = {
        "prepare_manifest": _file_hash(prepare_path),
        "prepared_splits": {
            split: prepared[split][2] for split in ("train", "validation")
        },
        "ontology_labels": label_provenance,
        "base_cache": base_hashes,
        "repeat_caches": [
            cache_hashes for _, _, cache_hashes, _ in repeat_payloads
        ],
        "evaluation_contract": _canonical_hash(
            {
                "version": PILOT_EVALUATION_VERSION,
                "alpha_grid": grid,
                "smoothing": PILOT_SMOOTHING,
                "repeat_jsd_limit": REPEAT_JSD_LIMIT,
                "repeat_distribution_fields": _DISTRIBUTION_FIELDS,
                "region_context": "previous observed region",
                "validation_partition_version": _VALIDATION_PARTITION_VERSION,
                "base_requested_by_split": _BASE_REQUESTED_BY_SPLIT,
                "repeat_requested_by_split": _REPEAT_REQUESTED_BY_SPLIT,
                "m2_gate": "strict held-out NLL improvement after alpha selection on a disjoint validation partition",
            }
        ),
    }
    payload: dict[str, Any] = {
        "version": PILOT_EVALUATION_VERSION,
        "dataset": prepare_manifest["dataset"],
        "unit": prepare_manifest["unit"],
        "fit_splits": ["train", "validation"],
        "data_fit_split": "train",
        "selection_split": "validation",
        "test_artifacts_read": False,
        "teacher_cache_hash": base_hashes["beliefs"],
        "teacher_activity_gain": activity["gain"],
        "teacher_destination_gain": destination["gain"],
        "m1_validation_nll": m1_m2["m1_validation_nll"],
        "m2_validation_nll": m1_m2["m2_validation_nll"],
        "repeat_jsd_median": repeatability["median"],
        "teacher_vs_empirical_prior": {
            "activity": activity,
            "destination": destination,
        },
        "m1_m2_validation": m1_m2,
        "repeatability": repeatability,
        "gates": gates,
        "hashes": hashes,
        "gate": "ready" if all(gates.values()) else "incomplete",
    }
    payload["evaluation_hash"] = _canonical_hash(payload)
    if write:
        destination = Path(output_path) if output_path is not None else root / "manifests" / "pilot_evaluation.json"
        _atomic_json(destination, payload)
    return payload


__all__ = [
    "ALPHA_GRID",
    "ONTOLOGY_LABEL_VERSION",
    "PILOT_EVALUATION_VERSION",
    "PilotEvaluationError",
    "evaluate_pilot",
]
