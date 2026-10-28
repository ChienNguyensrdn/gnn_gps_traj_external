from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "beliefmove.structured-belief.v1"
SCHEMA_FIELDS = (
    "schema_version",
    "schema_hash",
    "query_hash",
    "prompt_hash",
    "model_hash",
    "activity",
    "intent",
    "destination",
    "next_region",
    "confidence",
    "belief_hash",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_NORMALIZATION_TOLERANCE = 1e-8


def hash_text(value: str) -> str:
    """Return a stable SHA-256 digest for provenance fields."""

    if not isinstance(value, str):
        raise TypeError("hash input must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema_hash() -> str:
    contract = {
        "version": SCHEMA_VERSION,
        "distributions": ["activity", "intent", "destination", "next_region"],
        "confidence": "finite float in [0,1]",
        "provenance": ["query_hash", "prompt_hash", "model_hash", "belief_hash"],
    }
    return hash_text(json.dumps(contract, sort_keys=True, separators=(",", ":")))


SCHEMA_HASH = _schema_hash()


def normalize_distribution(values: Mapping[str, float], *, name: str = "distribution") -> dict[str, float]:
    """Normalize non-negative weights into a deterministic string-keyed map.

    This helper is intended for construction at a trusted boundary.  Parsed
    artifacts are *not* silently repaired: :meth:`StructuredBelief.from_dict`
    requires every distribution to already be normalized.
    """

    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    total = 0.0
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        if not key:
            raise ValueError(f"{name} keys must be non-empty")
        if isinstance(raw_value, bool):
            raise ValueError(f"{name} values must be numeric probabilities")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} values must be finite and non-negative")
        normalized[key] = value
        total += value
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return {key: normalized[key] / total for key in sorted(normalized)}


def _strict_distribution(values: Any, name: str) -> dict[str, float]:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{name} must be a non-empty JSON object")
    result: dict[str, float] = {}
    for key, raw_value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{name} values must be JSON numbers")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} values must be finite and non-negative")
        result[key] = value
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=_NORMALIZATION_TOLERANCE):
        raise ValueError(f"{name} must sum to one")
    return {key: result[key] for key in sorted(result)}


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class StructuredBelief:
    """Strict normalized LLM belief with complete content provenance.

    ``activity``, ``intent`` and ``destination`` use fixed experiment
    ontologies supplied by the caller. ``next_region`` is a distribution over
    a frozen support derived from the train-fitted spatial vocabulary.
    """

    schema_version: str
    schema_hash: str
    query_hash: str
    prompt_hash: str
    model_hash: str
    activity: Mapping[str, float]
    intent: Mapping[str, float]
    destination: Mapping[str, float]
    next_region: Mapping[str, float]
    confidence: float
    belief_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.schema_hash != SCHEMA_HASH:
            raise ValueError("schema_hash does not match the BeliefMove v1 contract")
        _require_hash(self.query_hash, "query_hash")
        _require_hash(self.prompt_hash, "prompt_hash")
        _require_hash(self.model_hash, "model_hash")
        _require_hash(self.belief_hash, "belief_hash")
        activity = _strict_distribution(dict(self.activity), "activity")
        intent = _strict_distribution(dict(self.intent), "intent")
        destination = _strict_distribution(dict(self.destination), "destination")
        next_region = _strict_distribution(dict(self.next_region), "next_region")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")
        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "next_region", next_region)
        object.__setattr__(self, "confidence", confidence)
        if self.belief_hash != self._computed_belief_hash():
            raise ValueError("belief_hash does not match the normalized belief payload")

    @classmethod
    def create(
        cls,
        *,
        query: str,
        prompt: str,
        model_identity: str,
        activity: Mapping[str, float],
        intent: Mapping[str, float],
        destination: Mapping[str, float],
        next_region: Mapping[str, float],
        confidence: float,
    ) -> "StructuredBelief":
        """Construct a normalized belief and compute all required hashes."""

        payload = {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": SCHEMA_HASH,
            "query_hash": hash_text(query),
            "prompt_hash": hash_text(prompt),
            "model_hash": hash_text(model_identity),
            "activity": normalize_distribution(activity, name="activity"),
            "intent": normalize_distribution(intent, name="intent"),
            "destination": normalize_distribution(destination, name="destination"),
            "next_region": normalize_distribution(next_region, name="next_region"),
            "confidence": float(confidence),
        }
        payload["belief_hash"] = cls._hash_payload(payload)
        return cls(**payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredBelief":
        """Parse an artifact without repairing extra, missing or unnormalized data."""

        if not isinstance(payload, Mapping):
            raise ValueError("structured belief must be an object")
        keys = set(payload)
        expected = set(SCHEMA_FIELDS)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(f"structured belief fields mismatch; missing={missing}, extra={extra}")
        for field in ("activity", "intent", "destination", "next_region"):
            _strict_distribution(payload[field], field)
        return cls(**{key: payload[key] for key in SCHEMA_FIELDS})

    @staticmethod
    def _hash_payload(payload: Mapping[str, Any]) -> str:
        canonical = {key: payload[key] for key in SCHEMA_FIELDS if key != "belief_hash"}
        return hash_text(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    def _computed_belief_hash(self) -> str:
        return self._hash_payload(self.to_dict(include_belief_hash=False))

    def to_dict(self, *, include_belief_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "query_hash": self.query_hash,
            "prompt_hash": self.prompt_hash,
            "model_hash": self.model_hash,
            "activity": dict(self.activity),
            "intent": dict(self.intent),
            "destination": dict(self.destination),
            "next_region": dict(self.next_region),
            "confidence": self.confidence,
        }
        if include_belief_hash:
            payload["belief_hash"] = self.belief_hash
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
