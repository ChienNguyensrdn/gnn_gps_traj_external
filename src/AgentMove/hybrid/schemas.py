from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    candidate_id: str
    logit: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    category: Optional[str] = None
    address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    query_id: str
    user_id: str
    city: str
    true_id: str
    candidates: List[Candidate]
    history: List[Any] = field(default_factory=list)
    context: List[Any] = field(default_factory=list)
    target_time: Optional[str] = None
    backbone: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Query":
        candidates = [c if isinstance(c, Candidate) else Candidate(**c) for c in value["candidates"]]
        payload = dict(value)
        payload["candidates"] = candidates
        payload["query_id"] = str(payload["query_id"])
        payload["user_id"] = str(payload["user_id"])
        payload["true_id"] = str(payload["true_id"])
        return cls(**payload)


@dataclass
class Evidence:
    candidate_id: str
    habit_score: float
    semantic_score: float
    habit_rationale: str = ""
    semantic_rationale: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    api_calls: int = 0
    valid: bool = True
    error: str = ""


@dataclass
class Prediction:
    query_id: str
    user_id: str
    city: str
    true_id: str
    ranking: List[str]
    probabilities: List[float]
    candidate_ids: List[str]
    raw_probabilities: List[float]
    evidence: List[Evidence]
    log_contributions: Dict[str, Dict[str, float]]
    timings: Dict[str, float]
    input_tokens: int
    output_tokens: int
    api_calls: int
    variant: str
    backbone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
