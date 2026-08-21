from __future__ import annotations

import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schemas import Candidate, Evidence, Query


class LLMServiceUnavailable(RuntimeError):
    """The LLM endpoint failed before a usable response was returned."""


class EvidenceExtractor(ABC):
    @abstractmethod
    def extract(self, query: Query, candidates: List[Candidate], memory: List[str]) -> List[Evidence]:
        raise NotImplementedError


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.lower()))


class HeuristicEvidenceExtractor(EvidenceExtractor):
    """Offline deterministic extractor for tests and non-LLM ablations."""

    def extract(self, query: Query, candidates: List[Candidate], memory: List[str]) -> List[Evidence]:
        started = time.perf_counter()
        memory_tokens = _tokens(" ".join(memory))
        context_tokens = _tokens(json.dumps(query.context, ensure_ascii=False))
        results: List[Evidence] = []
        for candidate in candidates:
            candidate_text = " ".join(
                value for value in [candidate.candidate_id, candidate.category, candidate.address] if value
            )
            candidate_tokens = _tokens(candidate_text)
            habit = len(candidate_tokens & memory_tokens) / max(len(candidate_tokens), 1)
            semantic = len(candidate_tokens & context_tokens) / max(len(candidate_tokens), 1)
            if candidate.metadata.get("habit_score") is not None:
                habit = float(candidate.metadata["habit_score"])
            if candidate.metadata.get("semantic_score") is not None:
                semantic = float(candidate.metadata["semantic_score"])
            results.append(
                Evidence(
                    candidate_id=candidate.candidate_id,
                    habit_score=float(min(max(habit, 0.0), 1.0)),
                    semantic_score=float(min(max(semantic, 0.0), 1.0)),
                    habit_rationale="Deterministic overlap with retrieved personal memory.",
                    semantic_rationale="Deterministic overlap with candidate/context metadata.",
                )
            )
        elapsed = time.perf_counter() - started
        if results:
            per_candidate = elapsed / len(results)
            for result in results:
                result.latency_seconds = per_candidate
        return results


class CachedEvidenceExtractor(EvidenceExtractor):
    """Replay evidence to make ablations paired and avoid repeated API cost."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self.cache: Dict[tuple[str, str], Evidence] = {}
        for row in rows:
            query_id = str(row["query_id"])
            evidence = Evidence(**row["evidence"] if "evidence" in row else {
                key: value for key, value in row.items() if key != "query_id"
            })
            self.cache[(query_id, str(evidence.candidate_id))] = evidence

    def extract(self, query: Query, candidates: List[Candidate], memory: List[str]) -> List[Evidence]:
        missing = [c.candidate_id for c in candidates if (query.query_id, c.candidate_id) not in self.cache]
        if missing:
            raise KeyError(f"Missing cached evidence for query={query.query_id}, candidates={missing}")
        return [self.cache[(query.query_id, c.candidate_id)] for c in candidates]


class AgentMoveLLMEvidenceExtractor(EvidenceExtractor):
    """Structured evidence scoring through AgentMove's existing LLM wrapper."""

    def __init__(self, model_name: str, platform: str, batch_size: int = 3, retries: int = 2,
                 missing_policy: str = "neutral", world_mode: str = "full",
                 include_rationales: bool = True) -> None:
        from models.llm_api import LLMWrapper

        self.wrapper = LLMWrapper(model_name=model_name, platform=platform)
        self.batch_size = max(1, batch_size)
        self.retries = max(0, retries)
        if missing_policy not in {"neutral", "error"}:
            raise ValueError("missing_policy must be neutral or error")
        self.missing_policy = missing_policy
        if world_mode not in {"full", "internal_only"}:
            raise ValueError("world_mode must be full or internal_only")
        self.world_mode = world_mode
        self.include_rationales = include_rationales

    def _candidate_payload(self, candidate: Candidate) -> Dict[str, Any]:
        payload = {
            "candidate_id": candidate.candidate_id,
            "category": candidate.category,
            "latitude": candidate.latitude,
            "longitude": candidate.longitude,
        }
        if getattr(self, "world_mode", "full") == "full":
            payload["address"] = candidate.address
            payload["metadata"] = candidate.metadata
        return payload

    def _prompt(self, query: Query, candidates: List[Candidate], memory: List[str]) -> str:
        payload = [self._candidate_payload(candidate) for candidate in candidates]
        world_instruction = (
            "Use the supplied OSM/address metadata plus your internal knowledge."
            if getattr(self, "world_mode", "full") == "full" else
            "No OSM/address metadata is available. Use category, coordinates, time and internal knowledge only."
        )
        output_rule = (
            'Also include short "habit_rationale" and "semantic_rationale" fields.'
            if getattr(self, "include_rationales", True) else
            "Return scores only; do not include rationales or chain-of-thought."
        )
        return f"""You are a structured evidence extractor for next-location prediction.
Do not rank or select a winner. For every candidate independently estimate:
- habit_score: fit with retrieved personal mobility memory, from 0 to 1.
- semantic_score: fit with urban, temporal, category and distance context, from 0 to 1.
Return JSON only: {{"evidence":[{{"candidate_id":"...","habit_score":0.0,
"semantic_score":0.0}}]}}. {output_rule}
Target time: {query.target_time}
Recent context: {json.dumps(query.context, ensure_ascii=False)}
Retrieved personal memory: {json.dumps(memory, ensure_ascii=False)}
World evidence rule: {world_instruction}
Candidates: {json.dumps(payload, ensure_ascii=False)}
"""

    @staticmethod
    def _parse(raw: str) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Local models sometimes wrap JSON in prose/Markdown and may emit
            # either the requested object or the evidence array directly.
            match = re.search(r"(?:\{.*\}|\[.*\])", raw, flags=re.DOTALL)
            if not match:
                raise ValueError("LLM response does not contain JSON")
            payload = json.loads(match.group(0))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("evidence")
        else:
            rows = None
        if not isinstance(rows, list):
            raise ValueError("LLM JSON must be an evidence list or contain an evidence list")
        return rows

    def _extract_batch(self, query: Query, candidates: List[Candidate], memory: List[str]) -> List[Evidence]:
        prompt = self._prompt(query, candidates, memory)
        started = time.perf_counter()
        try:
            raw = self.wrapper.get_response(prompt)
        except Exception as exc:
            # LLMWrapper may expose either the OpenAI/httpx exception or a
            # tenacity RetryError. Neither is a malformed evidence response:
            # treating it as neutral evidence would silently bias an entire
            # experiment when the local Ollama process is down.
            raise LLMServiceUnavailable(
                f"LLM endpoint failed for query={query.query_id}: {exc}"
            ) from exc
        latency = time.perf_counter() - started
        rows = self._parse(raw)
        expected = {candidate.candidate_id for candidate in candidates}
        evidence_by_id: Dict[str, Evidence] = {}
        for row in rows:
            if not isinstance(row, dict) or "candidate_id" not in row:
                continue
            candidate_id = str(row["candidate_id"])
            if candidate_id not in expected:
                continue
            try:
                habit = float(row["habit_score"])
                semantic = float(row["semantic_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0.0 <= habit <= 1.0 and 0.0 <= semantic <= 1.0):
                continue
            evidence_by_id[candidate_id] = Evidence(
                candidate_id=candidate_id,
                habit_score=habit,
                semantic_score=semantic,
                habit_rationale=str(row.get("habit_rationale", "")),
                semantic_rationale=str(row.get("semantic_rationale", "")),
                input_tokens=max(1, len(prompt) // 4),
                output_tokens=max(1, len(raw) // 4),
                latency_seconds=latency,
                api_calls=1,
            )
        evidence = [evidence_by_id[candidate.candidate_id] for candidate in candidates if candidate.candidate_id in evidence_by_id]
        # Query-level usage is assigned once to avoid multiplying it by k.
        for item in evidence[1:]:
            item.input_tokens = item.output_tokens = item.api_calls = 0
            item.latency_seconds = 0.0
        return evidence

    def extract(self, query: Query, candidates: List[Candidate], memory: List[str]) -> List[Evidence]:
        by_id: Dict[str, Evidence] = {}
        # Small batches are substantially more reliable for local 7B models.
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            try:
                for item in self._extract_batch(query, batch, memory):
                    by_id[item.candidate_id] = item
            except (ValueError, json.JSONDecodeError):
                pass

        # Retry only missing candidates, one at a time. This also repairs a
        # response that included only the first item of a multi-candidate batch.
        for _ in range(self.retries):
            missing = [candidate for candidate in candidates if candidate.candidate_id not in by_id]
            if not missing:
                break
            for candidate in missing:
                try:
                    for item in self._extract_batch(query, [candidate], memory):
                        by_id[item.candidate_id] = item
                except (ValueError, json.JSONDecodeError):
                    continue

        missing_ids = [candidate.candidate_id for candidate in candidates if candidate.candidate_id not in by_id]
        if missing_ids:
            if self.missing_policy == "error":
                raise ValueError(
                    f"LLM evidence incomplete after batched extraction and {self.retries} retries; "
                    f"query={query.query_id}, missing={missing_ids}"
                )
            for candidate_id in missing_ids:
                by_id[candidate_id] = Evidence(
                    candidate_id=candidate_id,
                    habit_score=0.5,
                    semantic_score=0.5,
                    habit_rationale="No valid LLM evidence; neutral Bayesian factor used.",
                    semantic_rationale="No valid LLM evidence; neutral Bayesian factor used.",
                    valid=False,
                    error=f"incomplete_after_{self.retries}_retries",
                )
        return [by_id[candidate.candidate_id] for candidate in candidates]
