from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable, List

def segment_to_text(segment: Any) -> str:
    if isinstance(segment, str):
        return segment
    return json.dumps(segment, ensure_ascii=False, sort_keys=True)


class EmbeddingMemoryRetriever:
    """Structured trajectory-feature embedding with cosine retrieval.

    Encodes category, POI identity, hour and weekday into a fixed vector. Unlike
    the former character n-gram baseline, similarity is not driven by shared
    timestamp/JSON substrings.
    """

    def __init__(self, top_m: int = 5) -> None:
        self.top_m = top_m

    def retrieve(self, history: Iterable[Any], context: Iterable[Any]) -> List[str]:
        documents = [segment_to_text(item) for item in history]
        if not documents:
            return []
        query = " ".join(segment_to_text(item) for item in context) or documents[-1]
        import numpy as np
        from datetime import datetime

        def hashed(value: str, offset: int, width: int, vector) -> None:
            digest = hashlib.sha256(value.lower().encode("utf-8")).digest()
            vector[offset + int.from_bytes(digest[:2], "big") % width] += 1.0

        def vector(segment: Any):
            result = np.zeros(132, dtype=float)
            value = segment
            if isinstance(value, str):
                try: value = json.loads(value)
                except json.JSONDecodeError: value = [value]
            if isinstance(value, (list, tuple)):
                stamp = str(value[0]) if value else ""
                category = str(value[-2]) if len(value) >= 2 else ""
                poi = str(value[-1]) if value else ""
            else:
                stamp, category, poi = "", str(value), ""
            for token in re.findall(r"[\w\u4e00-\u9fff]+", category): hashed(token, 0, 64, result)
            if poi: hashed(poi, 64, 64, result); result[64:128] *= 0.5
            try:
                dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                result[128:132] = [math.sin(2*math.pi*dt.hour/24), math.cos(2*math.pi*dt.hour/24),
                                   math.sin(2*math.pi*dt.weekday()/7), math.cos(2*math.pi*dt.weekday()/7)]
            except ValueError: pass
            norm = np.linalg.norm(result)
            return result / norm if norm else result

        context_rows = list(context)
        query_vectors = [vector(item) for item in context_rows]
        query_vector = np.mean(query_vectors, axis=0) if query_vectors else vector(documents[-1])
        similarities = [float(vector(item) @ query_vector) for item in history]
        order = sorted(range(len(documents)), key=lambda index: (-similarities[index], index))[: self.top_m]
        return [documents[index] for index in order]


class FrequencyMemoryRetriever:
    """Ablation matching the lossy/counting behavior of original AgentMove."""

    def __init__(self, top_m: int = 5) -> None:
        self.top_m = top_m

    def retrieve(self, history: Iterable[Any], context: Iterable[Any]) -> List[str]:
        documents = [segment_to_text(item) for item in history]
        counts: dict[str, int] = {}
        for document in documents:
            counts[document] = counts.get(document, 0) + 1
        return [item for item, _ in sorted(counts.items(), key=lambda row: (-row[1], row[0]))[: self.top_m]]
