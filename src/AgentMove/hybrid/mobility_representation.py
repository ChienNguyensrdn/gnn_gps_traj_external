from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MobilityFeatures:
    poi_id: str
    time_slot: int
    day_of_week: int
    speed_mps: float
    heading_sin: float
    heading_cos: float
    stop_duration_seconds: float
    historical_frequency: float
    missing_mask: tuple[int, ...]


def _number(value: Any, default: float = 0.0) -> tuple[float, int]:
    try:
        number = float(value)
        if math.isfinite(number):
            return number, 0
    except (TypeError, ValueError):
        pass
    return default, 1


def encode_event(event: Mapping[str, Any], history_counts: Mapping[str, int] | None = None) -> MobilityFeatures:
    """Encode one event using only values available at that event or in its past."""
    poi = str(event.get("poi_id", event.get("POI_id", event.get("region_id", "<missing>"))))
    timestamp = event.get("timestamp", event.get("UTC_time", event.get("time")))
    missing_time = 0
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        time_slot, day = parsed.hour * 2 + parsed.minute // 30, parsed.weekday()
    except (TypeError, ValueError):
        time_slot, day, missing_time = 0, 0, 1
    speed, missing_speed = _number(event.get("speed_mps", event.get("speed")))
    heading, missing_heading = _number(event.get("heading"))
    duration, missing_duration = _number(event.get("stop_duration_seconds", event.get("duration")))
    radians = math.radians(heading % 360.0)
    counts = history_counts or {}
    total = max(1, sum(int(value) for value in counts.values()))
    frequency = float(counts.get(poi, 0)) / total
    return MobilityFeatures(
        poi_id=poi, time_slot=time_slot, day_of_week=day, speed_mps=speed,
        heading_sin=math.sin(radians), heading_cos=math.cos(radians),
        stop_duration_seconds=duration, historical_frequency=frequency,
        missing_mask=(missing_time, missing_speed, missing_heading, missing_duration),
    )


def encode_trajectory(events: Sequence[Mapping[str, Any]]) -> list[MobilityFeatures]:
    counts: dict[str, int] = {}
    encoded = []
    for event in events:
        features = encode_event(event, counts)
        encoded.append(features)
        counts[features.poi_id] = counts.get(features.poi_id, 0) + 1
    return encoded


def representation_hash(rows: Iterable[MobilityFeatures]) -> str:
    payload = [asdict(row) for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
