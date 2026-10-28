from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


UNKNOWN_REGION = "__UNK_REGION__"
EARTH_RADIUS_METERS = 6_371_008.8


def _valid_coordinate(latitude: object, longitude: object) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


class SpatialRegionVocabulary:
    """Observed grid cells and centroids fitted exclusively from train data.

    Cell identifiers use a global origin, so their assignment is stable across
    datasets and row order.  Only cells observed during ``fit(..., split="train")``
    enter the vocabulary; invalid or unseen coordinates map to ``UNKNOWN_REGION``.
    """

    unknown_region = UNKNOWN_REGION

    def __init__(self, grid_size_degrees: float, centroids: Mapping[str, Sequence[float]], counts: Mapping[str, int]):
        grid_size = float(grid_size_degrees)
        if not math.isfinite(grid_size) or grid_size <= 0.0:
            raise ValueError("grid_size_degrees must be finite and positive")
        self.grid_size_degrees = grid_size
        self._centroids = {str(key): (float(value[0]), float(value[1])) for key, value in centroids.items()}
        self._counts = {str(key): int(value) for key, value in counts.items()}
        if set(self._centroids) != set(self._counts):
            raise ValueError("centroid and count vocabularies must match")
        if UNKNOWN_REGION in self._centroids:
            raise ValueError("unknown region is reserved and cannot be fitted")

    @classmethod
    def fit(
        cls,
        points: Iterable[Sequence[object]],
        *,
        split: str,
        grid_size_degrees: float = 0.01,
    ) -> "SpatialRegionVocabulary":
        if split != "train":
            raise ValueError("spatial region vocabulary may only be fitted on split='train'")
        grid_size = float(grid_size_degrees)
        if not math.isfinite(grid_size) or grid_size <= 0.0:
            raise ValueError("grid_size_degrees must be finite and positive")
        coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for point in points:
            if not isinstance(point, Sequence) or len(point) != 2:
                raise ValueError("each point must contain exactly (latitude, longitude)")
            latitude, longitude = point
            if not _valid_coordinate(latitude, longitude):
                continue
            lat = float(latitude)
            lon = float(longitude)
            region = cls._cell_id(lat, lon, grid_size)
            coordinates[region].append((lat, lon))
        centroids = {
            region: (
                sum(point[0] for point in values) / len(values),
                sum(point[1] for point in values) / len(values),
            )
            for region, values in sorted(coordinates.items())
        }
        counts = {region: len(values) for region, values in sorted(coordinates.items())}
        return cls(grid_size, centroids, counts)

    @staticmethod
    def _cell_id(latitude: float, longitude: float, grid_size: float) -> str:
        lat_index = math.floor((latitude + 90.0) / grid_size)
        lon_index = math.floor((longitude + 180.0) / grid_size)
        return f"grid:{lat_index}:{lon_index}"

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._centroids))

    @property
    def vocabulary_hash(self) -> str:
        payload = {
            "grid_size_degrees": self.grid_size_degrees,
            "centroids": {key: self._centroids[key] for key in sorted(self._centroids)},
            "counts": {key: self._counts[key] for key in sorted(self._counts)},
            "unknown_region": UNKNOWN_REGION,
            "fit_split": "train",
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def assign(self, latitude: object, longitude: object) -> str:
        if not _valid_coordinate(latitude, longitude):
            return UNKNOWN_REGION
        region = self._cell_id(float(latitude), float(longitude), self.grid_size_degrees)
        return region if region in self._centroids else UNKNOWN_REGION

    def centroid(self, region: str) -> tuple[float, float] | None:
        value = self._centroids.get(str(region))
        return None if value is None else (value[0], value[1])

    def count(self, region: str) -> int:
        return self._counts.get(str(region), 0)

    def to_dict(self) -> dict:
        return {
            "fit_split": "train",
            "grid_size_degrees": self.grid_size_degrees,
            "unknown_region": UNKNOWN_REGION,
            "centroids": {key: list(self._centroids[key]) for key in sorted(self._centroids)},
            "counts": {key: self._counts[key] for key in sorted(self._counts)},
            "vocabulary_hash": self.vocabulary_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SpatialRegionVocabulary":
        if payload.get("fit_split") != "train" or payload.get("unknown_region") != UNKNOWN_REGION:
            raise ValueError("invalid spatial vocabulary provenance")
        vocabulary = cls(
            float(payload["grid_size_degrees"]),
            payload["centroids"],  # type: ignore[arg-type]
            payload["counts"],  # type: ignore[arg-type]
        )
        if payload.get("vocabulary_hash") != vocabulary.vocabulary_hash:
            raise ValueError("spatial vocabulary hash mismatch")
        return vocabulary


class MetricSpatialRegionVocabulary:
    """Train-fitted square spatial cells expressed in metres.

    Latitude/longitude is projected to a local equirectangular plane.  The
    latitude of projection is the mean latitude of *training* coordinates,
    making the mapping deterministic, order invariant and free of
    validation/test information.  This avoids the longitude distortion of a
    fixed ``degrees = metres / 111320`` conversion while retaining an exact
    cell width in the local metric plane.

    Only cells observed in training become valid regions.  A coordinate in a
    geometrically valid but unseen cell maps to :data:`UNKNOWN_REGION`.
    """

    unknown_region = UNKNOWN_REGION
    projection = "local-equirectangular"

    def __init__(
        self,
        cell_size_meters: float,
        reference_latitude: float,
        centroids: Mapping[str, Sequence[float]],
        counts: Mapping[str, int],
    ):
        cell_size = float(cell_size_meters)
        reference = float(reference_latitude)
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("cell_size_meters must be finite and positive")
        if not math.isfinite(reference) or not -90.0 <= reference <= 90.0:
            raise ValueError("reference_latitude must be finite and valid")
        self.cell_size_meters = cell_size
        self.reference_latitude = reference
        self._cos_reference = math.cos(math.radians(reference))
        if self._cos_reference <= 0.0:
            raise ValueError("reference latitude cannot be a geographic pole")
        self._centroids = {
            str(key): (float(value[0]), float(value[1])) for key, value in centroids.items()
        }
        self._counts = {str(key): int(value) for key, value in counts.items()}
        if set(self._centroids) != set(self._counts):
            raise ValueError("centroid and count vocabularies must match")
        if UNKNOWN_REGION in self._centroids:
            raise ValueError("unknown region is reserved and cannot be fitted")
        if any(value <= 0 for value in self._counts.values()):
            raise ValueError("fitted region counts must be positive")

    @classmethod
    def fit(
        cls,
        points: Iterable[Sequence[object]],
        *,
        split: str,
        cell_size_meters: float = 500.0,
    ) -> "MetricSpatialRegionVocabulary":
        if split != "train":
            raise ValueError("metric region vocabulary may only be fitted on split='train'")
        cell_size = float(cell_size_meters)
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("cell_size_meters must be finite and positive")
        valid: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, Sequence) or len(point) != 2:
                raise ValueError("each point must contain exactly (latitude, longitude)")
            latitude, longitude = point
            if _valid_coordinate(latitude, longitude):
                valid.append((float(latitude), float(longitude)))
        if not valid:
            raise ValueError("training points contain no valid latitude/longitude")
        reference = math.fsum(point[0] for point in valid) / len(valid)
        helper = cls(cell_size, reference, {}, {})
        coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for latitude, longitude in valid:
            coordinates[helper._cell_id(latitude, longitude)].append((latitude, longitude))
        centroids = {
            region: (
                math.fsum(point[0] for point in values) / len(values),
                math.fsum(point[1] for point in values) / len(values),
            )
            for region, values in sorted(coordinates.items())
        }
        counts = {region: len(values) for region, values in sorted(coordinates.items())}
        return cls(cell_size, reference, centroids, counts)

    def project(self, latitude: object, longitude: object) -> tuple[float, float] | None:
        """Project a WGS84 coordinate to deterministic local metric ``(x,y)``."""

        if not _valid_coordinate(latitude, longitude):
            return None
        lat = math.radians(float(latitude))
        lon = math.radians(float(longitude))
        x = EARTH_RADIUS_METERS * lon * self._cos_reference
        y = EARTH_RADIUS_METERS * lat
        return x, y

    def _cell_id(self, latitude: float, longitude: float) -> str:
        projected = self.project(latitude, longitude)
        if projected is None:  # pragma: no cover - guarded by callers
            raise ValueError("invalid coordinate")
        x, y = projected
        return f"metric:{math.floor(x / self.cell_size_meters)}:{math.floor(y / self.cell_size_meters)}"

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._centroids))

    @property
    def vocabulary_hash(self) -> str:
        payload = {
            "projection": self.projection,
            "earth_radius_meters": EARTH_RADIUS_METERS,
            "cell_size_meters": self.cell_size_meters,
            "reference_latitude": self.reference_latitude,
            "centroids": {key: self._centroids[key] for key in sorted(self._centroids)},
            "counts": {key: self._counts[key] for key in sorted(self._counts)},
            "unknown_region": UNKNOWN_REGION,
            "fit_split": "train",
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def assign(self, latitude: object, longitude: object) -> str:
        if not _valid_coordinate(latitude, longitude):
            return UNKNOWN_REGION
        region = self._cell_id(float(latitude), float(longitude))
        return region if region in self._centroids else UNKNOWN_REGION

    def centroid(self, region: str) -> tuple[float, float] | None:
        value = self._centroids.get(str(region))
        return None if value is None else (value[0], value[1])

    def count(self, region: str) -> int:
        return self._counts.get(str(region), 0)

    def to_dict(self) -> dict:
        return {
            "fit_split": "train",
            "projection": self.projection,
            "earth_radius_meters": EARTH_RADIUS_METERS,
            "cell_size_meters": self.cell_size_meters,
            "reference_latitude": self.reference_latitude,
            "unknown_region": UNKNOWN_REGION,
            "centroids": {key: list(self._centroids[key]) for key in sorted(self._centroids)},
            "counts": {key: self._counts[key] for key in sorted(self._counts)},
            "vocabulary_hash": self.vocabulary_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MetricSpatialRegionVocabulary":
        if (
            payload.get("fit_split") != "train"
            or payload.get("unknown_region") != UNKNOWN_REGION
            or payload.get("projection") != cls.projection
            or float(payload.get("earth_radius_meters", -1.0)) != EARTH_RADIUS_METERS
        ):
            raise ValueError("invalid metric spatial vocabulary provenance")
        vocabulary = cls(
            float(payload["cell_size_meters"]),
            float(payload["reference_latitude"]),
            payload["centroids"],  # type: ignore[arg-type]
            payload["counts"],  # type: ignore[arg-type]
        )
        if payload.get("vocabulary_hash") != vocabulary.vocabulary_hash:
            raise ValueError("metric spatial vocabulary hash mismatch")
        return vocabulary


class AnonymousCellRegionVocabulary:
    """Train-only identity mapping for already discretized YJMob cells."""

    unknown_region = UNKNOWN_REGION
    prefix = "cell:"

    def __init__(self, counts: Mapping[str, int]):
        self._counts = {str(key): int(value) for key, value in counts.items()}
        if UNKNOWN_REGION in self._counts or any(value <= 0 for value in self._counts.values()):
            raise ValueError("anonymous cell counts must be positive and cannot use the reserved label")

    @classmethod
    def fit(cls, cell_ids: Iterable[object], *, split: str) -> "AnonymousCellRegionVocabulary":
        if split != "train":
            raise ValueError("anonymous cell vocabulary may only be fitted on split='train'")
        counts: dict[str, int] = defaultdict(int)
        for raw_value in cell_ids:
            value = str(raw_value).strip()
            if value:
                counts[f"{cls.prefix}{value}"] += 1
        if not counts:
            raise ValueError("training split contains no valid anonymous cells")
        return cls(counts)

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._counts))

    @property
    def vocabulary_hash(self) -> str:
        payload = {
            "fit_split": "train",
            "kind": "anonymous-cell-identity",
            "unknown_region": UNKNOWN_REGION,
            "counts": {key: self._counts[key] for key in sorted(self._counts)},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def assign(self, cell_id: object) -> str:
        value = str(cell_id).strip()
        region = f"{self.prefix}{value}" if value else UNKNOWN_REGION
        return region if region in self._counts else UNKNOWN_REGION

    def count(self, region: str) -> int:
        return self._counts.get(str(region), 0)

    def to_dict(self) -> dict:
        return {
            "fit_split": "train",
            "kind": "anonymous-cell-identity",
            "unknown_region": UNKNOWN_REGION,
            "counts": {key: self._counts[key] for key in sorted(self._counts)},
            "vocabulary_hash": self.vocabulary_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AnonymousCellRegionVocabulary":
        if (
            payload.get("fit_split") != "train"
            or payload.get("kind") != "anonymous-cell-identity"
            or payload.get("unknown_region") != UNKNOWN_REGION
        ):
            raise ValueError("invalid anonymous cell vocabulary provenance")
        vocabulary = cls(payload["counts"])  # type: ignore[arg-type]
        if payload.get("vocabulary_hash") != vocabulary.vocabulary_hash:
            raise ValueError("anonymous cell vocabulary hash mismatch")
        return vocabulary
