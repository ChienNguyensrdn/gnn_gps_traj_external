from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .regions import UNKNOWN_REGION, SpatialRegionVocabulary
from .schema import normalize_distribution


@dataclass(frozen=True)
class DecodedDestination:
    poi_id: str
    region_id: str
    poi_probabilities: Mapping[str, float]
    centroid: tuple[float, float] | None


class RegionAwareDecoder:
    """Decode region belief with a frozen lightweight POI prior.

    The fused score is ``p_prior(poi) * p_region(region(poi))**strength``.
    For faithful M1--M4 deployment this prior must be empirical and fitted on
    train.  M5 may use a lightweight prior distilled offline from the frozen
    quantitative teacher, but this decoder must not invoke that teacher online.
    Missing/unseen region assignments use the explicit unknown-region mass (or
    a small numerical floor).  The returned centroid is the expectation over
    train-fitted region centroids, falling back to the selected POI coordinate.
    """

    def __init__(
        self,
        vocabulary: SpatialRegionVocabulary,
        poi_to_region: Mapping[str, str],
        *,
        poi_coordinates: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.poi_to_region = {str(poi): str(region) for poi, region in poi_to_region.items()}
        self.poi_coordinates = {
            str(poi): (float(coordinate[0]), float(coordinate[1]))
            for poi, coordinate in (poi_coordinates or {}).items()
        }

    def decode(
        self,
        region_posterior: Mapping[str, float],
        poi_prior_probabilities: Mapping[str, float],
        *,
        region_strength: float = 1.0,
    ) -> DecodedDestination:
        strength = float(region_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("region_strength must be finite and non-negative")
        region = normalize_distribution(region_posterior, name="region_posterior")
        poi_prior = normalize_distribution(poi_prior_probabilities, name="poi_prior_probabilities")
        floor = 1e-12
        log_scores: dict[str, float] = {}
        for poi, probability in poi_prior.items():
            assigned_region = self.poi_to_region.get(poi, UNKNOWN_REGION)
            region_probability = region.get(assigned_region, region.get(UNKNOWN_REGION, floor))
            log_scores[poi] = math.log(max(probability, floor)) + strength * math.log(max(region_probability, floor))
        maximum = max(log_scores.values())
        weights = {poi: math.exp(value - maximum) for poi, value in log_scores.items()}
        poi_probabilities = normalize_distribution(weights, name="decoded_poi_probabilities")
        poi_id = min(poi_probabilities, key=lambda poi: (-poi_probabilities[poi], poi))
        region_id = self.poi_to_region.get(poi_id, UNKNOWN_REGION)
        centroid = self._posterior_centroid(region)
        if centroid is None:
            centroid = self.poi_coordinates.get(poi_id)
        return DecodedDestination(poi_id, region_id, poi_probabilities, centroid)

    def _posterior_centroid(self, region_posterior: Mapping[str, float]) -> tuple[float, float] | None:
        weighted_latitude = 0.0
        weighted_longitude = 0.0
        total = 0.0
        for region, probability in region_posterior.items():
            centroid = self.vocabulary.centroid(region)
            if centroid is None:
                continue
            weighted_latitude += probability * centroid[0]
            weighted_longitude += probability * centroid[1]
            total += probability
        if total <= 0.0:
            return None
        return weighted_latitude / total, weighted_longitude / total
