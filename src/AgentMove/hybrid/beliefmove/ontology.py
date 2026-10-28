from __future__ import annotations

"""Frozen semantic labels used by the faithful BeliefMove protocol.

The module deliberately uses explicit deterministic rules instead of a learned
category mapper.  This keeps the ontology auditable and prevents validation or
test categories from changing its meaning.  Unknown categories always map to
``other``.
"""

import hashlib
import json
import re
import unicodedata
from typing import Iterable, Sequence

from .regions import UNKNOWN_REGION


ONTOLOGY_VERSION = "beliefmove.ontology.v1"

ACTIVITY_ONTOLOGY = (
    "residential",
    "work-education",
    "food-retail-service",
    "leisure-social",
    "transport-outdoor",
    "other",
)

DESTINATION_ONTOLOGY = (
    "home-residential",
    "work-education",
    "food-retail-service",
    "leisure-social",
    "transport-outdoor",
    "other",
)

INTENT_ONTOLOGY = (
    "same-poi",
    "within-zone",
    "return-known-poi",
    "new-poi-known-zone",
    "new-zone",
)

# First matching rule wins.  Order is part of the versioned contract.  Food is
# placed before work so a specific label such as "college cafeteria" remains a
# food visit; transport/outdoor precedes leisure so "outdoors & recreation"
# maps to the intended transport-outdoor super-category.
CATEGORY_ACTIVITY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "residential",
        (
            "residence",
            "residential",
            "home",
            "housing",
            "apartment",
            "dormitory",
            "lodging",
            "hotel",
            "neighborhood",
        ),
    ),
    (
        "food-retail-service",
        (
            "food",
            "restaurant",
            "cafe",
            "coffee",
            "bakery",
            "cafeteria",
            "shop",
            "store",
            "retail",
            "service",
            "market",
            "mall",
            "salon",
            "pharmacy",
            "bank",
        ),
    ),
    (
        "work-education",
        (
            "professional",
            "office",
            "cowork",
            "workplace",
            "factory",
            "government",
            "school",
            "college",
            "university",
            "education",
            "library",
        ),
    ),
    (
        "transport-outdoor",
        (
            "travel",
            "transport",
            "airport",
            "station",
            "terminal",
            "road",
            "bridge",
            "outdoor",
            "outdoors",
            "park",
            "beach",
            "trail",
            "harbor",
            "bus",
            "metro",
            "train",
        ),
    ),
    (
        "leisure-social",
        (
            "arts",
            "entertainment",
            "nightlife",
            "social",
            "museum",
            "cinema",
            "theater",
            "stadium",
            "recreation",
            "club",
            "event",
            "temple",
            "church",
            "bar",
        ),
    ),
)

INTENT_PRECEDENCE = (
    "same-poi",
    "within-zone",
    "return-known-poi",
    "new-poi-known-zone",
    "new-zone",
)

_DESTINATION_BY_ACTIVITY = {
    "residential": "home-residential",
    "work-education": "work-education",
    "food-retail-service": "food-retail-service",
    "leisure-social": "leisure-social",
    "transport-outdoor": "transport-outdoor",
    "other": "other",
}


def normalize_category(value: object) -> str:
    """Normalize a raw category without using a fitted vocabulary."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _contains_keyword(normalized: str, keyword: str) -> bool:
    # Both values are normalized; padding spaces makes matching token/phrase
    # boundaries explicit and avoids e.g. ``park`` matching ``parking``.
    return f" {keyword} " in f" {normalized} "


def category_to_activity(category: object) -> str:
    """Map a raw dataset category to the fixed six-class activity ontology."""

    normalized = normalize_category(category)
    if not normalized:
        return "other"
    for label, keywords in CATEGORY_ACTIVITY_RULES:
        if any(_contains_keyword(normalized, keyword) for keyword in keywords):
            return label
    return "other"


def activity_to_destination(activity: object) -> str:
    """Map an activity label to the fixed destination-type ontology."""

    return _DESTINATION_BY_ACTIVITY.get(str(activity), "other")


def category_to_destination(category: object) -> str:
    return activity_to_destination(category_to_activity(category))


def derive_intent(
    prefix_poi_ids: Sequence[object] | Iterable[object],
    prefix_region_ids: Sequence[object] | Iterable[object],
    *,
    target_poi_id: object,
    target_region_id: object,
) -> str:
    """Derive a hard intent label from causal prefix and held-out target.

    Precedence is frozen as follows: same POI; different POI in the current
    zone; previously visited POI outside the current zone; unseen POI in any
    previously visited zone; otherwise a new zone.  This target-derived label
    belongs only in the separate target artifact and must never enter an LLM
    teacher prompt.
    """

    pois = tuple(str(value) for value in prefix_poi_ids)
    regions = tuple(str(value) for value in prefix_region_ids)
    if len(pois) != len(regions):
        raise ValueError("prefix POI and region sequences must have equal length")
    target_poi = str(target_poi_id)
    target_region = str(target_region_id)
    if not pois:
        return "new-zone"
    if target_poi == pois[-1]:
        return "same-poi"
    if target_region != UNKNOWN_REGION and target_region == regions[-1]:
        return "within-zone"
    if target_poi in pois:
        return "return-known-poi"
    if target_region != UNKNOWN_REGION and target_region in regions:
        return "new-poi-known-zone"
    return "new-zone"


def ontology_contract() -> dict:
    """Return the canonical, serializable semantic contract."""

    return {
        "version": ONTOLOGY_VERSION,
        "activity": list(ACTIVITY_ONTOLOGY),
        "destination": list(DESTINATION_ONTOLOGY),
        "intent": list(INTENT_ONTOLOGY),
        "category_activity_rules": [
            {"label": label, "keywords": list(keywords)}
            for label, keywords in CATEGORY_ACTIVITY_RULES
        ],
        "destination_by_activity": dict(sorted(_DESTINATION_BY_ACTIVITY.items())),
        "intent_precedence": list(INTENT_PRECEDENCE),
        "unknown_category": "other",
        "unknown_region": UNKNOWN_REGION,
    }


def ontology_hash() -> str:
    text = json.dumps(ontology_contract(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

