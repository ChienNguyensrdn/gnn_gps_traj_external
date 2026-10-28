"""Leakage-safe primitives for the faithful BeliefMove experiment.

This package deliberately contains only the small, testable mechanisms that
are shared by the planned experiments.  It performs no dataset I/O and makes
no LLM calls; orchestration code must keep train/validation/test provenance
explicit when using these primitives.
"""

from .cpt import AlphaSelection, BlendedCPT, DataCPT, TeacherSoftCountCPT
from .decoder import DecodedDestination, RegionAwareDecoder
from .ontology import (
    ACTIVITY_ONTOLOGY,
    DESTINATION_ONTOLOGY,
    INTENT_ONTOLOGY,
    activity_to_destination,
    category_to_activity,
    category_to_destination,
    derive_intent,
    ontology_contract,
    ontology_hash,
)
from .pilot import ALPHA_GRID, PilotEvaluationError, evaluate_pilot
from .regions import (
    AnonymousCellRegionVocabulary,
    MetricSpatialRegionVocabulary,
    SpatialRegionVocabulary,
)
from .schema import StructuredBelief, hash_text, normalize_distribution
from .sequential import SequentialBeliefState, fit_transition_matrix
from .uncertainty import UncertaintyScores, predictive_entropy, top2_margin, uncertainty_scores

__all__ = [
    "ACTIVITY_ONTOLOGY",
    "ALPHA_GRID",
    "DESTINATION_ONTOLOGY",
    "INTENT_ONTOLOGY",
    "AlphaSelection",
    "AnonymousCellRegionVocabulary",
    "BlendedCPT",
    "DataCPT",
    "DecodedDestination",
    "MetricSpatialRegionVocabulary",
    "PilotEvaluationError",
    "RegionAwareDecoder",
    "SequentialBeliefState",
    "SpatialRegionVocabulary",
    "StructuredBelief",
    "TeacherSoftCountCPT",
    "UncertaintyScores",
    "activity_to_destination",
    "category_to_activity",
    "category_to_destination",
    "derive_intent",
    "evaluate_pilot",
    "fit_transition_matrix",
    "hash_text",
    "normalize_distribution",
    "ontology_contract",
    "ontology_hash",
    "predictive_entropy",
    "top2_margin",
    "uncertainty_scores",
]
