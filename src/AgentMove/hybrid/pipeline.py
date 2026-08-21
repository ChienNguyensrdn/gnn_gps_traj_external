from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .calibration import BinaryPlattCalibrator, LikelihoodRatioCalibrator, TemperatureScaler, softmax
from .evidence import EvidenceExtractor
from .fusion import BayesianEvidenceFusion
from .memory import EmbeddingMemoryRetriever, FrequencyMemoryRetriever
from .schemas import Evidence, Prediction, Query


@dataclass
class PipelineConfig:
    top_k: int = 10
    top_m: int = 5
    variant: str = "full"
    # Preserve probability mass outside the LLM candidate set. Stage 2--3
    # redistribute only the original top-k mass; tail candidates retain their
    # Stage-1 probabilities, yielding a proper full-space distribution.
    renormalize_top_k: bool = False


class HybridPipeline:
    VARIANTS = {
        "full",
        "no_temperature",
        "no_world",
        "no_embedding_memory",
        "no_link_calibration",
        "no_bbn",
        "stage1_only",
        "stage1_uncalibrated",
        "top1_only",
    }

    def __init__(
        self,
        evidence_extractor: EvidenceExtractor,
        config: Optional[PipelineConfig] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        habit_calibrator: Optional[BinaryPlattCalibrator] = None,
        semantic_calibrator: Optional[BinaryPlattCalibrator] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        if self.config.variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {self.config.variant}; choose from {sorted(self.VARIANTS)}")
        self.evidence_extractor = evidence_extractor
        self.temperature_scaler = temperature_scaler or TemperatureScaler()
        self.habit_calibrator = habit_calibrator or LikelihoodRatioCalibrator()
        self.semantic_calibrator = semantic_calibrator or LikelihoodRatioCalibrator()
        self.fusion = BayesianEvidenceFusion()

    def predict(self, query: Query) -> Prediction:
        total_started = time.perf_counter()
        stage1_started = time.perf_counter()
        bundled_logits = query.metadata.get("_bundle_full_logits")
        bundled_ids = query.metadata.get("_bundle_candidate_ids")
        logits = np.asarray(
            bundled_logits if bundled_logits is not None else [candidate.logit for candidate in query.candidates],
            dtype=float,
        )
        temperature = 1.0 if self.config.variant in {"no_temperature", "stage1_uncalibrated"} else self.temperature_scaler.temperature
        full_probabilities = softmax(logits, temperature)
        requested_k = 1 if self.config.variant == "top1_only" else self.config.top_k
        k = min(requested_k, len(query.candidates))
        selected_indices = np.argsort(-full_probabilities, kind="stable")[:k]
        if bundled_ids is not None:
            available = {candidate.candidate_id: candidate for candidate in query.candidates}
            candidates = [available[str(bundled_ids[index])] for index in selected_indices]
        else:
            candidates = [query.candidates[index] for index in selected_indices]
        raw_top_k_probabilities = full_probabilities[selected_indices]
        top_k_mass = float(raw_top_k_probabilities.sum())
        # Conditional prior used for reranking inside the candidate set.
        priors = raw_top_k_probabilities / top_k_mass
        stage1_seconds = time.perf_counter() - stage1_started

        if self.config.variant in {"stage1_only", "stage1_uncalibrated"}:
            evidence: List[Evidence] = []
            posterior = priors
            contributions = {
                candidate.candidate_id: {"log_prior": float(np.log(max(prior, 1e-9)))}
                for candidate, prior in zip(candidates, priors)
            }
            memory_seconds = evidence_seconds = fusion_seconds = 0.0
        else:
            memory_started = time.perf_counter()
            retriever = (
                FrequencyMemoryRetriever(self.config.top_m)
                if self.config.variant == "no_embedding_memory"
                else EmbeddingMemoryRetriever(self.config.top_m)
            )
            memory = retriever.retrieve(query.history, query.context)
            memory_seconds = time.perf_counter() - memory_started

            evidence_started = time.perf_counter()
            evidence = self.evidence_extractor.extract(query, candidates, memory)
            by_id = {item.candidate_id: item for item in evidence}
            ordered = [by_id[candidate.candidate_id] for candidate in candidates]
            if self.config.variant == "no_world":
                semantic_scores = np.ones(len(ordered), dtype=float)
            else:
                semantic_scores = np.asarray([item.semantic_score for item in ordered], dtype=float)
            habit_scores = np.asarray([item.habit_score for item in ordered], dtype=float)
            evidence_seconds = time.perf_counter() - evidence_started

            if self.config.variant == "no_link_calibration":
                habit_likelihoods = np.clip(habit_scores, 1e-6, 1.0)
                semantic_likelihoods = np.clip(semantic_scores, 1e-6, 1.0)
            else:
                habit_likelihoods = (
                    self.habit_calibrator.predict_likelihood_ratio(habit_scores)
                    if hasattr(self.habit_calibrator, "predict_likelihood_ratio") else self.habit_calibrator.predict(habit_scores)
                )
                semantic_likelihoods = (
                    self.semantic_calibrator.predict_likelihood_ratio(semantic_scores)
                    if hasattr(self.semantic_calibrator, "predict_likelihood_ratio") else self.semantic_calibrator.predict(semantic_scores)
                )
            # Invalid/missing LLM output must contribute no evidence to the BN.
            # A likelihood factor of 1 leaves the CGM prior unchanged.
            valid_mask = np.asarray([item.valid for item in ordered], dtype=bool)
            habit_likelihoods = np.where(valid_mask, habit_likelihoods, 1.0)
            semantic_likelihoods = np.where(valid_mask, semantic_likelihoods, 1.0)
            fusion_started = time.perf_counter()
            if self.config.variant == "no_bbn":
                # Controlled raw reranking proxy. For a true free-text reranker,
                # supply its scores in evidence metadata through a custom extractor.
                raw_scores = habit_scores + semantic_scores
                posterior = softmax(raw_scores)
                contributions = {
                    candidate.candidate_id: {
                        "raw_habit_score": float(habit_scores[index]),
                        "raw_semantic_score": float(semantic_scores[index]),
                    }
                    for index, candidate in enumerate(candidates)
                }
            else:
                posterior, contributions = self.fusion.fuse(
                    [candidate.candidate_id for candidate in candidates],
                    priors,
                    habit_likelihoods,
                    semantic_likelihoods,
                )
            fusion_seconds = time.perf_counter() - fusion_started

        # Expand the selected posterior back to the complete candidate space.
        # The BBN changes relative probabilities inside top-k without claiming
        # that the true location must be in top-k.
        full_posterior = full_probabilities.copy()
        full_posterior[selected_indices] = posterior * top_k_mass
        full_posterior /= full_posterior.sum()
        # Metrics in this project are defined through rank 10. Avoid serialising
        # a 50k+ POI ranking for every TIST2015 query.
        output_k = min(max(10, self.config.top_k), len(full_posterior))
        order = np.argpartition(full_posterior, -output_k)[-output_k:]
        order = order[np.argsort(-full_posterior[order], kind="stable")]
        all_candidate_ids = bundled_ids or [candidate.candidate_id for candidate in query.candidates]
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        timings = {
            "stage1_seconds": stage1_seconds,
            "memory_seconds": memory_seconds,
            "evidence_seconds": evidence_seconds,
            "fusion_seconds": fusion_seconds,
            "total_seconds": time.perf_counter() - total_started,
        }
        return Prediction(
            query_id=query.query_id,
            user_id=query.user_id,
            city=query.city,
            true_id=query.true_id,
            ranking=[all_candidate_ids[index] for index in order],
            probabilities=[float(full_posterior[index]) for index in order],
            candidate_ids=candidate_ids,
            raw_probabilities=[float(full_probabilities[index]) for index in order],
            evidence=evidence,
            log_contributions=contributions,
            timings=timings,
            input_tokens=sum(item.input_tokens for item in evidence),
            output_tokens=sum(item.output_tokens for item in evidence),
            api_calls=sum(item.api_calls for item in evidence),
            variant=self.config.variant,
            backbone=query.backbone,
        )
