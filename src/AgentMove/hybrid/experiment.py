from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from typing import Dict, Iterable, List, Tuple

from .calibration import BinaryPlattCalibrator, LikelihoodRatioCalibrator, TemperatureScaler
from .bayesian_network import TrajectoryBayesianNetwork
from .evidence import CachedEvidenceExtractor, EvidenceExtractor
from .io import write_json, write_predictions
from .metrics import grouped_summary, paired_bootstrap_delta, summarize
from .memory import EmbeddingMemoryRetriever, FrequencyMemoryRetriever
from .pipeline import HybridPipeline, PipelineConfig
from .schemas import Prediction, Query


ABLATIONS = [
    "full",
    "no_temperature",
    "no_world",
    "no_embedding_memory",
    "no_bbn",
    "no_link_calibration",
    "stage1_only",
    "stage1_uncalibrated",
    "top1_only",
]


def _populate_evidence_cache(
    path: Path, queries: List[Query], extractor: EvidenceExtractor, retriever, top_k: int
) -> CachedEvidenceExtractor:
    rows, keys = [], set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line); rows.append(row)
                keys.add((str(row["query_id"]), str(row["evidence"]["candidate_id"])))
    with path.open("a", encoding="utf-8") as cache_handle:
        for query in queries:
            ranked = sorted(query.candidates, key=lambda item: item.logit, reverse=True)[:top_k]
            missing = [candidate for candidate in ranked if (query.query_id, candidate.candidate_id) not in keys]
            if not missing:
                continue
            extracted = extractor.extract(query, missing, retriever.retrieve(query.history, query.context))
            for evidence in extracted:
                row = {"query_id": query.query_id, "evidence": asdict(evidence)}
                rows.append(row); keys.add((query.query_id, evidence.candidate_id))
                cache_handle.write(json.dumps(row, ensure_ascii=False) + "\n"); cache_handle.flush()
    return CachedEvidenceExtractor(rows)


def fit_calibrators(
    validation: Iterable[Query], extractor: EvidenceExtractor, top_k: int, top_m: int
) -> Tuple[TemperatureScaler, BinaryPlattCalibrator, BinaryPlattCalibrator]:
    queries = list(validation)
    logits, labels = [], []
    for query in queries:
        full_logits = query.metadata.get("_bundle_full_logits")
        true_index = int(query.metadata.get("_bundle_true_index", -1))
        if full_logits is not None:
            if true_index < 0:
                continue
            logits.append(full_logits)
            labels.append(true_index)
            continue
        candidate_ids = [candidate.candidate_id for candidate in query.candidates]
        if query.true_id not in candidate_ids:
            continue
        logits.append([candidate.logit for candidate in query.candidates])
        labels.append(candidate_ids.index(query.true_id))
    if not logits:
        raise ValueError("Validation data has no ground-truth candidate for temperature fitting")
    # Temperature fitting requires a shared class dimension.
    dimensions = {len(row) for row in logits}
    if len(dimensions) != 1:
        raise ValueError("All validation logits must share the same candidate dimension")
    temperature = TemperatureScaler().fit(logits, labels)

    habit_scores, semantic_scores, candidate_labels = [], [], []
    retriever = EmbeddingMemoryRetriever(top_m)
    for query in queries:
        ranked = sorted(query.candidates, key=lambda item: item.logit, reverse=True)[:top_k]
        evidence = extractor.extract(query, ranked, retriever.retrieve(query.history, query.context))
        by_id = {item.candidate_id: item for item in evidence}
        for candidate in ranked:
            item = by_id[candidate.candidate_id]
            if not item.valid:
                continue
            habit_scores.append(item.habit_score)
            semantic_scores.append(item.semantic_score)
            candidate_labels.append(int(candidate.candidate_id == query.true_id))
    return (
        temperature,
        LikelihoodRatioCalibrator().fit(habit_scores, candidate_labels),
        LikelihoodRatioCalibrator().fit(semantic_scores, candidate_labels),
    )


def run_rq_experiments(
    validation: Iterable[Query],
    test: Iterable[Query],
    extractor: EvidenceExtractor,
    output_dir: str | Path,
    top_k: int = 10,
    top_m: int = 5,
    variants: Iterable[str] = ABLATIONS,
) -> Dict[str, Dict[str, float]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "bbn_structure.json", TrajectoryBayesianNetwork(["candidate_i"]).structure())
    validation_rows = list(validation)
    test_rows = list(test)
    variants = list(variants)
    all_queries = validation_rows + test_rows
    cached_extractor = _populate_evidence_cache(
        destination / "evidence_cache.jsonl", all_queries, extractor, EmbeddingMemoryRetriever(top_m), top_k
    )
    variant_extractors = {"no_embedding_memory": cached_extractor}
    if "no_embedding_memory" in variants:
        variant_extractors["no_embedding_memory"] = _populate_evidence_cache(
            destination / "evidence_cache_no_embedding_memory.jsonl", all_queries, extractor,
            FrequencyMemoryRetriever(top_m), top_k,
        )
    temperature, habit, semantic = fit_calibrators(validation_rows, cached_extractor, top_k, top_m)
    write_json(destination / "calibration.json", {
        "temperature": temperature.to_dict(),
        "habit": habit.to_dict(),
        "semantic": semantic.to_dict(),
    })
    predictions: Dict[str, List[Prediction]] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        pipeline = HybridPipeline(
            variant_extractors.get(variant, cached_extractor),
            PipelineConfig(top_k=top_k, top_m=top_m, variant=variant),
            temperature,
            habit,
            semantic,
        )
        rows = [pipeline.predict(query) for query in test_rows]
        predictions[variant] = rows
        summaries[variant] = summarize(rows)
        write_predictions(destination / variant / "predictions.jsonl", rows)
        write_json(destination / variant / "metrics.json", summaries[variant])

    full = predictions.get("full", [])
    write_json(destination / "rq1_main_results.json", {"ours_full": summaries.get("full"), "stage1_only": summaries.get("stage1_only")})
    write_json(destination / "rq2_ablation.json", summaries)
    write_json(destination / "rq3_efficiency.json", {
        name: {key: value for key, value in metrics.items() if "token" in key or "latency" in key or key in {"acc@1", "api_calls_mean", "invalid_evidence_rate"}}
        for name, metrics in summaries.items()
    })
    write_json(destination / "rq4_calibration_and_generalization.json", {
        "overall": summaries.get("full"),
        "stage1_before_temperature": summaries.get("stage1_uncalibrated"),
        "stage1_after_temperature": summaries.get("stage1_only"),
        "stage3_final": summaries.get("full"),
        "per_city": grouped_summary(full, "city") if full else {},
        "per_backbone": grouped_summary(full, "backbone") if full else {},
        "city_acc1_variance": _variance([row["acc@1"] for row in grouped_summary(full, "city").values()]) if full else None,
    })
    if full and predictions.get("stage1_only"):
        write_json(destination / "rq1_full_vs_stage1_bootstrap.json", paired_bootstrap_delta(full, predictions["stage1_only"]))
    if full:
        write_json(destination / "rq2_paired_bootstrap.json", {
            variant: paired_bootstrap_delta(full, rows)
            for variant, rows in predictions.items() if variant != "full"
        })
    return summaries


def _variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
