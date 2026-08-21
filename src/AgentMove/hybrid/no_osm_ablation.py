from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import BinaryPlattCalibrator, TemperatureScaler
from .evidence import AgentMoveLLMEvidenceExtractor, LLMServiceUnavailable
from .experiment import _populate_evidence_cache
from .io import load_queries, write_json
from .memory import EmbeddingMemoryRetriever
from .metrics import summarize
from .pipeline import HybridPipeline, PipelineConfig


def _binary(payload: dict) -> BinaryPlattCalibrator:
    result = BinaryPlattCalibrator()
    if "constant" in payload:
        result.constant = float(payload["constant"])
    else:
        result.alpha = float(payload["alpha"])
        result.beta = float(payload["beta"])
    return result


def _osm_coverage(queries) -> tuple[int, int]:
    candidates = queries[0].candidates if queries else []
    covered = sum(bool((item.address or "").strip()) for item in candidates)
    return covered, len(candidates)


def run(args) -> dict:
    validation = load_queries(args.validation)
    test = load_queries(args.test)
    covered, total = _osm_coverage(validation or test)
    coverage = covered / total if total else 0.0
    if coverage < args.min_osm_coverage:
        raise ValueError(
            f"Exact no-OSM ablation refused: source metadata has OSM address coverage "
            f"{covered}/{total} ({coverage:.1%}), below {args.min_osm_coverage:.1%}. "
            "Enrich OSM metadata and rerun the full baseline before this ablation."
        )

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    temperature = TemperatureScaler(float(calibration["temperature"]["temperature"]))
    habit, semantic = _binary(calibration["habit"]), _binary(calibration["semantic"])
    extractor = AgentMoveLLMEvidenceExtractor(
        model_name=args.model_name, platform=args.platform, batch_size=args.batch_size,
        retries=args.retries, missing_policy=args.missing_policy, world_mode="internal_only",
    )
    destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True)
    cached = _populate_evidence_cache(
        destination / "evidence_cache_no_osm.jsonl", validation + test, extractor,
        EmbeddingMemoryRetriever(args.top_m), args.top_k,
    )
    pipeline = HybridPipeline(
        cached, PipelineConfig(top_k=args.top_k, top_m=args.top_m, variant="full"),
        temperature, habit, semantic,
    )
    predictions = [pipeline.predict(query) for query in test]
    metrics = summarize(predictions)
    with (destination / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            ranking = sorted(item.probabilities, key=item.probabilities.get, reverse=True)
            handle.write(json.dumps({
                "query_id": item.query_id, "true_id": item.true_id,
                "ranking_top_k": ranking[:args.top_k],
                "true_rank": ranking.index(item.true_id) + 1,
            }, ensure_ascii=False) + "\n")
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "protocol.json", {
        "name": "w/o OSM (LLM internal knowledge only)", "world_mode": "internal_only",
        "validation": str(Path(args.validation).resolve()), "test": str(Path(args.test).resolve()),
        "calibration": str(Path(args.calibration).resolve()), "calibration_reused_from_full": True,
        "source_osm_coverage": {"covered": covered, "total": total, "ratio": coverage},
        "top_k": args.top_k, "top_m": args.top_m,
    })
    print(json.dumps(metrics, indent=2)); return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact RQ2 no-OSM ablation")
    parser.add_argument("--validation", required=True); parser.add_argument("--test", required=True)
    parser.add_argument("--calibration", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True); parser.add_argument("--platform", default="Ollama")
    parser.add_argument("--top-k", type=int, default=10); parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3); parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--missing-policy", choices=["error", "neutral"], default="neutral")
    parser.add_argument("--min-osm-coverage", type=float, default=0.9)
    args = parser.parse_args()
    try:
        run(args)
    except LLMServiceUnavailable as exc:
        print(f"LLM service unavailable: {exc}", flush=True); raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
