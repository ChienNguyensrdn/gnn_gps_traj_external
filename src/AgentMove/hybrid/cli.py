from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import AgentMoveLLMEvidenceExtractor, HeuristicEvidenceExtractor, LLMServiceUnavailable
from .experiment import ABLATIONS, run_rq_experiments
from .io import load_queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid trajectory experiments for RQ1--RQ4")
    parser.add_argument("--validation", required=True, help="Validation queries in hybrid JSONL schema")
    parser.add_argument("--test", required=True, help="Test queries in hybrid JSONL schema")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--extractor", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-name")
    parser.add_argument("--platform")
    parser.add_argument("--llm-batch-size", type=int, default=3,
                        help="Candidates per LLM request; use 1 for weak/local models")
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-missing-policy", choices=["neutral", "error"], default="neutral")
    parser.add_argument("--compact-evidence", action="store_true")
    parser.add_argument("--llm-world-mode", choices=["full", "internal_only"], default="full")
    parser.add_argument("--validation-limit", type=int,
                        help="Process only the first N validation queries (smoke test)")
    parser.add_argument("--test-limit", type=int,
                        help="Process only the first N test queries (smoke test)")
    parser.add_argument("--variants", nargs="+", choices=ABLATIONS, default=ABLATIONS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        # Load only the requested query subset and instantiate a small metadata
        # buffer around Stage 1's top-k. Full logits stay mmap-backed.
        candidate_limit = max(args.top_k * 10, args.top_k)
        validation = load_queries(args.validation, args.validation_limit, candidate_limit)
        test = load_queries(args.test, args.test_limit, candidate_limit)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.extractor == "llm":
        if not args.model_name or not args.platform:
            raise SystemExit("--model-name and --platform are required with --extractor llm")
        extractor = AgentMoveLLMEvidenceExtractor(
            args.model_name, args.platform, batch_size=args.llm_batch_size,
            retries=args.llm_retries, missing_policy=args.llm_missing_policy,
            include_rationales=not args.compact_evidence, world_mode=args.llm_world_mode,
        )
        for query in validation + test:
            query.backbone = args.model_name
    else:
        extractor = HeuristicEvidenceExtractor()
    try:
        summaries = run_rq_experiments(
            validation,
            test,
            extractor,
            args.output_dir,
            top_k=args.top_k,
            top_m=args.top_m,
            variants=args.variants,
        )
    except LLMServiceUnavailable as exc:
        print(f"LLM service unavailable: {exc}", flush=True)
        print("Evidence already written to evidence_cache.jsonl is preserved; rerun to resume.", flush=True)
        raise SystemExit(75) from exc
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
