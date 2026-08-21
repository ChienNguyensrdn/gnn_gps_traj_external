from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize original AgentMove JSON outputs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(args.input_dir).glob("*.json"))
        if path.name != "metrics.json"
    ]
    if not rows:
        raise SystemExit(f"No prediction JSON files found under {args.input_dir}")
    ranks, prediction_lengths, input_tokens, output_tokens, latency, calls = [], [], [], [], [], []
    for row in rows:
        prediction = row.get("prediction", [])
        if not isinstance(prediction, list):
            prediction = [prediction]
        prediction = [str(value) for value in prediction]
        prediction_lengths.append(len(prediction))
        truth = str(row.get("true"))
        rank = prediction.index(truth) + 1 if truth in prediction else None
        ranks.append(rank)
        stats = row.get("call_stats", [])
        input_tokens.append(sum(int(item.get("input_tokens", 0)) for item in stats))
        output_tokens.append(sum(int(item.get("output_tokens", 0)) for item in stats))
        latency.append(sum(float(item.get("latency_seconds", 0.0)) for item in stats))
        calls.append(len(stats))
    metrics = {
        "queries": len(rows),
        "acc@1": float(np.mean([rank == 1 for rank in ranks])),
        "acc@5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
        # A top-5 prompt cannot support Acc@10. Reporting Acc@5 again under an
        # Acc@10 heading would overstate what was actually evaluated.
        "acc@10": (
            float(np.mean([rank is not None and rank <= 10 for rank in ranks]))
            if prediction_lengths and min(prediction_lengths) >= 10 else None
        ),
        "mrr": float(np.mean([1.0 / rank if rank else 0.0 for rank in ranks])),
        "ndcg@5": float(np.mean([1.0 / math.log2(rank + 1) if rank and rank <= 5 else 0.0 for rank in ranks])),
        "input_tokens_mean": float(np.mean(input_tokens)),
        "output_tokens_mean": float(np.mean(output_tokens)),
        "api_calls_mean": float(np.mean(calls)),
        "llm_latency_mean": float(np.mean(latency)),
        "llm_latency_median": float(np.median(latency)),
        "llm_latency_p95": float(np.quantile(latency, 0.95)),
        "token_accounting": "Ollama usage when returned; len(text)//4 fallback",
        "prediction_count_min": min(prediction_lengths),
        "prediction_count_max": max(prediction_lengths),
    }
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
