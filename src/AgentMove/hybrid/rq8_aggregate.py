from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .rq8_routing import POLICIES


METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "llm_call_rate",
           "latency_mean", "latency_p95", "tokens_per_query")


def summarize(values):
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def render(payload):
    lines = ["# RQ8 — Uncertainty-aware LLM routing", "",
             "> Threshold được fit trên validation; quality và chi phí được báo cáo trên test.", "",
             "| Router | R@1 | R@5 | R@10 | MRR | LLM call rate | Latency mean | Latency p95 | Tokens/query |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        row = payload["policies"][policy]; value = lambda name: row[name]["mean"]
        lines.append(f"| {policy} | {value('recall@1'):.6f} | {value('recall@5'):.6f} | "
                     f"{value('recall@10'):.6f} | {value('mrr'):.6f} | {value('llm_call_rate'):.6f} | "
                     f"{value('latency_mean'):.6f} | {value('latency_p95'):.6f} | {value('tokens_per_query'):.2f} |")
    lines += ["", "## Ghi chú", "", "- Never và Always là hai biên chi phí.",
              "- Random được budget-match với call rate của Entropy trên test và chỉ dùng làm control.",
              "- Kết quả limit hữu hạn là bounded experiment, không được gọi là full-query result.",
              "- Chưa được suy diễn thành kết quả 12-city nếu mới chạy Tokyo.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ8 routing runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); runs = []
    for seed in args.seeds:
        path = args.root / f"seed-{seed}" / "rq8.metrics.json"
        if not path.is_file(): raise FileNotFoundError(f"missing RQ8 metrics: {path}")
        runs.append(json.loads(path.read_text()))
    payload = {"rq": "RQ8", "seeds": args.seeds, "limits": [row["limit"] for row in runs],
               "selected_thresholds": [row["selected_thresholds"] for row in runs],
               "policies": {policy: {metric: summarize([row["metrics"][policy][metric] for row in runs])
                                    for metric in METRICS} for policy in POLICIES}, "gate": "ready-bounded"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload))
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": "ready-bounded"}))


if __name__ == "__main__": main()
