from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import bootstrap_and_permutation_many, holm_adjust
from .rq8_routing import POLICIES

METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "llm_call_rate", "latency_mean", "latency_p95", "tokens_per_query")
PAIRED_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
COMPARISONS = (("entropy", "never"), ("entropy", "always"), ("entropy", "random-budget-matched"), ("always", "never"))


def summarize(values):
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None, "runs": len(values)}


def load_predictions(path):
    with np.load(path, allow_pickle=False) as payload: return {key: payload[key] for key in payload.files}


def query_values(payload, metric):
    ranks = payload["ranks"]
    if metric.startswith("recall@"): return (ranks <= int(metric.split("@")[1])).astype(float)
    if metric == "mrr": return 1.0 / ranks
    raise ValueError(metric)


def paired_tests(root, seeds, iterations, random_seed):
    rows = []
    for comparison_index, (target, reference) in enumerate(COMPARISONS):
        comparison_seeds = seeds if "random-budget-matched" in (target, reference) else seeds[:1]
        differences = []
        for seed in comparison_seeds:
            folder = root / f"seed-{seed}"
            left = load_predictions(folder / f"{target}.test.predictions.npz")
            right = load_predictions(folder / f"{reference}.test.predictions.npz")
            if not np.array_equal(left["query_id"], right["query_id"]): raise ValueError("unaligned RQ8 queries")
            differences.append(np.column_stack([query_values(left, metric) - query_values(right, metric) for metric in PAIRED_METRICS]))
        effects, intervals, pvalues = bootstrap_and_permutation_many(differences, iterations, random_seed + comparison_index)
        for index, metric in enumerate(PAIRED_METRICS):
            rows.append({"comparison": f"{target}-vs-{reference}", "metric": metric,
                         "effect_favoring_first": float(effects[index]), "bootstrap_ci95": intervals[index].tolist(),
                         "permutation_p": float(pvalues[index]),
                         "random_permutations": len(comparison_seeds) if "random-budget-matched" in (target, reference) else None})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted): row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return rows


def render(payload):
    lines = ["# RQ8 — Uncertainty-aware LLM routing", "",
             "> Threshold fit trên validation; deterministic policies chỉ tính một run, không pseudo-replicate theo random seed.", "",
             "## Kết quả primary budget", "",
             "| Router | Runs | R@1 | R@5 | R@10 | MRR | LLM call rate | Latency mean | Latency p95 | Tokens/query |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        row = payload["policies"][policy]; value = lambda name: row[name]["mean"]
        lines.append(f"| {policy} | {row['recall@1']['runs']} | {value('recall@1'):.6f} | {value('recall@5'):.6f} | "
                     f"{value('recall@10'):.6f} | {value('mrr'):.6f} | {value('llm_call_rate'):.6f} | "
                     f"{value('latency_mean'):.6f} | {value('latency_p95'):.6f} | {value('tokens_per_query'):.2f} |")
    lines += ["", "## Validation selection", "", "| Router | Threshold | Validation call rate |", "|---|---:|---:|"]
    for kind, row in payload["validation_selection"].items(): lines.append(f"| {kind} | {row['threshold']:.8g} | {row['call_rate']:.6f} |")
    lines += ["", "## Budget sweep trên test", "", "| Budget | Router | R@1 | R@5 | MRR | Call rate | Tokens/query |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for budget, policies in payload["budget_sweep"].items():
        for kind, row in policies.items():
            lines.append(f"| {float(budget):.2f} | {kind} | {row['recall@1']:.6f} | {row['recall@5']:.6f} | "
                         f"{row['mrr']:.6f} | {row['llm_call_rate']:.6f} | {row['tokens_per_query']:.2f} |")
    oracle = payload["oracle_upper_bound"]
    lines += ["", "## Oracle upper bound tại primary budget", "",
              f"Oracle chỉ gọi trên query có positive realized LLM gain: R@1={oracle['recall@1']:.6f}, R@5={oracle['recall@5']:.6f}, "
              f"MRR={oracle['mrr']:.6f}, call rate={oracle['llm_call_rate']:.6f}.", "",
              "## Paired significance", "", "| Comparison | Metric | Effect | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_first']:.6f} | {low:.6f}–{high:.6f} | "
                     f"{row['holm_adjusted_p']:.6g} | {'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Ghi chú", "", "- Random seeds chỉ áp dụng cho Random-budget-matched.",
              "- Kết quả limit hữu hạn là bounded experiment, không phải full-query result.",
              "- Chưa được suy diễn thành kết quả 12-city nếu mới chạy Tokyo.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ8 routing runs")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); runs = []
    for seed in args.seeds:
        path = args.root / f"seed-{seed}" / "rq8.metrics.json"
        if not path.is_file(): raise FileNotFoundError(f"missing RQ8 metrics: {path}")
        runs.append(json.loads(path.read_text()))
    for policy in ("never", "always", "entropy", "margin"):
        reference = runs[0]["metrics"][policy]
        if any(any(not np.isclose(row["metrics"][policy][metric], reference[metric]) for metric in METRICS) for row in runs[1:]):
            raise ValueError(f"deterministic policy differs across random seeds: {policy}")
    policies = {}
    for policy in POLICIES:
        selected_runs = runs if policy == "random-budget-matched" else runs[:1]
        policies[policy] = {metric: summarize([row["metrics"][policy][metric] for row in selected_runs]) for metric in METRICS}
    payload = {"rq": "RQ8", "random_seeds": args.seeds, "limit": runs[0]["limit"],
               "validation_selection": runs[0]["validation_selection"], "budget_sweep": runs[0]["budget_sweep"],
               "oracle_upper_bound": runs[0]["oracle_upper_bound"], "policies": policies,
               "paired_tests": paired_tests(args.root, args.seeds, args.iterations, args.random_seed), "gate": "ready-bounded"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload))
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": "ready-bounded"}))


if __name__ == "__main__": main()
