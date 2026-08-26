from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_order_test import bootstrap_and_permutation_many, holm_adjust


VARIANTS = ("memory-true", "memory-shuffled", "memory-random-user", "memory-none",
            "context-shuffled", "context-random-poi", "context-none")
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def values(rows, metric):
    ranks = np.asarray([row["true_rank"] for row in rows])
    if metric.startswith("recall@"): return (ranks <= int(metric.split("@")[1])).astype(float)
    return 1.0 / ranks


def summarize(rows):
    if not rows: return {"queries": 0, "recall@1": None, "recall@5": None, "recall@10": None, "mrr": None,
                         "invalid_output_rate": None, "tokens_per_query": None, "latency_mean": None}
    ranks = np.asarray([row["true_rank"] for row in rows])
    return {"queries": len(rows), "recall@1": float(np.mean(ranks <= 1)), "recall@5": float(np.mean(ranks <= 5)),
            "recall@10": float(np.mean(ranks <= 10)), "mrr": float(np.mean(1.0 / ranks)),
            "invalid_output_rate": float(np.mean([not row["valid"] for row in rows])),
            "tokens_per_query": float(np.mean([row["input_tokens"] + row["output_tokens"] for row in rows])),
            "latency_mean": float(np.mean([row["latency_seconds"] for row in rows]))}


def ranking_sensitivity(reference, corrupted):
    top1_changed, exact_changed, correlations = [], [], []
    for left, right in zip(reference, corrupted):
        left_rank = left["ranking_top_k"]; right_rank = right["ranking_top_k"]
        top1_changed.append(left_rank[0] != right_rank[0]); exact_changed.append(left_rank != right_rank)
        shared = [value for value in left_rank if value in set(right_rank)]
        if len(shared) > 1:
            x = np.asarray([left_rank.index(value) for value in shared], dtype=float)
            y = np.asarray([right_rank.index(value) for value in shared], dtype=float)
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return {"top1_change_rate": float(np.mean(top1_changed)), "ranking_change_rate": float(np.mean(exact_changed)),
            "mean_spearman_topk": float(np.mean(correlations)) if correlations else None}


def render(payload):
    lines = ["# RQ9 — Semantic knowledge verification", "",
             "> One-axis corruption: memory variants giữ context=true; context variants giữ memory=true.", "",
             "| Variant | R@1 | R@5 | R@10 | MRR | Invalid rate | Tokens/query | Latency mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        row = payload["variants"][variant]
        lines.append(f"| {variant} | {row['recall@1']:.6f} | {row['recall@5']:.6f} | {row['recall@10']:.6f} | "
                     f"{row['mrr']:.6f} | {row['invalid_output_rate']:.6f} | {row['tokens_per_query']:.2f} | {row['latency_mean']:.6f} |")
    lines += ["", "## Paired significance: true vs corruption", "",
              "| Comparison | Metric | Effect favoring true | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_true']:.6f} | "
                     f"{low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | "
                     f"{'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Jointly-valid paired analysis", "",
              "| Comparison | Jointly valid | Metric | Effect favoring true | 95% CI | Holm p | Significant |",
              "|---|---:|---|---:|---:|---:|---|"]
    for row in payload["jointly_valid_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['jointly_valid_queries']} | {row['metric']} | "
                     f"{row['effect_favoring_true']:.6f} | {low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | "
                     f"{'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Invalid-output confound", "",
              "| Comparison | Invalid-rate effect favoring true | 95% CI | Holm p | Significant |",
              "|---|---:|---:|---:|---|"]
    for row in payload["invalid_rate_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['effect_favoring_true']:.6f} | {low:.6f}–{high:.6f} | "
                     f"{row['holm_adjusted_p']:.6g} | {'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Ranking sensitivity", "", "| Variant | Top-1 change rate | Ranking change rate | Mean Spearman top-k |",
              "|---|---:|---:|---:|"]
    for variant, row in payload["ranking_sensitivity"].items():
        correlation = "N/A" if row["mean_spearman_topk"] is None else f"{row['mean_spearman_topk']:.6f}"
        lines.append(f"| {variant} | {row['top1_change_rate']:.6f} | {row['ranking_change_rate']:.6f} | {correlation} |")
    lines += ["", "## Protocol gate", "", "- Corruption deterministic và không dùng test để tuning.",
              "- Kết quả limit hữu hạn là bounded experiment; chưa phải full-query hay 12-city result.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ9 semantic corruption experiments")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True); args = parser.parse_args()
    loaded = {variant: load_rows(args.root / variant / "predictions.jsonl") for variant in VARIANTS}
    reference = loaded["memory-true"]; query_ids = [row["query_id"] for row in reference]
    tests = []; jointly_valid_tests = []; invalid_tests = []; sensitivities = {}
    for comparison_index, variant in enumerate(VARIANTS[1:]):
        rows = loaded[variant]
        if [row["query_id"] for row in rows] != query_ids: raise ValueError(f"unaligned RQ9 queries: {variant}")
        differences = np.column_stack([values(reference, metric) - values(rows, metric) for metric in METRICS])
        effects, intervals, pvalues = bootstrap_and_permutation_many([differences], args.iterations, args.seed + comparison_index)
        for index, metric in enumerate(METRICS):
            tests.append({"comparison": f"memory-true-vs-{variant}", "metric": metric,
                          "effect_favoring_true": float(effects[index]), "bootstrap_ci95": intervals[index].tolist(),
                          "permutation_p": float(pvalues[index])})
        valid_mask = np.asarray([left["valid"] and right["valid"] for left, right in zip(reference, rows)])
        jointly_left = [row for row, keep in zip(reference, valid_mask) if keep]
        jointly_right = [row for row, keep in zip(rows, valid_mask) if keep]
        if jointly_left:
            valid_differences = np.column_stack([values(jointly_left, metric) - values(jointly_right, metric) for metric in METRICS])
            effects, intervals, pvalues = bootstrap_and_permutation_many(
                [valid_differences], args.iterations, args.seed + 100 + comparison_index)
            for index, metric in enumerate(METRICS):
                jointly_valid_tests.append({"comparison": f"memory-true-vs-{variant}", "jointly_valid_queries": len(jointly_left),
                    "metric": metric, "effect_favoring_true": float(effects[index]),
                    "bootstrap_ci95": intervals[index].tolist(), "permutation_p": float(pvalues[index])})
        # Positive means true has a lower invalid rate than corruption.
        invalid_difference = np.asarray([float(not right["valid"]) - float(not left["valid"])
                                         for left, right in zip(reference, rows)])[:, None]
        effects, intervals, pvalues = bootstrap_and_permutation_many(
            [invalid_difference], args.iterations, args.seed + 200 + comparison_index)
        invalid_tests.append({"comparison": f"memory-true-vs-{variant}", "effect_favoring_true": float(effects[0]),
                              "bootstrap_ci95": intervals[0].tolist(), "permutation_p": float(pvalues[0])})
        sensitivities[variant] = ranking_sensitivity(reference, rows)
    adjusted = holm_adjust([row["permutation_p"] for row in tests])
    for row, value in zip(tests, adjusted): row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    adjusted = holm_adjust([row["permutation_p"] for row in jointly_valid_tests])
    for row, value in zip(jointly_valid_tests, adjusted): row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    adjusted = holm_adjust([row["permutation_p"] for row in invalid_tests])
    for row, value in zip(invalid_tests, adjusted): row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    payload = {"rq": "RQ9", "reference": "memory-true", "variants": {key: summarize(rows) for key, rows in loaded.items()},
               "validity_strata": {key: {"valid": summarize([row for row in rows if row["valid"]]),
                                         "fallback": summarize([row for row in rows if not row["valid"]])}
                                    for key, rows in loaded.items()},
               "paired_tests": tests, "jointly_valid_tests": jointly_valid_tests,
               "invalid_rate_tests": invalid_tests, "ranking_sensitivity": sensitivities, "gate": "ready-bounded"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload))
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": "ready-bounded"}))


if __name__ == "__main__": main()
