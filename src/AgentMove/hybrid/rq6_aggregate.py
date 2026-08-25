from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import (METRICS, bootstrap_and_permutation_many, holm_adjust,
                                load_npz, paired_differences)


DEFAULT_VARIANTS = ("E1-kd", "E2-kd-traj", "E3-kd-vel", "E4-layer", "E6-temporal", "E5-dual")
COMPARISONS = (("E6-temporal", "E1-kd"), ("E4-layer", "E1-kd"),
               ("E5-dual", "E4-layer"), ("E5-dual", "E6-temporal"))
OVERALL_METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece",
                   "cka", "transition_cosine", "layer_transition_cosine")


def summarize(values: list[float], seed: int = 42) -> dict:
    array = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return {"mean": float(array.mean()), "std": statistics.stdev(values) if len(values) > 1 else None,
            "bootstrap_ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}


def load_runs(root: Path, variants: list[str], seeds: list[int]) -> dict:
    runs = {}
    for variant in variants:
        rows = []
        for seed in seeds:
            path = root / variant / "correct" / f"seed-{seed}" / "rq6.metrics.json"
            if not path.is_file(): raise FileNotFoundError(f"missing RQ6 metrics: {path}")
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        thresholds = [row["length_thresholds"] for row in rows]
        if any(value != thresholds[0] for value in thresholds[1:]):
            raise ValueError(f"length thresholds differ across seeds for {variant}")
        runs[variant] = rows
    return runs


def aggregate_runs(runs: dict) -> dict:
    output = {}
    for variant, rows in runs.items():
        overall = {metric: summarize([row["metrics"][metric] for row in rows]) for metric in OVERALL_METRICS}
        buckets = {}
        for bucket in ("short", "medium", "long"):
            names = ("queries", "recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")
            buckets[bucket] = {name: summarize([row["length_buckets"][bucket][name] for row in rows]) for name in names}
        output[variant] = {"length_thresholds": rows[0]["length_thresholds"], "overall": overall, "buckets": buckets}
    return output


def paired_tests(root: Path, seeds: list[int], iterations: int, random_seed: int) -> list[dict]:
    rows = []
    for comparison_index, (target, reference) in enumerate(COMPARISONS):
        loaded = []
        for seed in seeds:
            target_data = load_npz(root / target / "correct" / f"seed-{seed}" / "rq6.predictions.npz")
            reference_data = load_npz(root / reference / "correct" / f"seed-{seed}" / "rq6.predictions.npz")
            loaded.append((target_data, reference_data))
        differences = [np.column_stack([paired_differences(left, right, metric) for metric in METRICS])
                       for left, right in loaded]
        effects, intervals, p_values = bootstrap_and_permutation_many(differences, iterations, random_seed + comparison_index)
        for index, metric in enumerate(METRICS):
            rows.append({"comparison": f"{target}-vs-{reference}", "metric": metric,
                         "effect_favoring_first": float(effects[index]),
                         "bootstrap_ci95": intervals[index].tolist(), "permutation_p": float(p_values[index])})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return rows


def render(payload: dict) -> str:
    lines = ["# RQ6 — Dual-Axis Evolution", "",
             "> Sinh tự động từ checkpoint test đã đóng băng; positive paired effect nghĩa là variant đứng trước tốt hơn.", "",
             "## Overall test", "",
             "| Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ | CKA | Temporal cosine | Layer cosine |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, data in payload["variants"].items():
        value = lambda name: data["overall"][name]["mean"]
        lines.append(f"| {variant} | {value('recall@1'):.6f} | {value('recall@5'):.6f} | {value('recall@10'):.6f} | "
                     f"{value('mrr'):.6f} | {value('nll'):.6f} | {value('brier'):.6f} | {value('ece'):.6f} | "
                     f"{value('cka'):.6f} | {value('transition_cosine'):.6f} | {value('layer_transition_cosine'):.6f} |")
    lines += ["", "## Theo độ dài trajectory", ""]
    for bucket in ("short", "medium", "long"):
        lines += [f"### {bucket}", "", "| Variant | Queries | R@1 | R@5 | R@10 | MRR |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant, data in payload["variants"].items():
            values = data["buckets"][bucket]
            lines.append(f"| {variant} | {values['queries']['mean']:.0f} | {values['recall@1']['mean']:.6f} | "
                         f"{values['recall@5']['mean']:.6f} | {values['recall@10']['mean']:.6f} | {values['mrr']['mean']:.6f} |")
        lines.append("")
    lines += ["## Paired significance", "", "| Comparison | Metric | Effect | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        ci = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_first']:.6f} | "
                     f"{ci[0]:.6f}–{ci[1]:.6f} | {row['holm_adjusted_p']:.6g} | "
                     f"{'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "Ngưỡng short/medium/long được fit bằng tertile trên validation rồi khóa trước khi áp dụng test.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RQ6 dual-axis experiments")
    parser.add_argument("--artifacts-root", type=Path, required=True); parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]); parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True); args = parser.parse_args()
    runs = load_runs(args.artifacts_root, args.variants, args.seeds)
    payload = {"seeds": args.seeds, "variants": aggregate_runs(runs),
               "paired_tests": paired_tests(args.artifacts_root, args.seeds, args.iterations, args.random_seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "variants": len(args.variants)}))


if __name__ == "__main__": main()
