from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import (METRICS, bootstrap_and_permutation_many, holm_adjust,
                                load_npz, paired_differences)
from .rq7_belief_memory import VARIANTS


COMPARISONS = (("B1-history", "B0-static"), ("B2-sequential", "B0-static"),
               ("B3-dbn", "B0-static"), ("B3-dbn", "B2-sequential"))
SUMMARY_METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")


def summary(values):
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def load_runs(root: Path, seeds: list[int]):
    rows = []
    for seed in seeds:
        path = root / f"seed-{seed}" / "rq7" / "rq7.metrics.json"
        if not path.is_file(): raise FileNotFoundError(f"missing RQ7 metrics: {path}")
        row = json.loads(path.read_text());
        if row.get("evaluation_split") != "test" or row.get("fit_splits") != ["train", "validation"]:
            raise ValueError(f"invalid RQ7 split protocol: {path}")
        rows.append(row)
    return rows


def paired(root: Path, seeds: list[int], iterations: int, random_seed: int):
    rows = []
    for comparison_index, (target, reference) in enumerate(COMPARISONS):
        differences = []
        for seed in seeds:
            folder = root / f"seed-{seed}" / "rq7"
            left = load_npz(folder / f"{target}.test.predictions.npz")
            right = load_npz(folder / f"{reference}.test.predictions.npz")
            differences.append(np.column_stack([paired_differences(left, right, metric) for metric in METRICS]))
        effects, intervals, pvalues = bootstrap_and_permutation_many(differences, iterations,
                                                                      random_seed + comparison_index)
        for index, metric in enumerate(METRICS):
            rows.append({"comparison": f"{target}-vs-{reference}", "metric": metric,
                         "effect_favoring_first": float(effects[index]),
                         "bootstrap_ci95": intervals[index].tolist(), "permutation_p": float(pvalues[index])})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return rows


def render(payload):
    lines = ["# RQ7 — Belief memory", "",
             "> Báo cáo sinh từ test all-prefix. Positive paired effect nghĩa là biến thể đứng trước tốt hơn; NLL/Brier đã đảo dấu.", "",
             "## Kết quả test", "",
             "| Variant | Weight (validation) | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        data = payload["variants"][variant]; value = lambda name: data[name]["mean"]
        weights = ", ".join(f"{weight:g}" for weight in payload["selected_weights"][variant])
        lines.append(f"| {variant} | {weights} | {value('recall@1'):.6f} | {value('recall@5'):.6f} | "
                     f"{value('recall@10'):.6f} | {value('mrr'):.6f} | {value('nll'):.6f} | "
                     f"{value('brier'):.6f} | {value('ece'):.6f} |")
    lines += ["", "## Paired significance", "", "| Comparison | Metric | Effect | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_first']:.6f} | "
                     f"{low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | "
                     f"{'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Ghi chú protocol", "",
              "- Transition/prior chỉ fit trên train; weight chỉ chọn trên validation; test không dùng để tuning.",
              "- Belief reset ở biên trajectory và mỗi query chỉ dùng prefix đã quan sát.",
              "- RQ7 dùng mọi prefix nên không so trực tiếp trị tuyệt đối với RQ4/RQ6.",
              "- Gate hoàn thành yêu cầu đủ các seed được khai báo và đầy đủ per-query predictions.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ7 belief-memory experiments")
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); runs = load_runs(args.artifacts_root, args.seeds)
    payload = {"rq": "RQ7", "seeds": args.seeds,
               "selected_weights": {variant: [row["selected_weights"][variant] for row in runs] for variant in VARIANTS},
               "variants": {variant: {metric: summary([row["test_metrics"][variant][metric] for row in runs])
                                      for metric in SUMMARY_METRICS} for variant in VARIANTS},
               "paired_tests": paired(args.artifacts_root, args.seeds, args.iterations, args.random_seed),
               "gate": "ready"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload))
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": "ready"}))


if __name__ == "__main__": main()
