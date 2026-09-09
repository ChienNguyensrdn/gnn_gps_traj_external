from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_order_test import (METRICS, bootstrap_and_permutation_many,
                                holm_adjust, load_npz, paired_differences)


METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")
PAIRED_METRICS = tuple(metric for metric in METRICS if metric != "ece")
PAIRED_COMPARISONS = (
    ("E1-kd-vs-E0-ce", "E1-kd", "correct", "E0-ce", "correct"),
    ("E5-dual-vs-E1-kd", "E5-dual", "correct", "E1-kd", "correct"),
    ("correct-vs-reverse", "E5-dual", "correct", "E5-dual", "reverse"),
    ("correct-vs-random", "E5-dual", "correct", "E5-dual", "random"),
    ("frozen-correct-vs-reverse", "E5-dual", "frozen-correct", "E5-dual", "frozen-reverse"),
    ("frozen-correct-vs-random", "E5-dual", "frozen-correct", "E5-dual", "frozen-random"),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_path(root: Path, variant: str, order: str, seed: int) -> Path:
    return root / variant / order / f"seed-{seed}" / "test.predictions.npz"


def paired_summary(root: Path, seeds: list[int], iterations: int,
                   random_seed: int = 42) -> list[dict]:
    rows: list[dict] = []
    for comparison_index, (name, left_variant, left_order, right_variant, right_order) in enumerate(PAIRED_COMPARISONS):
        differences = []
        for seed in seeds:
            left = load_npz(prediction_path(root, left_variant, left_order, seed))
            right = load_npz(prediction_path(root, right_variant, right_order, seed))
            differences.append(np.column_stack([
                paired_differences(left, right, metric) for metric in PAIRED_METRICS
            ]))
        effects, intervals, p_values = bootstrap_and_permutation_many(
            differences, iterations, random_seed + comparison_index
        )
        for metric_index, metric in enumerate(PAIRED_METRICS):
            rows.append({
                "comparison": name,
                "metric": metric,
                "effect_favoring_first": float(effects[metric_index]),
                "bootstrap_ci95": intervals[metric_index].tolist(),
                "permutation_p": float(p_values[metric_index]),
            })
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
        row["significant_at_0.05"] = value < 0.05
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict WWW2019 cross-dataset result gate")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hybrid-metrics", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dataset-label", default="WWW2019-Shanghai-ISP")
    parser.add_argument("--report-title", default="WWW2019 — Cross-dataset validation")
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("--iterations must be at least 1000")

    variants = ("E0-ce", "E1-kd", "E5-dual")
    missing: list[str] = []
    rows: dict[str, dict] = {}
    for variant in variants:
        runs = []
        for seed in args.seeds:
            path = args.root / variant / "correct" / f"seed-{seed}" / "test.metrics.json"
            prediction = path.with_name("test.predictions.npz")
            if not path.is_file() or not prediction.is_file():
                missing.extend(str(p) for p in (path, prediction) if not p.is_file())
                continue
            payload = read(path)
            runs.append(payload.get("metrics", payload))
        if runs:
            rows[variant] = {
                metric: {"mean": float(np.mean([r[metric] for r in runs])),
                         "std": float(np.std([r[metric] for r in runs], ddof=1)) if len(runs) > 1 else None}
                for metric in METRICS if all(metric in r for r in runs)
            }

    belief = {}
    for seed in args.seeds:
        path = args.root / "E5-dual" / "correct" / f"seed-{seed}" / "rq7" / "rq7.metrics.json"
        if not path.is_file():
            missing.append(str(path)); continue
        payload = read(path).get("test_metrics", {})
        for variant in ("B0-static", "B3-dbn"):
            for metric, value in payload.get(variant, {}).items():
                if metric in METRICS:
                    belief.setdefault(variant, {}).setdefault(metric, []).append(float(value))
    belief_summary = {variant: {metric: {"mean": float(np.mean(values)),
                                                   "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                                for metric, values in metrics.items()}
                      for variant, metrics in belief.items()}

    paired_missing = sorted({
        str(prediction_path(args.root, variant, order, seed))
        for _, left_variant, left_order, right_variant, right_order in PAIRED_COMPARISONS
        for variant, order in ((left_variant, left_order), (right_variant, right_order))
        for seed in args.seeds
        if not prediction_path(args.root, variant, order, seed).is_file()
    })
    missing.extend(paired_missing)
    paired_tests = [] if paired_missing else paired_summary(
        args.root, args.seeds, args.iterations, args.random_seed
    )

    hybrid = None
    if args.hybrid_metrics:
        if args.hybrid_metrics.is_file(): hybrid = read(args.hybrid_metrics)
        else: missing.append(str(args.hybrid_metrics))
    gate = "ready-www2019" if not missing else "incomplete"
    result = {"dataset": args.dataset_label, "protocol": "cross-dataset confirmation",
              "seeds": args.seeds, "neural": rows, "belief": belief_summary,
              "llm_bounded": hybrid, "paired_tests": paired_tests,
              "missing": sorted(set(missing)), "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [f"# {args.report_title}", "", f"> Gate: **{gate}**. Dataset: {args.dataset_label}.", "",
             "## Neural last-query", "", "| Variant | R@1 | R@5 | R@10 | MRR |", "|---|---:|---:|---:|---:|"]
    def fmt(item):
        if not item: return "N/A"
        value = f"{item['mean']:.6f}"
        return value if item["std"] is None else f"{value} ± {item['std']:.6f}"
    for variant in variants:
        row = rows.get(variant, {})
        lines.append(f"| {variant} | {fmt(row.get('recall@1'))} | {fmt(row.get('recall@5'))} | {fmt(row.get('recall@10'))} | {fmt(row.get('mrr'))} |")
    lines += ["", "## Bayesian all-prefix", "", "| Variant | R@1 | R@5 | R@10 | MRR |", "|---|---:|---:|---:|---:|"]
    for variant in ("B0-static", "B3-dbn"):
        row = belief_summary.get(variant, {})
        lines.append(f"| {variant} | {fmt(row.get('recall@1'))} | {fmt(row.get('recall@5'))} | {fmt(row.get('recall@10'))} | {fmt(row.get('mrr'))} |")
    if paired_tests:
        lines += ["", "## Paired significance", "",
                  "> Positive effect nghĩa là variant đứng trước tốt hơn; NLL/Brier đã đảo dấu. Holm correction áp dụng chung.", "",
                  "| Comparison | Metric | Effect | 95% CI | Holm p | Significant |",
                  "|---|---|---:|---:|---:|---|"]
        for row in paired_tests:
            low, high = row["bootstrap_ci95"]
            lines.append(
                f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_first']:.6f} | "
                f"{low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | "
                f"{'yes' if row['significant_at_0.05'] else 'no'} |"
            )
    lines += ["", "## Giới hạn", "", f"- Kết quả chỉ áp dụng cho {args.dataset_label}, không phải thí nghiệm 12 thành phố.",
              "- Neural last-query và Bayesian all-prefix được báo cáo riêng.",
              "- LLM bounded không được gọi là full-query."]
    if missing:
        lines += ["", "## Artifact còn thiếu", ""] + [f"- `{path}`" for path in sorted(set(missing))]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "missing": len(set(missing)), "output": str(args.output)}))
    if missing and not args.allow_incomplete: raise SystemExit(2)


if __name__ == "__main__":
    main()
