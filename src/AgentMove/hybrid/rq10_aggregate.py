from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import (METRICS, bootstrap_and_permutation_many, holm_adjust,
                                load_npz, paired_differences)


STUDENTS = ("none", "gru", "transformer")
TEACHERS = ("gru", "transformer")
COMPARISONS = (("gru", "none"), ("transformer", "none"), ("transformer", "gru"))
SUMMARY_METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")


def summarize(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)),
            "std": statistics.stdev(values) if len(values) > 1 else None}


def read_metric(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing RQ10 metric artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def aggregate_group(root: Path, group: str, names: tuple[str, ...], seeds: list[int]) -> dict:
    result = {}
    for name in names:
        rows = [read_metric(root / group / name / f"seed-{seed}" / "test.metrics.json") for seed in seeds]
        result[name] = {metric: summarize([row[metric] for row in rows]) for metric in SUMMARY_METRICS}
    return result


def paired_tests(root: Path, seeds: list[int], iterations: int, random_seed: int) -> list[dict]:
    rows = []
    for comparison_index, (target, reference) in enumerate(COMPARISONS):
        differences = []
        for seed in seeds:
            left = load_npz(root / "students" / target / f"seed-{seed}" / "test.predictions.npz")
            right = load_npz(root / "students" / reference / f"seed-{seed}" / "test.predictions.npz")
            differences.append(np.column_stack([paired_differences(left, right, metric) for metric in METRICS]))
        effects, intervals, p_values = bootstrap_and_permutation_many(
            differences, iterations, random_seed + comparison_index)
        for index, metric in enumerate(METRICS):
            rows.append({"comparison": f"{target}-vs-{reference}", "metric": metric,
                         "effect_favoring_first": float(effects[index]),
                         "bootstrap_ci95": intervals[index].tolist(),
                         "permutation_p": float(p_values[index])})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return rows


def metric_table(title: str, rows: dict) -> list[str]:
    output = [f"## {title}", "", "| Kiến trúc | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in rows.items():
        value = lambda key: metrics[key]["mean"]
        output.append(f"| {name} | {value('recall@1'):.6f} | {value('recall@5'):.6f} | "
                      f"{value('recall@10'):.6f} | {value('mrr'):.6f} | {value('nll'):.6f} | "
                      f"{value('brier'):.6f} | {value('ece'):.6f} |")
    return output


def render(payload: dict) -> str:
    lines = ["# RQ10 — Độ bền theo kiến trúc teacher", "",
             "> Teacher và student dùng cùng split, candidate set và seed; mọi lựa chọn checkpoint chỉ dùng validation.", ""]
    lines += metric_table("Chất lượng teacher trên test", payload["teachers"])
    lines += [""] + metric_table("Student sau distillation trên test", payload["students"])
    lines += ["", "`none` là student chỉ học cross-entropy; `gru` và `transformer` là cùng một student GRU "
              "nhưng nhận tín hiệu distillation từ teacher tương ứng.", "", "## Paired significance", "",
              "| So sánh student | Metric | Effect | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        ci = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_first']:.6f} | "
                     f"{ci[0]:.6f}–{ci[1]:.6f} | {row['holm_adjusted_p']:.6g} | "
                     f"{'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Diễn giải protocol", "",
              "- Positive effect nghĩa là student đứng trước tốt hơn; NLL/Brier đã đảo dấu.",
              "- Kết luận về robustness chỉ áp dụng cho GRU và Transformer đã chạy đủ seed khai báo.",
              "- PMT/UniTraj chưa được đưa vào bảng cho tới khi adapter candidate space và preprocessing được xác minh.",
              "- Kết quả một thành phố không được suy diễn thành kết quả 12 thành phố.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched RQ10 teacher-architecture runs")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = {"rq": "RQ10", "seeds": args.seeds,
               "teachers": aggregate_group(args.root, "teachers", TEACHERS, args.seeds),
               "students": aggregate_group(args.root, "students", STUDENTS, args.seeds),
               "paired_tests": paired_tests(args.root, args.seeds, args.iterations, args.random_seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "seeds": args.seeds}))


if __name__ == "__main__": main()
