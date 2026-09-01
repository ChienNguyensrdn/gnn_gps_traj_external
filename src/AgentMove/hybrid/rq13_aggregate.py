from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import METRICS, bootstrap_and_permutation_many, holm_adjust, load_npz, paired_differences
from .rq13_robustness import VARIANTS

SUMMARY = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")


def summarize(values):
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def aggregate(root: Path, seeds: list[int], iterations: int, random_seed: int):
    variants = {}; tests = []
    for variant in VARIANTS:
        rows = []
        for seed in seeds:
            path = root / variant / f"seed-{seed}" / "rq13.metrics.json"
            if not path.is_file(): raise FileNotFoundError(f"missing RQ13 metrics: {path}")
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        variants[variant] = {"metrics": {name: summarize([row["metrics"][name] for row in rows]) for name in SUMMARY},
                             "changed_query_rate": summarize([row["changed_query_rate"] for row in rows])}
    for comparison_index, variant in enumerate(VARIANTS[1:]):
        differences = []
        for seed in seeds:
            clean = load_npz(root / "clean" / f"seed-{seed}" / "test.predictions.npz")
            corrupt = load_npz(root / variant / f"seed-{seed}" / "test.predictions.npz")
            differences.append(np.column_stack([paired_differences(clean, corrupt, metric) for metric in METRICS]))
        effects, intervals, p_values = bootstrap_and_permutation_many(differences, iterations, random_seed + comparison_index)
        for index, metric in enumerate(METRICS):
            tests.append({"comparison": f"clean-vs-{variant}", "metric": metric,
                          "effect_favoring_clean": float(effects[index]), "bootstrap_ci95": intervals[index].tolist(),
                          "permutation_p": float(p_values[index])})
    adjusted = holm_adjust([row["permutation_p"] for row in tests])
    for row, value in zip(tests, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return {"rq": "RQ13", "seeds": seeds, "variants": variants, "paired_tests": tests,
            "gate": "ready-tokyo-frozen-e5-last-query"}


def render(payload):
    lines = ["# RQ13 — Robustness với đầu vào thiếu hoặc nhiễu", "",
             "> Frozen E5-dual được đánh giá trên cùng test query; chỉ context quan sát bị perturb, target/label không đổi.", "",
             "## Kết quả test", "",
             "| Variant | Changed queries | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in payload["variants"].items():
        get = lambda name: row["metrics"][name]["mean"]
        lines.append(f"| {variant} | {row['changed_query_rate']['mean']:.6f} | {get('recall@1'):.6f} | {get('recall@5'):.6f} | "
                     f"{get('recall@10'):.6f} | {get('mrr'):.6f} | {get('nll'):.6f} | {get('brier'):.6f} | {get('ece'):.6f} |")
    lines += ["", "## Paired significance: clean so với perturbation", "",
              "| Comparison | Metric | Effect favoring clean | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        ci = row["bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_clean']:.6f} | "
                     f"{ci[0]:.6f}–{ci[1]:.6f} | {row['holm_adjusted_p']:.6g} | {'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Định nghĩa perturbation", "",
              "- `gps-drop-25/50`: bỏ ngẫu nhiên 25%/50% điểm context, luôn giữ điểm quan sát cuối để query còn hợp lệ.",
              "- `time-noise-30m/60m`: thêm nhiễu rời rạc đối xứng vào time slot của prefix và target time; target POI không đổi.",
              "- `position-noise-200m/500m`: thêm Gaussian noise vào tọa độ POI context rồi ánh xạ về POI gần nhất trong test candidate coordinates.",
              "- `context-missing`: dùng unknown-user embedding và time slot 0 theo missing convention của model hiện tại.",
              "- `context-wrong`: thay user bằng user kế tiếp và dịch target time 12 giờ; deterministic theo seed/query order.", "",
              "## Protocol gate và giới hạn", "",
              "- Checkpoint được đóng băng; không fine-tune hay chọn perturbation theo test.",
              "- Paired test dùng cùng query/seed; positive effect nghĩa là clean tốt hơn, NLL/Brier đã đảo dấu.",
              "- Position noise đo robustness sau nearest-POI remapping, không phải robustness của raw-coordinate encoder.",
              "- Missing-context dùng quy ước sentinel tương thích kiến trúc hiện tại; mô hình chưa có learned missing-time token riêng.",
              "- Kết quả hiện chỉ áp dụng cho Tokyo, E5-dual và seed được khai báo; chưa phải 12-city.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ13 robustness runs")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); payload = aggregate(args.root, args.seeds, args.iterations, args.random_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": payload["gate"]}))


if __name__ == "__main__": main()
