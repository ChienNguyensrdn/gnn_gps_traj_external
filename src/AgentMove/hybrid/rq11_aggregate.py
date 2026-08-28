from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import bootstrap_and_permutation_many, holm_adjust, load_npz, paired_differences
from .rq11_calibration import reliability

GROUPS = {"distillation": ("none", "gru", "transformer"), "bayesian": ("B0-static", "B3-dbn")}
STRATEGIES = ("identity", "nll", "brier", "ece")
METRICS = ("nll", "brier", "ece", "adaptive_ece", "accuracy_confidence_gap")
DIRECT = {"distillation": (("gru", "none"), ("transformer", "none")),
          "bayesian": (("B3-dbn", "B0-static"),)}


def summary(values):
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def load_runs(root: Path, seeds: list[int]) -> dict:
    output = {}
    for group, variants in GROUPS.items():
        output[group] = {}; expected = "last-query" if group == "distillation" else "all-prefix"
        for variant in variants:
            rows = []
            for seed in seeds:
                path = root / group / variant / f"seed-{seed}" / "rq11.metrics.json"
                if not path.is_file(): raise FileNotFoundError(f"missing RQ11 artifact: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("temperature_fit_split") != "validation" or row.get("evaluation_split") != "test":
                    raise ValueError(f"calibration split leakage gate failed: {path}")
                if row.get("protocol") != expected: raise ValueError(f"protocol mismatch: {path}")
                if set(row.get("temperatures", {})) != set(STRATEGIES) or set(row.get("metrics", {})) != set(STRATEGIES):
                    raise ValueError(f"incomplete multi-objective artifact: {path}")
                rows.append(row)
            output[group][variant] = rows
    return output


def aggregate_runs(runs: dict) -> dict:
    return {group: {variant: {
        "temperatures": {strategy: summary([row["temperatures"][strategy] for row in rows]) for strategy in STRATEGIES},
        "metrics": {strategy: {metric: summary([row["metrics"][strategy][metric] for row in rows])
                               for metric in METRICS} for strategy in STRATEGIES},
    } for variant, rows in variants.items()} for group, variants in runs.items()}


def paired_tests(root: Path, seeds: list[int], iterations: int, random_seed: int) -> list[dict]:
    specs = []
    for group, variants in GROUPS.items():
        for variant in variants:
            specs += [("calibration", group, variant, variant, "nll", "identity", "nll"),
                      ("calibration", group, variant, variant, "brier", "identity", "brier")]
        for target, reference in DIRECT[group]:
            specs += [("model", group, target, reference, "nll", "nll", "nll"),
                      ("model", group, target, reference, "brier", "brier", "brier")]
    rows = []
    for index, (kind, group, target, reference, left_strategy, right_strategy, metric) in enumerate(specs):
        differences = []
        for seed in seeds:
            left = load_npz(root / group / target / f"seed-{seed}" / f"{left_strategy}.predictions.npz")
            right = load_npz(root / group / reference / f"seed-{seed}" / f"{right_strategy}.predictions.npz")
            differences.append(paired_differences(left, right, metric)[:, None])
        effect, interval, pvalue = bootstrap_and_permutation_many(differences, iterations, random_seed + index)
        rows.append({"kind": kind, "group": group, "comparison": f"{target}-vs-{reference}",
                     "strategy": left_strategy, "metric": metric, "effect_favoring_first": float(effect[0]),
                     "bootstrap_ci95": interval[0].tolist(), "permutation_p": float(pvalue[0])})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return rows


def _ece(confidence, correct, bins):
    rows = reliability(confidence, correct, bins, False)
    return sum(row["count"] * abs(row["accuracy"] - row["confidence"]) for row in rows if row["count"]) / len(confidence)


def _weighted_ece(confidence, correct, weights, bins):
    identifiers = np.minimum((confidence * bins).astype(int), bins - 1); delta = correct.astype(float) - confidence
    numerator = np.zeros(len(weights))
    for index in range(bins):
        mask = identifiers == index
        if np.any(mask): numerator += np.abs(weights[:, mask] @ delta[mask])
    return numerator / np.maximum(weights.sum(axis=1), 1)


def ece_pair(root, group, target, reference, left_strategy, right_strategy, seeds, iterations, rng, bins):
    samples = np.zeros(iterations); observed = []
    for seed in seeds:
        left = load_npz(root / group / target / f"seed-{seed}" / f"{left_strategy}.predictions.npz")
        right = load_npz(root / group / reference / f"seed-{seed}" / f"{right_strategy}.predictions.npz")
        for key in ("query_index", "labels"):
            if not np.array_equal(left[key], right[key]): raise ValueError(f"unaligned RQ11 ECE pair: {key}")
        observed.append(_ece(right["confidence"], right["top1_correct"], bins) -
                        _ece(left["confidence"], left["top1_correct"], bins))
        for start in range(0, iterations, 32):
            size = min(32, iterations - start)
            weights = rng.poisson(1.0, size=(size, len(left["labels"]))).astype(np.float32)
            samples[start:start + size] += (_weighted_ece(right["confidence"], right["top1_correct"], weights, bins) -
                                            _weighted_ece(left["confidence"], left["top1_correct"], weights, bins)) / len(seeds)
    return float(np.mean(observed)), np.quantile(samples, [.025, .975]).tolist()


def ece_tests(root: Path, seeds: list[int], iterations: int, random_seed: int, bins: int) -> list[dict]:
    rng = np.random.default_rng(random_seed); rows = []
    for group, variants in GROUPS.items():
        for variant in variants:
            effect, interval = ece_pair(root, group, variant, variant, "ece", "identity", seeds, iterations, rng, bins)
            rows.append({"kind": "calibration", "group": group, "comparison": f"{variant}-vs-{variant}",
                         "strategy": "ece", "metric": "ece", "effect_favoring_first": effect, "bootstrap_ci95": interval})
        for target, reference in DIRECT[group]:
            effect, interval = ece_pair(root, group, target, reference, "ece", "ece", seeds, iterations, rng, bins)
            rows.append({"kind": "model", "group": group, "comparison": f"{target}-vs-{reference}",
                         "strategy": "ece", "metric": "ece", "effect_favoring_first": effect, "bootstrap_ci95": interval})
    return rows


def render_svg(root: Path, group: str, variants: tuple[str, ...], seeds: list[int], bins: int, output: Path):
    colors = ["#2563eb", "#dc2626", "#16a34a"]; width, height, margin = 680, 520, 65; lines = []
    for variant_index, variant in enumerate(variants):
        for stage_index, strategy in enumerate(("identity", "ece")):
            confidence, correct = [], []
            for seed in seeds:
                data = load_npz(root / group / variant / f"seed-{seed}" / f"{strategy}.predictions.npz")
                confidence.append(data["confidence"]); correct.append(data["top1_correct"])
            rows = reliability(np.concatenate(confidence), np.concatenate(correct), bins, False)
            points = [(margin + row["confidence"] * (width - 2 * margin), height - margin - row["accuracy"] * (height - 2 * margin))
                      for row in rows if row["count"]]
            color = colors[variant_index]; dash = "8 5" if strategy == "identity" else "none"
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points); label_y = 28 + 20 * (variant_index * 2 + stage_index)
            lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
            lines.append(f'<line x1="410" y1="{label_y}" x2="445" y2="{label_y}" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/><text x="452" y="{label_y + 4}" font-size="13">{variant} {strategy}</text>')
    plot = "\n".join(lines); diagonal = f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#64748b" stroke-dasharray="5 5"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{margin}" y="25" font-size="18" font-weight="bold">RQ11 {group}: identity vs ECE-calibrated</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="black"/>{diagonal}
{plot}<text x="{width/2-35}" y="{height-15}" font-size="14">Confidence</text><text x="15" y="{height/2}" font-size="14" transform="rotate(-90 15 {height/2})">Accuracy</text></svg>'''
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(svg, encoding="utf-8")


def render(payload: dict) -> str:
    lines = ["# RQ11 — Calibration đa mục tiêu", "", "> Identity và temperature tối ưu NLL/Brier/ECE đều được chọn trên validation; test không dùng để tuning. Hai protocol được báo cáo riêng.", ""]
    for group, variants in GROUPS.items():
        protocol = "last-query" if group == "distillation" else "all-prefix"
        lines += [f"## {group} ({protocol})", "", "| Variant | T-NLL | NLL id→opt | T-Brier | Brier id→opt | T-ECE | ECE id→opt |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for variant in variants:
            row = payload["groups"][group][variant]; m = row["metrics"]; t = row["temperatures"]
            lines.append(f"| {variant} | {t['nll']['mean']:.4f} | {m['identity']['nll']['mean']:.6f}→{m['nll']['nll']['mean']:.6f} | "
                         f"{t['brier']['mean']:.4f} | {m['identity']['brier']['mean']:.6f}→{m['brier']['brier']['mean']:.6f} | "
                         f"{t['ece']['mean']:.4f} | {m['identity']['ece']['mean']:.6f}→{m['ece']['ece']['mean']:.6f} |")
        lines += ["", "### Trade-off trên test", "", "| Variant | Objective | NLL | Brier | ECE | Adaptive ECE | Confidence gap |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for variant in variants:
            for strategy in STRATEGIES:
                values = payload["groups"][group][variant]["metrics"][strategy]
                lines.append(f"| {variant} | {strategy} | {values['nll']['mean']:.6f} | {values['brier']['mean']:.6f} | "
                             f"{values['ece']['mean']:.6f} | {values['adaptive_ece']['mean']:.6f} | {values['accuracy_confidence_gap']['mean']:.6f} |")
        lines += ["", f"Reliability diagram identity–ECE: `results/beliefmove-evo/aggregated/rq11_{group}_reliability.svg`", ""]
    lines += ["## Paired NLL/Brier tests", "", "| Loại | Protocol | Comparison | Objective | Metric | Effect | 95% CI | Holm p | Significant |",
              "|---|---|---|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['kind']} | {row['group']} | {row['comparison']} | {row['strategy']} | {row['metric']} | {row['effect_favoring_first']:.6f} | {low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | {'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Bootstrap ECE tests", "", "| Loại | Protocol | Comparison | Effect | 95% CI |", "|---|---|---|---:|---:|"]
    for row in payload["ece_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['kind']} | {row['group']} | {row['comparison']} | {row['effect_favoring_first']:.6f} | {low:.6f}–{high:.6f} |")
    lines += ["", "## Protocol gate", "", "- Temperature của từng objective chỉ fit validation; identity luôn là T=1.",
              "- Temperature scaling giữ nguyên ranking; thay đổi R@k/MRR ngoài sai số số học là lỗi protocol.",
              "- Distillation last-query và Bayesian all-prefix không được so trực tiếp trị tuyệt đối.",
              "- Transition/prior fit train; B3 weight lấy từ validation RQ7.",
              "- Kết quả hiện chỉ áp dụng cho Tokyo và các seed được khai báo, chưa phải 12-city.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-objective RQ11 calibration")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--ece-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42); parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); runs = load_runs(args.root, args.seeds)
    payload = {"rq": "RQ11", "seeds": args.seeds, "objectives": list(STRATEGIES), "groups": aggregate_runs(runs),
               "paired_tests": paired_tests(args.root, args.seeds, args.iterations, args.random_seed),
               "ece_tests": ece_tests(args.root, args.seeds, args.ece_iterations, args.random_seed + 100, args.bins), "gate": "ready"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    for group, variants in GROUPS.items(): render_svg(args.root, group, variants, args.seeds, args.bins, args.output.parent / f"rq11_{group}_reliability.svg")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": "ready"}))


if __name__ == "__main__": main()
