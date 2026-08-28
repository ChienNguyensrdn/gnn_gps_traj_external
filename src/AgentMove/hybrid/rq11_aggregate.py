from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .paired_order_test import (bootstrap_and_permutation_many, holm_adjust,
                                load_npz, paired_differences)
from .rq11_calibration import reliability


GROUPS = {"distillation": ("none", "gru", "transformer"),
          "bayesian": ("B0-static", "B3-dbn")}
METRICS = ("nll", "brier", "ece", "adaptive_ece", "accuracy_confidence_gap")


def summary(values):
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def load_runs(root: Path, seeds: list[int]) -> dict:
    output = {}
    for group, variants in GROUPS.items():
        output[group] = {}
        expected_protocol = "last-query" if group == "distillation" else "all-prefix"
        for variant in variants:
            rows = []
            for seed in seeds:
                path = root / group / variant / f"seed-{seed}" / "rq11.metrics.json"
                if not path.is_file(): raise FileNotFoundError(f"missing RQ11 artifact: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("temperature_fit_split") != "validation" or row.get("evaluation_split") != "test":
                    raise ValueError(f"calibration split leakage gate failed: {path}")
                if row.get("protocol") != expected_protocol: raise ValueError(f"protocol mismatch: {path}")
                rows.append(row)
            output[group][variant] = rows
    return output


def aggregate_runs(runs: dict) -> dict:
    result = {}
    for group, variants in runs.items():
        result[group] = {}
        for variant, rows in variants.items():
            result[group][variant] = {
                "temperature": summary([row["temperature"] for row in rows]),
                "before": {metric: summary([row["metrics_before"][metric] for row in rows]) for metric in METRICS},
                "after": {metric: summary([row["metrics_after"][metric] for row in rows]) for metric in METRICS},
            }
    return result


def paired_improvements(root: Path, seeds: list[int], iterations: int, random_seed: int) -> list[dict]:
    rows = []
    for variant_index, (group, variant) in enumerate((
        (group, variant) for group, variants in GROUPS.items() for variant in variants)):
        differences = []
        for seed in seeds:
            folder = root / group / variant / f"seed-{seed}"
            after = load_npz(folder / "after.predictions.npz"); before = load_npz(folder / "before.predictions.npz")
            differences.append(np.column_stack([paired_differences(after, before, metric) for metric in ("nll", "brier")]))
        effects, intervals, pvalues = bootstrap_and_permutation_many(differences, iterations, random_seed + variant_index)
        for index, metric in enumerate(("nll", "brier")):
            rows.append({"group": group, "variant": variant, "metric": metric,
                         "effect_after_better": float(effects[index]), "bootstrap_ci95": intervals[index].tolist(),
                         "permutation_p": float(pvalues[index])})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < .05
    return rows


def _ece(confidence, correct, bins):
    rows = reliability(confidence, correct, bins, False)
    return sum(row["count"] * abs(row["accuracy"] - row["confidence"]) for row in rows if row["count"]) / len(confidence)


def _poisson_ece(confidence, correct, weights, bins):
    identifiers = np.minimum((confidence * bins).astype(int), bins - 1); delta = correct.astype(float) - confidence
    numerator = np.zeros(len(weights))
    for index in range(bins):
        mask = identifiers == index
        if np.any(mask): numerator += np.abs(weights[:, mask] @ delta[mask])
    return numerator / np.maximum(weights.sum(axis=1), 1)


def ece_bootstrap(root: Path, seeds: list[int], iterations: int, random_seed: int, bins: int) -> list[dict]:
    rng = np.random.default_rng(random_seed); rows = []
    for group, variants in GROUPS.items():
        for variant in variants:
            loaded = []
            for seed in seeds:
                folder = root / group / variant / f"seed-{seed}"
                loaded.append((load_npz(folder / "after.predictions.npz"), load_npz(folder / "before.predictions.npz")))
            samples = np.zeros(iterations); observed = []
            for after, before in loaded:
                observed.append(_ece(before["confidence"], before["top1_correct"], bins) -
                                _ece(after["confidence"], after["top1_correct"], bins))
                for start in range(0, iterations, 32):
                    size = min(32, iterations - start)
                    weights = rng.poisson(1.0, size=(size, len(after["labels"]))).astype(np.float32)
                    samples[start:start + size] += (_poisson_ece(before["confidence"], before["top1_correct"], weights, bins) -
                                                    _poisson_ece(after["confidence"], after["top1_correct"], weights, bins)) / len(loaded)
            rows.append({"group": group, "variant": variant, "metric": "ece",
                         "method": "paired Poisson bootstrap, seed-macro",
                         "effect_after_better": float(np.mean(observed)),
                         "bootstrap_ci95": np.quantile(samples, [.025, .975]).tolist()})
    return rows


def render_svg(root: Path, group: str, variants: tuple[str, ...], seeds: list[int], bins: int, output: Path):
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    width, height, margin = 680, 520, 65; lines = []
    for variant_index, variant in enumerate(variants):
        for stage_index, stage in enumerate(("before", "after")):
            confidence, correct = [], []
            for seed in seeds:
                data = load_npz(root / group / variant / f"seed-{seed}" / f"{stage}.predictions.npz")
                confidence.append(data["confidence"]); correct.append(data["top1_correct"])
            rows = reliability(np.concatenate(confidence), np.concatenate(correct), bins, False)
            points = [(margin + row["confidence"] * (width - 2 * margin),
                       height - margin - row["accuracy"] * (height - 2 * margin)) for row in rows if row["count"]]
            color = colors[variant_index]; dash = "8 5" if stage == "before" else "none"
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
            label_y = 28 + 20 * (variant_index * 2 + stage_index)
            lines.append(f'<line x1="410" y1="{label_y}" x2="445" y2="{label_y}" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/><text x="452" y="{label_y + 4}" font-size="13">{variant} {stage}</text>')
    plot = "\n".join(lines); diagonal = f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#64748b" stroke-dasharray="5 5"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{margin}" y="25" font-size="18" font-weight="bold">RQ11 {group} reliability</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="black"/>{diagonal}
{plot}<text x="{width/2-35}" y="{height-15}" font-size="14">Confidence</text><text x="15" y="{height/2}" font-size="14" transform="rotate(-90 15 {height/2})">Accuracy</text></svg>'''
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(svg, encoding="utf-8")


def render(payload: dict) -> str:
    lines = ["# RQ11 — Calibration", "", "> Temperature chỉ fit trên validation; test không dùng để tuning. Hai protocol được báo cáo riêng.", ""]
    for group, variants in GROUPS.items():
        protocol = "last-query" if group == "distillation" else "all-prefix"
        lines += [f"## {group} ({protocol})", "", "| Variant | T | NLL trước | NLL sau | Brier trước | Brier sau | ECE trước | ECE sau | Adaptive ECE trước | Adaptive ECE sau |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in variants:
            row = payload["groups"][group][variant]; before, after = row["before"], row["after"]
            value = lambda section, metric: section[metric]["mean"]
            lines.append(f"| {variant} | {row['temperature']['mean']:.4f} | {value(before,'nll'):.6f} | {value(after,'nll'):.6f} | "
                         f"{value(before,'brier'):.6f} | {value(after,'brier'):.6f} | {value(before,'ece'):.6f} | {value(after,'ece'):.6f} | "
                         f"{value(before,'adaptive_ece'):.6f} | {value(after,'adaptive_ece'):.6f} |")
        lines += ["", f"Reliability diagram: `results/beliefmove-evo/aggregated/rq11_{group}_reliability.svg`", ""]
    lines += ["## Paired NLL/Brier improvement", "", "| Protocol | Variant | Metric | Effect sau tốt hơn | 95% CI | Holm p | Significant |",
              "|---|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['group']} | {row['variant']} | {row['metric']} | {row['effect_after_better']:.6f} | {low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | {'yes' if row['significant_at_0.05'] else 'no'} |")
    lines += ["", "## Bootstrap ECE improvement", "", "| Protocol | Variant | Metric | Effect sau tốt hơn | 95% CI |",
              "|---|---|---|---:|---:|"]
    for row in payload["ece_bootstrap"]:
        low, high = row["bootstrap_ci95"]
        lines.append(f"| {row['group']} | {row['variant']} | {row['metric']} | {row['effect_after_better']:.6f} | {low:.6f}–{high:.6f} |")
    lines += ["", "## Protocol gate", "", "- Distillation last-query và Bayesian all-prefix không được so trực tiếp trị tuyệt đối.",
              "- Temperature fit trên validation; transition/prior fit train; B3 weight lấy từ validation RQ7.",
              "- Kết quả hiện chỉ áp dụng cho Tokyo và các seed được khai báo, chưa phải 12-city.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ11 calibration experiments")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--ece-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=15); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); runs = load_runs(args.root, args.seeds)
    payload = {"rq": "RQ11", "seeds": args.seeds, "groups": aggregate_runs(runs),
               "paired_tests": paired_improvements(args.root, args.seeds, args.iterations, args.random_seed),
               "ece_bootstrap": ece_bootstrap(args.root, args.seeds, args.ece_iterations, args.random_seed + 100, args.bins), "gate": "ready"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    for group, variants in GROUPS.items(): render_svg(args.root, group, variants, args.seeds, args.bins, args.output.parent / f"rq11_{group}_reliability.svg")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": "ready"}))


if __name__ == "__main__": main()
