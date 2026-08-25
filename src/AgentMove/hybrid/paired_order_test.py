from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier")


def query_values(data: dict[str, np.ndarray], metric: str) -> np.ndarray:
    if metric.startswith("recall@"):
        return (data["ranks"] <= int(metric.split("@")[1])).astype(float)
    if metric == "mrr":
        return data["reciprocal_rank"].astype(float)
    if metric == "nll":
        return -np.log(np.clip(data["true_probability"].astype(float), 1e-12, 1.0))
    if metric == "brier":
        return data["brier"].astype(float)
    raise ValueError(f"unsupported paired metric: {metric}")


def paired_differences(correct: dict[str, np.ndarray], corrupt: dict[str, np.ndarray], metric: str) -> np.ndarray:
    for key in ("query_index", "labels"):
        if not np.array_equal(correct[key], corrupt[key]):
            raise ValueError(f"unaligned paired predictions: {key} differs")
    left = query_values(correct, metric); right = query_values(corrupt, metric)
    # Positive always means that correct order is better.
    return right - left if metric in {"nll", "brier"} else left - right


def bootstrap_and_permutation(differences: list[np.ndarray], iterations: int, seed: int) -> tuple[float, list[float], float]:
    if not differences or any(len(values) == 0 for values in differences):
        raise ValueError("paired test requires non-empty differences for every seed")
    rng = np.random.default_rng(seed)
    observed = float(np.mean([values.mean() for values in differences]))
    bootstrap = np.empty(iterations, dtype=float); null = np.empty(iterations, dtype=float)
    for index in range(iterations):
        bootstrap[index] = np.mean([
            values[rng.integers(0, len(values), size=len(values))].mean() for values in differences
        ])
        null[index] = np.mean([
            (values * rng.choice((-1.0, 1.0), size=len(values))).mean() for values in differences
        ])
    ci = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (iterations + 1))
    return observed, ci, p_value


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values); adjusted = np.empty(len(p_values), dtype=float); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing paired prediction artifact: {path}")
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def analyze(args) -> dict:
    root = Path(args.artifacts_root); rows = []
    for comparison_index, comparison in enumerate(args.comparisons):
        loaded = []
        for seed in args.seeds:
            correct = load_npz(root / "correct" / f"seed-{seed}" / "test.predictions.npz")
            corrupt = load_npz(root / comparison / f"seed-{seed}" / "test.predictions.npz")
            loaded.append((correct, corrupt))
        for metric_index, metric in enumerate(METRICS):
            differences = [paired_differences(left, right, metric) for left, right in loaded]
            effect, ci, p_value = bootstrap_and_permutation(
                differences, args.iterations, args.seed + comparison_index * len(METRICS) + metric_index
            )
            rows.append({"comparison": f"correct-vs-{comparison}", "metric": metric,
                         "effect_favoring_correct": effect, "bootstrap_ci95": ci,
                         "permutation_p": p_value, "seeds": args.seeds,
                         "queries_per_seed": [len(values) for values in differences]})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value; row["significant_at_0.05"] = value < 0.05
    result = {"method": "query-paired, seed-macro bootstrap and sign-flip permutation",
              "iterations": args.iterations, "random_seed": args.seed, "results": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        markdown = Path(args.markdown); markdown.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# RQ5 — Kiểm định paired significance", "",
                 "> Positive effect nghĩa là correct order tốt hơn corrupted order. Holm correction áp dụng cho toàn bộ phép kiểm định.", "",
                 "| Comparison | Metric | Effect | Bootstrap 95% CI | p | Holm p | Significant |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for row in rows:
            ci = row["bootstrap_ci95"]
            lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect_favoring_correct']:.6f} | "
                         f"{ci[0]:.6f}–{ci[1]:.6f} | {row['permutation_p']:.6g} | "
                         f"{row['holm_adjusted_p']:.6g} | {'yes' if row['significant_at_0.05'] else 'no'} |")
        lines += ["", "Phương pháp dùng cùng query và cùng seed 42–44; ECE không được kiểm định vì không phân rã theo query.", ""]
        markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "tests": len(rows)})); return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired order-corruption significance tests for RQ5")
    parser.add_argument("--artifacts-root", required=True); parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--comparisons", nargs="+", choices=["reverse", "random"], required=True)
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True); parser.add_argument("--markdown")
    args = parser.parse_args()
    if args.iterations < 1000: parser.error("--iterations must be at least 1000")
    analyze(args)


if __name__ == "__main__":
    main()
