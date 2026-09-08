from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict WWW2019 cross-dataset result gate")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hybrid-metrics", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

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
            runs.append(read(path))
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

    hybrid = None
    if args.hybrid_metrics:
        if args.hybrid_metrics.is_file(): hybrid = read(args.hybrid_metrics)
        else: missing.append(str(args.hybrid_metrics))
    gate = "ready-www2019" if not missing else "incomplete"
    result = {"dataset": "WWW2019-Shanghai-ISP", "protocol": "cross-dataset confirmation",
              "seeds": args.seeds, "neural": rows, "belief": belief_summary,
              "llm_bounded": hybrid, "missing": sorted(set(missing)), "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# WWW2019 — Cross-dataset validation", "", f"> Gate: **{gate}**. Dataset: Shanghai-ISP.", "",
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
    lines += ["", "## Giới hạn", "", "- WWW2019 là Shanghai-ISP, không phải thí nghiệm 12 thành phố.",
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
