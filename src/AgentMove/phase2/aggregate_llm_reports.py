from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from hybrid.paired_order_test import (METRICS as PAIRED_METRICS,
                                      bootstrap_and_permutation_many, holm_adjust,
                                      load_npz, paired_differences)
from hybrid.rq3_distillation import VARIANTS as RQ3_VARIANTS
from hybrid.rq8_routing import POLICIES
from phase2.aggregate_reports import CITIES, METRIC_NAMES, dataset_specs


RQ3_COMPARISONS = (("M2-llm", "M1-data-only"), ("M4-both", "M3-quantitative"))
RQ8_METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "llm_call_rate",
               "latency_mean", "latency_p95", "tokens_per_query")
RQ8_PAIRED = ("recall@1", "recall@5", "recall@10", "mrr")


def summary(values: list[float]) -> dict:
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def macro(unit_values: dict[str, list[float]]) -> dict:
    if len(unit_values) == 1:
        return summary(next(iter(unit_values.values())))
    return summary([float(np.mean(values)) for values in unit_values.values()])


def adjusted_tests(raw: list[dict]) -> list[dict]:
    adjusted = holm_adjust([row["p"] for row in raw])
    for row, value in zip(raw, adjusted):
        row["holm_p"] = value; row["significant"] = value < .05
    return raw


def rq3_root(spec: dict, unit: str, slug: str, limit: int) -> Path:
    return spec["root"] / unit / f"rq3/{slug}/limit-{limit}/seed-42"


def collect_rq3(root: Path, slug: str, limit: int, cities: list[str], iterations: int) -> dict:
    datasets, tests, missing = {}, [], []
    for spec_index, spec in enumerate(dataset_specs(root, cities)):
        runs = {}; predictions = {}
        for unit in spec["units"]:
            folder = rq3_root(spec, unit, slug, limit); metric = folder / "rq3.metrics.json"
            required = [metric] + [folder / f"{variant}.test.predictions.npz" for variant in RQ3_VARIANTS]
            missing.extend(str(path) for path in required if not path.is_file())
            if all(path.is_file() for path in required):
                run = json.loads(metric.read_text());
                if run.get("limit") != limit: raise ValueError(f"RQ3 limit mismatch: {metric}")
                runs[unit] = run
                predictions[unit] = {variant: load_npz(folder / f"{variant}.test.predictions.npz")
                                     for variant in RQ3_VARIANTS}
        if len(runs) != len(spec["units"]): continue
        variants = {}
        for variant in RQ3_VARIANTS:
            variants[variant] = {metric: macro({
                unit: [runs[unit]["test_metrics"][variant][metric]]
                for unit in spec["units"]
            })
                                 for metric in METRIC_NAMES}
            variants[variant]["queries"] = int(sum(runs[unit]["test_metrics"][variant]["queries"]
                                                    for unit in spec["units"]))
            variants[variant]["weights"] = sorted({
                (float(runs[unit]["selected_weights"][variant]["quantitative"]),
                 float(runs[unit]["selected_weights"][variant]["llm"]))
                for unit in spec["units"]})
        coverage = macro({unit: [runs[unit]["test_evidence"]["query_coverage"]]
                          for unit in spec["units"]})
        valid_rate = macro({unit: [runs[unit]["test_evidence"]["valid_evidence_rate"]]
                            for unit in spec["units"]})
        datasets[spec["key"]] = {"label": spec["label"], "variants": variants,
                                  "cache_coverage": coverage, "valid_evidence_rate": valid_rate}
        for comparison_index, (left, right) in enumerate(RQ3_COMPARISONS):
            blocks = [np.column_stack([paired_differences(predictions[unit][left], predictions[unit][right], metric)
                                       for metric in PAIRED_METRICS]) for unit in spec["units"]]
            effects, intervals, pvalues = bootstrap_and_permutation_many(
                blocks, iterations, 100 + spec_index * 10 + comparison_index)
            for index, metric in enumerate(PAIRED_METRICS):
                tests.append({"dataset": spec["key"], "comparison": f"{left}-vs-{right}", "metric": metric,
                              "effect": float(effects[index]), "ci95": intervals[index].tolist(),
                              "p": float(pvalues[index])})
    return {"rq": "RQ3", "model": slug, "limit": limit, "datasets": datasets,
            "paired_tests": adjusted_tests(tests) if tests else [], "missing": sorted(set(missing)),
            "gate": "ready-bounded" if not missing else "incomplete"}


def rq8_root(spec: dict, unit: str, slug: str, limit: int) -> Path:
    return spec["root"] / unit / f"rq8/{slug}/limit-{limit}"


def rq8_query_values(payload: dict, metric: str) -> np.ndarray:
    ranks = payload["ranks"]
    if metric.startswith("recall@"):
        return (ranks <= int(metric.split("@")[1])).astype(float)
    return 1.0 / ranks


def collect_rq8(root: Path, slug: str, limit: int, cities: list[str], seeds: list[int], iterations: int) -> dict:
    datasets, tests, missing = {}, [], []
    for spec_index, spec in enumerate(dataset_specs(root, cities)):
        runs = {}
        for unit in spec["units"]:
            folder = rq8_root(spec, unit, slug, limit); unit_runs = []
            for seed in seeds:
                path = folder / f"seed-{seed}/rq8.metrics.json"
                if not path.is_file(): missing.append(str(path)); continue
                row = json.loads(path.read_text())
                if row.get("limit") != limit: raise ValueError(f"RQ8 limit mismatch: {path}")
                unit_runs.append(row)
            required_predictions = [folder / "seed-42" / f"{p}.test.predictions.npz"
                                    for p in ("entropy", "never", "always")]
            required_predictions += [folder / f"seed-{seed}/random-budget-matched.test.predictions.npz"
                                     for seed in seeds]
            missing.extend(str(path) for path in required_predictions if not path.is_file())
            if len(unit_runs) == len(seeds) and all(path.is_file() for path in required_predictions):
                for policy in ("never", "always", "entropy", "margin"):
                    reference = unit_runs[0]["metrics"][policy]
                    if any(any(not np.isclose(row["metrics"][policy][metric], reference[metric])
                                   for metric in RQ8_METRICS) for row in unit_runs[1:]):
                        raise ValueError(f"deterministic RQ8 policy differs across seeds: {unit}/{policy}")
                runs[unit] = unit_runs
        if len(runs) != len(spec["units"]): continue
        policies = {}
        for policy in POLICIES:
            policies[policy] = {}
            for metric in RQ8_METRICS:
                unit_values = {}
                for unit in spec["units"]:
                    selected = runs[unit] if policy == "random-budget-matched" else runs[unit][:1]
                    unit_values[unit] = [row["metrics"][policy][metric] for row in selected]
                policies[policy][metric] = macro(unit_values)
            policies[policy]["runs_per_unit"] = len(seeds) if policy == "random-budget-matched" else 1
        oracle = {metric: macro({unit: [runs[unit][0]["oracle_upper_bound"][metric]]
                                         for unit in spec["units"]})
                  for metric in ("recall@1", "recall@5", "recall@10", "mrr", "llm_call_rate")}
        datasets[spec["key"]] = {"label": spec["label"], "policies": policies, "oracle": oracle}
        # Average random-control effects over its 50 permutations before city-macro inference.
        blocks = []
        for unit in spec["units"]:
            folder = rq8_root(spec, unit, slug, limit)
            entropy = load_npz(folder / "seed-42/entropy.test.predictions.npz")
            random_payloads = [load_npz(folder / f"seed-{seed}/random-budget-matched.test.predictions.npz")
                               for seed in seeds]
            if any(not np.array_equal(entropy["query_id"], payload["query_id"])
                   for payload in random_payloads):
                raise ValueError(f"unaligned RQ8 queries: {spec['key']}/{unit}")
            columns = []
            for metric in RQ8_PAIRED:
                left = rq8_query_values(entropy, metric)
                random_mean = np.mean([rq8_query_values(payload, metric) for payload in random_payloads], axis=0)
                columns.append(left - random_mean)
            blocks.append(np.column_stack(columns))
        effects, intervals, pvalues = bootstrap_and_permutation_many(blocks, iterations, 200 + spec_index)
        for index, metric in enumerate(RQ8_PAIRED):
            tests.append({"dataset": spec["key"], "comparison": "entropy-vs-random-budget-matched",
                          "metric": metric, "effect": float(effects[index]),
                          "ci95": intervals[index].tolist(), "p": float(pvalues[index])})
    return {"rq": "RQ8", "model": slug, "limit": limit, "random_seeds": seeds,
            "datasets": datasets, "paired_tests": adjusted_tests(tests) if tests else [],
            "missing": sorted(set(missing)), "gate": "ready-bounded" if not missing else "incomplete"}


def fmt(value: dict, digits: int = 6) -> str:
    return f"{value['mean']:.{digits}f}" if value["std"] is None else \
        f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def test(payload: dict, dataset: str, comparison: str, metric: str) -> dict | None:
    return next((row for row in payload["paired_tests"] if row["dataset"] == dataset
                 and row["comparison"] == comparison and row["metric"] == metric), None)


def verdict(row: dict | None) -> str:
    if row is None: return "chưa đủ dữ liệu"
    if not row["significant"]: return "chưa có ý nghĩa"
    return "cải thiện có ý nghĩa" if row["effect"] > 0 else "suy giảm có ý nghĩa"


def render_rq3(payload: dict) -> str:
    lines = ["# Phase 2 — RQ3: Giá trị gia tăng của tri thức LLM", "",
             f"> Bounded matched experiment: `LLM_LIMIT={payload['limit']}`, model `{payload['model']}`; không phải full-query.", "",
             "## Câu hỏi và mục tiêu", "",
             "Evidence có cấu trúc từ LLM có bổ sung thông tin ngoài prior dữ liệu và teacher định lượng không? So sánh chính M4-both với M3-quantitative cô lập giá trị gia tăng của LLM khi tín hiệu định lượng được giữ nguyên.", "",
             "## Kết quả", "", "| Dataset | Variant | Queries | q-weight/LLM-weight | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        for variant in RQ3_VARIANTS:
            row = data["variants"][variant]
            weights = ", ".join(f"{q:g}/{l:g}" for q, l in row["weights"])
            lines.append(f"| {data['label']} | {variant} | {row['queries']} | {weights} | " +
                         " | ".join(fmt(row[m]) for m in METRIC_NAMES) + " |")
    lines += ["", "## Giá trị gia tăng của LLM", "", "| Dataset | Metric | Effect M4−M3 | 95% CI | Holm p | Kết luận |",
              "|---|---|---:|---:|---:|---|"]
    for key, data in payload["datasets"].items():
        for metric in ("recall@1", "mrr"):
            row = test(payload, key, "M4-both-vs-M3-quantitative", metric)
            if row:
                lines.append(f"| {data['label']} | {metric} | {row['effect']:.6f} | {row['ci95'][0]:.6f}–{row['ci95'][1]:.6f} | {row['holm_p']:.6g} | {verdict(row)} |")
    lines += ["", "## Phân tích", ""]
    for key, data in payload["datasets"].items():
        row = test(payload, key, "M4-both-vs-M3-quantitative", "mrr")
        lines.append(f"- **{data['label']}**: M4 so với M3 trên MRR — {verdict(row)}; cache coverage={data['cache_coverage']['mean']:.3f}, valid evidence rate={data['valid_evidence_rate']['mean']:.3f}.")
    lines += ["", "## Protocol gate", "", "- Prior fit train; weight chọn validation; test không dùng tuning.",
              "- Evidence được replay từ immutable cache; không pseudo-replicate LLM.",
              "- TIST2015 chỉ là macro khi đủ đúng 12 thành phố.",
              f"- Gate: **{payload['gate']}**; artifact thiếu: **{len(payload['missing'])}**.", ""]
    return "\n".join(lines)


def render_rq8(payload: dict) -> str:
    lines = ["# Phase 2 — RQ8: Uncertainty-aware LLM routing", "",
             f"> Bounded matched experiment: `LLM_LIMIT={payload['limit']}`, model `{payload['model']}`; không phải full-query.", "",
             "## Câu hỏi và mục tiêu", "", "Entropy routing có chọn đúng những query mà LLM tạo realized gain và tạo Pareto improvement so với random-budget-matched tại cùng call rate hay không?", "",
             "## Kết quả primary budget", "", "| Dataset | Router | Runs/unit | R@1 | R@5 | R@10 | MRR | Call rate | Latency mean | Latency p95 | Tokens/query |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        for policy in POLICIES:
            row = data["policies"][policy]
            lines.append(f"| {data['label']} | {policy} | {row['runs_per_unit']} | " +
                         " | ".join(fmt(row[m], 2 if m == "tokens_per_query" else 6) for m in RQ8_METRICS) + " |")
    lines += ["", "## Entropy so với random-budget-matched", "", "| Dataset | Metric | Effect | 95% CI | Holm p | Kết luận |",
              "|---|---|---:|---:|---:|---|"]
    for key, data in payload["datasets"].items():
        for metric in ("recall@1", "mrr"):
            row = test(payload, key, "entropy-vs-random-budget-matched", metric)
            if row:
                lines.append(f"| {data['label']} | {metric} | {row['effect']:.6f} | {row['ci95'][0]:.6f}–{row['ci95'][1]:.6f} | {row['holm_p']:.6g} | {verdict(row)} |")
    lines += ["", "## Oracle upper bound", "", "| Dataset | Oracle R@1 | Oracle R@5 | Oracle R@10 | Oracle MRR | Oracle call rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        oracle = data["oracle"]
        lines.append(f"| {data['label']} | " + " | ".join(fmt(oracle[m]) for m in ("recall@1", "recall@5", "recall@10", "mrr", "llm_call_rate")) + " |")
    lines += ["", "## Phân tích", ""]
    for key, data in payload["datasets"].items():
        row = test(payload, key, "entropy-vs-random-budget-matched", "mrr")
        lines.append(f"- **{data['label']}**: entropy so với random cùng ngân sách trên MRR — {verdict(row)}.")
    lines += ["", "## Protocol gate", "", "- Threshold chọn trên validation; test không dùng tuning.",
              "- Deterministic policy chỉ tính một run; 50 seed chỉ áp dụng cho random control.",
              "- Latency lấy từ live Ollama cache-generation và kết quả phải mang nhãn bounded.",
              f"- Gate: **{payload['gate']}**; artifact thiếu: **{len(payload['missing'])}**.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2 bounded RQ3/RQ8 reports")
    parser.add_argument("rq", choices=("rq3", "rq8", "all")); parser.add_argument("--root", type=Path, default=Path("results/phase2"))
    parser.add_argument("--model-slug", default="qwen2-7b"); parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--cities", nargs="+", default=list(CITIES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 92)))
    parser.add_argument("--iterations", type=int, default=10000); parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(); targets = ("rq3", "rq8") if args.rq == "all" else (args.rq,)
    output = args.root / "aggregated"; output.mkdir(parents=True, exist_ok=True); report_dir = Path(__file__).resolve().parent
    results = []
    for rq in targets:
        payload = (collect_rq3(args.root, args.model_slug, args.limit, args.cities, args.iterations)
                   if rq == "rq3" else collect_rq8(args.root, args.model_slug, args.limit, args.cities, args.seeds, args.iterations))
        json_path = output / f"{rq}_limit-{args.limit}_summary.json"; report_path = report_dir / f"{rq}_report.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if payload["gate"] != "incomplete" or args.allow_incomplete:
            report_path.write_text((render_rq3 if rq == "rq3" else render_rq8)(payload))
        print(json.dumps({"rq": rq.upper(), "gate": payload["gate"], "missing": len(payload["missing"]),
                          "json": str(json_path), "report": str(report_path)}, ensure_ascii=False))
        results.append(payload)
    if any(row["gate"] == "incomplete" for row in results) and not args.allow_incomplete: raise SystemExit(1)


if __name__ == "__main__": main()
