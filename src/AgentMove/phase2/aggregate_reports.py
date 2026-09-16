from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np

from hybrid.paired_order_test import (METRICS, bootstrap_and_permutation_many,
                                      holm_adjust, load_npz, paired_differences)


METRIC_NAMES = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")
RQ4_VARIANTS = ("E0-ce", "E1-kd", "E5-dual")
RQ7_VARIANTS = ("B0-static", "B1-history", "B2-sequential", "B3-dbn")
CITIES = ("Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris", "Beijing",
          "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow")


def dataset_specs(root: Path, cities: list[str]) -> list[dict]:
    return [
        {"key": "tist2015", "label": "TIST2015 macro (12 thành phố)", "units": cities,
         "root": root / "tist2015" / "artifacts" / "full"},
        {"key": "www2019", "label": "WWW2019-Shanghai", "units": ["Shanghai"],
         "root": root / "www2019" / "artifacts" / "full"},
        {"key": "yjmob100k", "label": "YJMob100K-Dataset1", "units": ["YJMob"],
         "root": root / "yjmob100k" / "artifacts" / "full"},
    ]


def summarize(values: list[float]) -> dict:
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def summarize_units(rows: dict[str, list[dict]], metric: str) -> dict:
    # Each TIST city has equal weight; single-unit datasets retain seed dispersion.
    if len(rows) == 1:
        return summarize([row[metric] for row in next(iter(rows.values()))])
    return summarize([np.mean([row[metric] for row in unit_rows]) for unit_rows in rows.values()])


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def paired_tests(blocks: dict[tuple[str, str], list[np.ndarray]], iterations: int) -> list[dict]:
    rows = []
    for comparison_index, ((dataset, comparison), differences) in enumerate(blocks.items()):
        effects, intervals, pvalues = bootstrap_and_permutation_many(
            differences, iterations, 42 + comparison_index)
        for index, metric in enumerate(METRICS):
            rows.append({"dataset": dataset, "comparison": comparison, "metric": metric,
                         "effect_favoring_first": float(effects[index]),
                         "bootstrap_ci95": intervals[index].tolist(),
                         "permutation_p": float(pvalues[index])})
    adjusted = holm_adjust([row["permutation_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
        row["significant_at_0.05"] = value < .05
    return rows


def prediction_block(paths: list[tuple[Path, Path]]) -> list[np.ndarray]:
    blocks = []
    for left_path, right_path in paths:
        left, right = load_npz(left_path), load_npz(right_path)
        blocks.append(np.column_stack([paired_differences(left, right, metric) for metric in METRICS]))
    return blocks


def collect_rq4(root: Path, seeds: list[int], cities: list[str], iterations: int) -> dict:
    datasets, blocks, missing = {}, {}, []
    for spec in dataset_specs(root, cities):
        all_rows = {variant: {} for variant in RQ4_VARIANTS}
        for unit in spec["units"]:
            for variant in RQ4_VARIANTS:
                rows = []
                for seed in seeds:
                    folder = spec["root"] / unit / variant / "correct" / f"seed-{seed}"
                    for name in ("test.metrics.json", "test.predictions.npz"):
                        if not (folder / name).is_file(): missing.append(str(folder / name))
                    if (folder / "test.metrics.json").is_file() and (folder / "test.predictions.npz").is_file():
                        rows.append(read_metrics(folder / "test.metrics.json"))
                if len(rows) == len(seeds): all_rows[variant][unit] = rows
        if any(len(all_rows[v]) != len(spec["units"]) for v in RQ4_VARIANTS): continue
        datasets[spec["key"]] = {
            "label": spec["label"],
            "variants": {variant: {metric: summarize_units(all_rows[variant], metric)
                                    for metric in METRIC_NAMES} for variant in RQ4_VARIANTS},
        }
        for left, right in (("E1-kd", "E0-ce"), ("E5-dual", "E1-kd")):
            paths = []
            for unit in spec["units"]:
                for seed in seeds:
                    base = spec["root"] / unit
                    paths.append((base / left / "correct" / f"seed-{seed}" / "test.predictions.npz",
                                  base / right / "correct" / f"seed-{seed}" / "test.predictions.npz"))
            blocks[(spec["key"], f"{left}-vs-{right}")] = prediction_block(paths)
    return {"rq": "RQ4", "seeds": seeds, "datasets": datasets,
            "paired_tests": paired_tests(blocks, iterations) if blocks else [],
            "missing": sorted(set(missing)), "gate": "ready" if not missing else "incomplete"}


def collect_rq7(root: Path, seeds: list[int], cities: list[str], iterations: int) -> dict:
    datasets, blocks, missing = {}, {}, []
    for spec in dataset_specs(root, cities):
        runs_by_unit = {}
        for unit in spec["units"]:
            runs = []
            for seed in seeds:
                folder = spec["root"] / unit / "E5-dual" / "correct" / f"seed-{seed}" / "rq7"
                metric_path = folder / "rq7.metrics.json"
                required = [metric_path] + [folder / f"{v}.test.predictions.npz" for v in ("B0-static", "B3-dbn")]
                missing.extend(str(path) for path in required if not path.is_file())
                if metric_path.is_file():
                    run = json.loads(metric_path.read_text(encoding="utf-8"))
                    if run.get("fit_splits") != ["train", "validation"] or run.get("evaluation_split") != "test":
                        raise ValueError(f"RQ7 sai protocol split: {metric_path}")
                    runs.append(run)
            if len(runs) == len(seeds): runs_by_unit[unit] = runs
        if len(runs_by_unit) != len(spec["units"]): continue
        variants, weights = {}, {}
        for variant in RQ7_VARIANTS:
            per_unit = {unit: [run["test_metrics"][variant] for run in runs]
                        for unit, runs in runs_by_unit.items()}
            variants[variant] = {metric: summarize_units(per_unit, metric) for metric in METRIC_NAMES}
            weights[variant] = [float(run["selected_weights"][variant])
                                for runs in runs_by_unit.values() for run in runs]
        datasets[spec["key"]] = {"label": spec["label"], "variants": variants,
                                  "selected_weights": weights}
        paths = []
        for unit in spec["units"]:
            for seed in seeds:
                folder = spec["root"] / unit / "E5-dual" / "correct" / f"seed-{seed}" / "rq7"
                paths.append((folder / "B3-dbn.test.predictions.npz",
                              folder / "B0-static.test.predictions.npz"))
        blocks[(spec["key"], "B3-dbn-vs-B0-static")] = prediction_block(paths)
    return {"rq": "RQ7", "seeds": seeds, "datasets": datasets,
            "paired_tests": paired_tests(blocks, iterations) if blocks else [],
            "missing": sorted(set(missing)), "gate": "ready" if not missing else "incomplete"}


def fmt(value: dict) -> str:
    return f"{value['mean']:.6f}" if value["std"] is None else f"{value['mean']:.6f} ± {value['std']:.6f}"


def find_test(payload: dict, dataset: str, comparison: str, metric: str) -> dict | None:
    return next((row for row in payload["paired_tests"] if row["dataset"] == dataset
                 and row["comparison"] == comparison and row["metric"] == metric), None)


def verdict(row: dict | None) -> str:
    if row is None: return "chưa đủ artifact"
    if not row["significant_at_0.05"]: return "chưa có ý nghĩa sau Holm"
    return "cải thiện có ý nghĩa" if row["effect_favoring_first"] > 0 else "suy giảm có ý nghĩa"


def result_row(label: str, variant: str, values: dict, weight: str | None = None) -> str:
    cells = [label, variant]
    if weight is not None: cells.append(weight)
    cells.extend(fmt(values[metric]) for metric in METRIC_NAMES)
    return "| " + " | ".join(cells) + " |"


def render_rq4(payload: dict) -> str:
    lines = ["# Phase 2 — RQ4: Chắt lọc tri thức và dual-axis evolution", "",
             "## Câu hỏi và mục tiêu", "",
             "KD có cải thiện CE và dual-axis có tạo lợi ích bổ sung ngoài KD trên cả ba bộ dữ liệu không? RQ4 tách đóng góp của KD (E1 so với E0) khỏi đóng góp thêm của dual-axis (E5 so với E1).", "",
             "## Thử nghiệm", "", "- E0-ce, E1-kd và E5-dual; seed 42–44; checkpoint chọn trên validation.",
             "- Test last-query; cùng split và candidate space trong từng dataset.",
             "- TIST2015 là macro đều 12 thành phố; hai dataset còn lại lấy trung bình seed.",
             "- Paired bootstrap/sign-flip dùng cùng query và seed; Holm correction áp dụng chung.", "",
             "## Kết quả", "", "| Dataset | Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        lines.extend(result_row(data["label"], variant, data["variants"][variant]) for variant in RQ4_VARIANTS)
    lines += ["", "## Kiểm định R@1", "", "| Dataset | So sánh | Effect | 95% CI | Holm p | Kết luận |",
              "|---|---|---:|---:|---:|---|"]
    for key, data in payload["datasets"].items():
        for comparison in ("E1-kd-vs-E0-ce", "E5-dual-vs-E1-kd"):
            row = find_test(payload, key, comparison, "recall@1")
            if row:
                low, high = row["bootstrap_ci95"]
                lines.append(f"| {data['label']} | {comparison} | {row['effect_favoring_first']:.6f} | {low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | {verdict(row)} |")
    lines += ["", "## Phân tích", ""]
    for key, data in payload["datasets"].items():
        lines.append(f"- **{data['label']}**: KD {verdict(find_test(payload, key, 'E1-kd-vs-E0-ce', 'recall@1'))}; dual-axis ngoài KD {verdict(find_test(payload, key, 'E5-dual-vs-E1-kd', 'recall@1'))}.")
    lines += ["", "## Điều kiện đạt", "", "KD được xác nhận khi E1 vượt E0 ổn định trên nhiều dataset. Dual-axis chỉ được gọi là khái quát nếu E5 vượt E1 có ý nghĩa trên ít nhất hai dataset; nếu không phải giới hạn kết luận theo miền.", "",
              f"Gate: **{payload['gate']}**; artifact thiếu: **{len(payload['missing'])}**.", ""]
    return "\n".join(lines)


def render_rq7(payload: dict) -> str:
    lines = ["# Phase 2 — RQ7: Belief memory tuần tự", "", "## Câu hỏi và mục tiêu", "",
             "Cập nhật belief bằng prior và transition theo lịch sử có cải thiện dự đoán all-prefix ổn định trên ba bộ dữ liệu không? Trọng tâm là B3-dbn so với B0-static và đánh đổi calibration.", "",
             "## Thử nghiệm", "", "- Frozen E5-dual; prior/transition chỉ fit train; weight chọn trên validation.",
             "- Test all-prefix và reset belief tại biên trajectory; không so trị tuyệt đối với RQ4 last-query.",
             "- Paired test dùng cùng prefix/seed; Holm correction áp dụng chung.", "", "## Kết quả", "",
             "| Dataset | Variant | Weight validation | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for data in payload["datasets"].values():
        for variant in RQ7_VARIANTS:
            weights = ", ".join(f"{value:g}" for value in sorted(set(data["selected_weights"][variant])))
            lines.append(result_row(data["label"], variant, data["variants"][variant], weights))
    lines += ["", "## Kiểm định B3-dbn so với B0-static", "", "| Dataset | Metric | Effect | 95% CI | Holm p | Kết luận |",
              "|---|---|---:|---:|---:|---|"]
    for key, data in payload["datasets"].items():
        for metric in ("recall@1", "mrr", "nll", "brier"):
            row = find_test(payload, key, "B3-dbn-vs-B0-static", metric)
            if row:
                low, high = row["bootstrap_ci95"]
                lines.append(f"| {data['label']} | {metric} | {row['effect_favoring_first']:.6f} | {low:.6f}–{high:.6f} | {row['holm_adjusted_p']:.6g} | {verdict(row)} |")
    lines += ["", "## Phân tích", ""]
    for key, data in payload["datasets"].items():
        lines.append(f"- **{data['label']}**: R@1 {verdict(find_test(payload, key, 'B3-dbn-vs-B0-static', 'recall@1'))}; NLL {verdict(find_test(payload, key, 'B3-dbn-vs-B0-static', 'nll'))}. Positive effect luôn nghĩa là B3 tốt hơn vì NLL/Brier đã đảo dấu.")
    lines += ["", "## Điều kiện đạt", "", "B3 được xem là khái quát nếu R@1/MRR tăng có ý nghĩa trên ít nhất hai dataset. Nếu ranking tăng nhưng NLL/Brier/ECE xấu đi, chỉ kết luận lợi ích ranking và yêu cầu calibration sau fusion.", "",
              f"Gate: **{payload['gate']}**; artifact thiếu: **{len(payload['missing'])}**.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp Phase 2 RQ4/RQ7")
    parser.add_argument("rq", choices=("rq4", "rq7", "all"))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PHASE2_ROOT", "results/phase2")))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cities", nargs="+", default=list(CITIES))
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("SIGNIFICANCE_ITERATIONS", "10000")))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1000: parser.error("--iterations phải >= 1000")
    targets = ("rq4", "rq7") if args.rq == "all" else (args.rq,)
    output_dir = args.root / "aggregated"; output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(__file__).resolve().parent
    payloads = []
    for rq in targets:
        payload = (collect_rq4 if rq == "rq4" else collect_rq7)(args.root, args.seeds, args.cities, args.iterations)
        json_path, report_path = output_dir / f"{rq}_summary.json", report_dir / f"{rq}_report.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text((render_rq4 if rq == "rq4" else render_rq7)(payload), encoding="utf-8")
        print(json.dumps({"rq": rq.upper(), "gate": payload["gate"], "missing": len(payload["missing"]),
                          "json": str(json_path), "report": str(report_path)}, ensure_ascii=False))
        payloads.append(payload)
    if any(payload["gate"] != "ready" for payload in payloads) and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
