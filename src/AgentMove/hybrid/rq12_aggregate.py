from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

NEURAL = ("teacher-gru", "teacher-transformer", "student-none", "student-gru", "student-transformer")
BAYESIAN = ("B0-static", "B3-dbn")
TIMING = ("latency_mean_seconds", "latency_p50_seconds", "latency_p95_seconds", "throughput_queries_per_second")


def summary(values):
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def optional_summary(values):
    return {"mean": None, "std": None} if all(value is None for value in values) else summary(values)


def load_group(root: Path, group: str, variants, seeds):
    output = {}
    for variant in variants:
        rows = []
        for seed in seeds:
            path = root / group / variant / f"seed-{seed}" / "rq12.metrics.json"
            if not path.is_file(): raise FileNotFoundError(f"missing RQ12 benchmark: {path}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("max_batches") is not None: raise ValueError(f"smoke benchmark cannot enter full RQ12: {path}")
            rows.append(row)
        hardware = {(row["hardware"]["device_name"], row["batch_size"], row["repeats"], row["warmup_batches"]) for row in rows}
        if len(hardware) != 1: raise ValueError(f"unmatched hardware/timing protocol for {group}/{variant}: {hardware}")
        output[variant] = {
            "quality": {metric: summary([row["quality"][metric] for row in rows]) for metric in ("recall@1", "recall@5", "recall@10", "mrr")},
            "timing": {metric: summary([row["timing"][metric] for row in rows]) for metric in TIMING},
            "model_seconds": summary([row["timing"]["model_seconds"] for row in rows]),
            "fusion_seconds": summary([row["timing"]["fusion_seconds"] for row in rows]),
            "gpu_peak_allocated_mb": optional_summary([row["memory"]["gpu_peak_allocated_mb"] for row in rows]),
            "rss_peak_mb": summary([row["memory"]["rss_peak_mb"] for row in rows]),
            "parameters": summary([row["parameters"] for row in rows]),
            "protocol": rows[0]["protocol"], "hardware": rows[0]["hardware"],
            "batch_size": rows[0]["batch_size"], "repeats": rows[0]["repeats"],
        }
    return output


def load_routing(path: Path):
    if not path.is_file(): raise FileNotFoundError(f"missing bounded RQ8 summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != "ready-bounded": raise ValueError("RQ8 routing source must be explicitly bounded")
    return {"limit": payload["limit"], "latency_source": "recorded live Ollama cache-generation latency",
            "policies": payload["policies"]}


def value(row, section, metric): return row[section][metric]["mean"]


def memory_text(row):
    value_mb = row["gpu_peak_allocated_mb"]["mean"]
    return "N/A" if value_mb is None else f"{value_mb:.1f}"


def render(payload):
    lines = ["# RQ12 — Accuracy–Efficiency Trade-off", "",
             "> Neural/Bayesian latency được benchmark trên cùng hardware với warm-up và CUDA synchronization. LLM latency lấy từ live cache-generation của RQ8 và được ghi nhãn bounded.", "",
             "## Neural — last-query", "",
             "| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P50 ms/q | P95 ms/q | Query/s | GPU peak MB | RSS peak MB | Params |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in payload["neural"].items():
        lines.append(f"| {variant} | {value(row,'quality','recall@1'):.6f} | {value(row,'quality','recall@5'):.6f} | "
                     f"{value(row,'quality','recall@10'):.6f} | {value(row,'quality','mrr'):.6f} | "
                     f"{1000*value(row,'timing','latency_mean_seconds'):.4f} | {1000*value(row,'timing','latency_p50_seconds'):.4f} | "
                     f"{1000*value(row,'timing','latency_p95_seconds'):.4f} | {value(row,'timing','throughput_queries_per_second'):.2f} | "
                     f"{memory_text(row)} | {row['rss_peak_mb']['mean']:.1f} | {row['parameters']['mean']:.0f} |")
    lines += ["", "## Bayesian — all-prefix", "",
              "| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P95 ms/q | Query/s | Model s | Fusion s | GPU peak MB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in payload["bayesian"].items():
        lines.append(f"| {variant} | {value(row,'quality','recall@1'):.6f} | {value(row,'quality','recall@5'):.6f} | "
                     f"{value(row,'quality','recall@10'):.6f} | {value(row,'quality','mrr'):.6f} | "
                     f"{1000*value(row,'timing','latency_mean_seconds'):.4f} | {1000*value(row,'timing','latency_p95_seconds'):.4f} | "
                     f"{value(row,'timing','throughput_queries_per_second'):.2f} | {row['model_seconds']['mean']:.3f} | "
                     f"{row['fusion_seconds']['mean']:.3f} | {memory_text(row)} |")
    routing = payload["routing"]
    lines += ["", f"## LLM routing — bounded Tokyo limit={routing['limit']}", "",
              f"> Latency source: {routing['latency_source']}. Không so trực tiếp với benchmark neural như cùng timing harness.", "",
              "| Policy | R@1 | R@5 | R@10 | MRR | Call rate | Mean latency s/q | P95 latency s/q | Tokens/query |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in ("never", "entropy", "always", "random-budget-matched"):
        row = routing["policies"][policy]; get = lambda name: row[name]["mean"]
        lines.append(f"| {policy} | {get('recall@1'):.6f} | {get('recall@5'):.6f} | {get('recall@10'):.6f} | "
                     f"{get('mrr'):.6f} | {get('llm_call_rate'):.6f} | {get('latency_mean'):.6f} | "
                     f"{get('latency_p95'):.6f} | {get('tokens_per_query'):.2f} |")
    lines += ["", "## Protocol và giới hạn", "",
              "- Neural last-query và Bayesian all-prefix được báo cáo riêng; không so trực tiếp latency/quality tuyệt đối giữa hai query protocol.",
              "- Timing loại checkpoint loading, CSV loading, preprocessing và warm-up; có tính CPU→device transfer và online forward/fusion.",
              "- Offline teacher training và LLM cache construction chưa có timer chuẩn từ đầu nên ghi N/A, không suy diễn số liệu.",
              "- RQ8 là bounded limit hữu hạn; latency của nó là recorded live Ollama latency, không phải cùng harness với PyTorch.",
              "- Kết quả hiện chỉ áp dụng cho Tokyo và hardware đã ghi trong JSON, chưa phải 12-city hoặc cross-hardware.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ12 quality-efficiency benchmarks")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rq8-summary", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True); args = parser.parse_args()
    neural = load_group(args.root, "neural", NEURAL, args.seeds)
    bayesian = load_group(args.root, "bayesian", BAYESIAN, args.seeds)
    protocols = {(json.dumps(row["hardware"], sort_keys=True), row["batch_size"], row["repeats"])
                 for row in list(neural.values()) + list(bayesian.values())}
    if len(protocols) != 1: raise ValueError(f"RQ12 requires one matched hardware/batch/repeat protocol, found {len(protocols)}")
    payload = {"rq": "RQ12", "seeds": args.seeds, "neural": neural, "bayesian": bayesian,
               "routing": load_routing(args.rq8_summary), "offline_cost": {"teacher_training": None, "llm_cache_generation": None},
               "gate": "ready-tokyo-mixed-timing-sources"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": payload["gate"]}))


if __name__ == "__main__": main()
