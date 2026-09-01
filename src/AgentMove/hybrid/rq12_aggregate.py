from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

NEURAL = ("teacher-gru", "teacher-transformer", "student-none", "student-gru", "student-transformer")
BAYESIAN = ("B0-static", "B3-dbn")
PROFILES = ("batch-1", "batch-256")
TIMING = ("latency_mean_seconds", "latency_p50_seconds", "latency_p95_seconds", "throughput_queries_per_second")


def summary(values):
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None}


def optional_summary(values):
    return {"mean": None, "std": None} if all(value is None for value in values) else summary(values)


def load_group(root: Path, profile: str, group: str, variants, seeds, allow_contention=False):
    output = {}; expected_batch = 1 if profile == "batch-1" else 256
    for variant in variants:
        rows = []
        for seed in seeds:
            path = root / profile / group / variant / f"seed-{seed}" / "rq12.metrics.json"
            if not path.is_file(): raise FileNotFoundError(f"missing RQ12 benchmark: {path}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("max_batches") is not None: raise ValueError(f"smoke benchmark cannot enter full RQ12: {path}")
            if row["batch_size"] != expected_batch: raise ValueError(f"{profile} requires batch_size={expected_batch}: {path}")
            foreign = row.get("hardware", {}).get("foreign_gpu_processes", [])
            if foreign and not allow_contention: raise ValueError(f"GPU contention detected in {path}: {foreign}")
            rows.append(row)
        protocol = {(row["hardware"]["device_name"], row["batch_size"], row["repeats"],
                     row["warmup_batches"], row.get("query_limit")) for row in rows}
        if len(protocol) != 1: raise ValueError(f"unmatched protocol for {profile}/{group}/{variant}: {protocol}")
        output[variant] = {
            "quality": {metric: summary([row["quality"][metric] for row in rows]) for metric in ("recall@1", "recall@5", "recall@10", "mrr")},
            "timing": {metric: summary([row["timing"][metric] for row in rows]) for metric in TIMING},
            "model_seconds": summary([row["timing"]["model_seconds"] for row in rows]),
            "postprocessing_fusion_seconds": summary([row["timing"]["fusion_seconds"] for row in rows]),
            "gpu_peak_allocated_mb": optional_summary([row["memory"]["gpu_peak_allocated_mb"] for row in rows]),
            "rss_peak_mb": summary([row["memory"]["rss_peak_mb"] for row in rows]),
            "parameters": summary([row["parameters"] for row in rows]),
            "protocol": rows[0]["protocol"], "hardware": rows[0]["hardware"],
            "batch_size": rows[0]["batch_size"], "repeats": rows[0]["repeats"],
            "query_limit": rows[0].get("query_limit"), "available_queries": rows[0].get("available_queries"),
            "timed_unique_queries": rows[0].get("timed_unique_queries"),
        }
    return output


def load_routing(path: Path):
    if not path.is_file(): raise FileNotFoundError(f"missing bounded RQ8 summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != "ready-bounded": raise ValueError("RQ8 routing source must be explicitly bounded")
    return {"limit": payload["limit"], "latency_source": "recorded live Ollama cache-generation latency", "policies": payload["policies"]}


def value(row, section, metric): return row[section][metric]["mean"]


def mean_std(item, scale=1.0, digits=4):
    if item["mean"] is None: return "N/A"
    mean = item["mean"] * scale
    return f"{mean:.{digits}f}" if item["std"] is None else f"{mean:.{digits}f} ± {item['std'] * scale:.{digits}f}"


def render_profile(lines, profile, data):
    title = "Batch-1 — single-request latency" if profile == "batch-1" else "Batch-256 — throughput"
    neural_sample = next(iter(data["neural"].values()))
    bayesian_sample = next(iter(data["bayesian"].values()))
    sample_text = lambda row: (f"mẫu timing xác định {row['timed_unique_queries']}/{row['available_queries']} query"
                               if row["query_limit"] else f"toàn bộ {row['available_queries']} query")
    lines += [f"## {title}", "", "### Neural — last-query", "",
              f"> {sample_text(neural_sample)}; chất lượng lấy từ full frozen test metrics. Timing/memory là mean ± std qua seed.", "",
              "| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P50 ms/q | P95 ms/q | Query/s | GPU peak MB | RSS peak MB | Params |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in data["neural"].items():
        lines.append(f"| {variant} | {mean_std(row['quality']['recall@1'],1,6)} | {mean_std(row['quality']['recall@5'],1,6)} | {mean_std(row['quality']['recall@10'],1,6)} | {mean_std(row['quality']['mrr'],1,6)} | "
                     f"{mean_std(row['timing']['latency_mean_seconds'],1000)} | {mean_std(row['timing']['latency_p50_seconds'],1000)} | {mean_std(row['timing']['latency_p95_seconds'],1000)} | "
                     f"{mean_std(row['timing']['throughput_queries_per_second'],1,2)} | {mean_std(row['gpu_peak_allocated_mb'],1,1)} | {mean_std(row['rss_peak_mb'],1,1)} | {row['parameters']['mean']:.0f} |")
    lines += ["", "### Bayesian — all-prefix", "",
              f"> {sample_text(bayesian_sample)} all-prefix; chất lượng lấy từ full frozen test metrics. Không so trực tiếp với Neural last-query.", "",
              "| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P95 ms/q | Query/s | Model s | Post-processing/Fusion s | GPU peak MB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in data["bayesian"].items():
        lines.append(f"| {variant} | {mean_std(row['quality']['recall@1'],1,6)} | {mean_std(row['quality']['recall@5'],1,6)} | {mean_std(row['quality']['recall@10'],1,6)} | {mean_std(row['quality']['mrr'],1,6)} | "
                     f"{mean_std(row['timing']['latency_mean_seconds'],1000)} | {mean_std(row['timing']['latency_p95_seconds'],1000)} | {mean_std(row['timing']['throughput_queries_per_second'],1,2)} | "
                     f"{mean_std(row['model_seconds'],1,3)} | {mean_std(row['postprocessing_fusion_seconds'],1,3)} | {mean_std(row['gpu_peak_allocated_mb'],1,1)} |")
    lines.append("")


def render(payload):
    lines = ["# RQ12 — Accuracy–Efficiency Trade-off", "",
             "> Batch-1 và batch-256 được báo cáo riêng. Neural/Bayesian dùng warm-up và CUDA synchronization; LLM latency lấy từ live cache-generation RQ8 và được ghi nhãn bounded.", ""]
    for profile in PROFILES: render_profile(lines, profile, payload["profiles"][profile])
    routing = payload["routing"]
    lines += [f"## LLM routing — bounded Tokyo limit={routing['limit']}", "", f"> Latency source: {routing['latency_source']}. Không cùng timing harness với neural.", "",
              "| Policy | R@1 | R@5 | R@10 | MRR | Call rate | Mean latency s/q | P95 latency s/q | Tokens/query |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in ("never", "entropy", "always", "random-budget-matched"):
        row = routing["policies"][policy]; get = lambda name: row[name]["mean"]
        lines.append(f"| {policy} | {get('recall@1'):.6f} | {get('recall@5'):.6f} | {get('recall@10'):.6f} | {get('mrr'):.6f} | {get('llm_call_rate'):.6f} | {get('latency_mean'):.6f} | {get('latency_p95'):.6f} | {get('tokens_per_query'):.2f} |")
    lines += ["", "## Protocol và giới hạn", "",
              "- Batch-1 đo single-request latency trên mẫu query xác định; batch-256 đo throughput trên toàn bộ test query.",
              "- Neural last-query và Bayesian all-prefix được báo cáo riêng; không so trực tiếp latency/quality tuyệt đối giữa hai query protocol.",
              "- Timing loại checkpoint loading, CSV loading, preprocessing và warm-up; có tính CPU→device transfer và online forward/post-processing/fusion.",
              "- Aggregate mặc định từ chối run có GPU process ngoại lai; override sẽ được ghi rõ trong gate.",
              "- Offline teacher training và LLM cache construction ghi N/A vì chưa có timer chuẩn từ đầu.",
              "- RQ8 là bounded limit hữu hạn và không cùng timing harness với PyTorch.",
              "- Kết quả chỉ áp dụng cho Tokyo và hardware đã ghi trong JSON, chưa phải 12-city/cross-hardware.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ12 quality-efficiency benchmarks")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rq8-summary", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True); parser.add_argument("--allow-contention", action="store_true")
    args = parser.parse_args(); profiles = {}
    for profile in PROFILES:
        neural = load_group(args.root, profile, "neural", NEURAL, args.seeds, args.allow_contention)
        bayesian = load_group(args.root, profile, "bayesian", BAYESIAN, args.seeds, args.allow_contention)
        protocols = {(row["hardware"]["device_name"], row["batch_size"], row["repeats"], row["query_limit"])
                     for row in list(neural.values()) + list(bayesian.values())}
        if len(protocols) != 1: raise ValueError(f"{profile} requires one matched timing protocol: {protocols}")
        profiles[profile] = {"neural": neural, "bayesian": bayesian}
    payload = {"rq": "RQ12", "seeds": args.seeds, "profiles": profiles, "routing": load_routing(args.rq8_summary),
               "offline_cost": {"teacher_training": None, "llm_cache_generation": None},
               "gpu_contention_allowed": args.allow_contention,
               "gate": "ready-tokyo-mixed-timing-sources" + ("-contention-allowed" if args.allow_contention else "")}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": payload["gate"]}))


if __name__ == "__main__": main()
