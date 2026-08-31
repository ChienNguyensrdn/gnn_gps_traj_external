from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint_models import build_checkpoint_model
from .dual_evolution import _device
from .neural_cgm import _batches, _torch, build_examples
from .rq7_belief_memory import (fuse, sequence_queries, training_statistics,
                                transition_prior)


def synchronize(torch, device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def rss_megabytes() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024)


def deterministic_limit(rows, limit):
    """Select a reproducible, trajectory-order-spanning timing sample."""
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("query-limit must be positive")
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def gpu_process_snapshot(device) -> list[dict]:
    if device.type != "cuda":
        return []
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return [{"pid": None, "process_name": "snapshot-unavailable", "used_gpu_memory_mb": None}]
    output = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or fields[0] == str(os.getpid()):
            continue
        try:
            memory = float(fields[2])
        except ValueError:
            memory = None
        output.append({"pid": int(fields[0]) if fields[0].isdigit() else None,
                       "process_name": fields[1], "used_gpu_memory_mb": memory})
    return output


def quality(path: str, variant: str | None) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if variant:
        values = payload["test_metrics"][variant]
    else:
        values = payload.get("metrics", payload)
    return {name: float(values[name]) for name in ("recall@1", "recall@5", "recall@10", "mrr")}


def benchmark(args):
    torch = _torch(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_checkpoint_model(checkpoint); model.load_state_dict(checkpoint["model_state"])
    device = _device(torch, args.device); model.to(device).eval()
    test = pd.read_csv(args.test_csv); model_times, fusion_times, total_times, batch_sizes = [], [], [], []
    if args.protocol == "last-query":
        all_examples = build_examples(test, checkpoint["user_map"], all_prefixes=False)
        examples = deterministic_limit(all_examples, args.query_limit)
        available_queries = len(all_examples)
        def batches(): return _batches(examples, args.batch_size, False, args.seed)
        prior = transitions = None
    else:
        if not args.train_csv or not args.rq7_metrics or args.variant not in {"B0-static", "B3-dbn"}:
            raise ValueError("all-prefix requires train CSV, RQ7 metrics and B0-static/B3-dbn variant")
        train = pd.read_csv(args.train_csv); all_queries = sequence_queries(test, checkpoint["user_map"])
        queries = deterministic_limit(all_queries, args.query_limit); available_queries = len(all_queries)
        rq7 = json.loads(Path(args.rq7_metrics).read_text(encoding="utf-8")); weight = float(rq7["selected_weights"][args.variant])
        prior, transitions = training_statistics(train, int(checkpoint["config"]["num_pois"]), args.smoothing)
        def batches():
            for start in range(0, len(queries), args.batch_size):
                chunk = queries[start:start + args.batch_size]
                examples = [(q["pois"][:q["step"]], q["slots"][:q["step"]], q["user"], q["target_slot"], q["label"]) for q in chunk]
                yield next(_batches(examples, len(examples), False, args.seed)), chunk
    # Warm-up is excluded from every reported timing.
    iterator = batches()
    for index, item in enumerate(iterator):
        batch = item[0] if args.protocol == "all-prefix" else item
        poi, slots, lengths, users, targets, _ = [value.to(device) for value in batch]
        with torch.no_grad(): model(poi, slots, lengths, users, targets)
        synchronize(torch, device)
        if index + 1 >= args.warmup_batches: break
    contention_before = gpu_process_snapshot(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    rss_before = rss_megabytes(); total_queries = 0
    for repeat in range(args.repeats):
        for batch_index, item in enumerate(batches()):
            if args.max_batches and batch_index >= args.max_batches: break
            if args.protocol == "all-prefix": batch, chunk = item
            else: batch, chunk = item, None
            batch_size = len(batch[-1]); start_total = time.perf_counter()
            poi, slots, lengths, users, targets, _ = [value.to(device) for value in batch]
            synchronize(torch, device); start_model = time.perf_counter()
            with torch.no_grad(): logits = model(poi, slots, lengths, users, targets)
            synchronize(torch, device)
            end_model = time.perf_counter(); start_fusion = end_model
            if args.protocol == "all-prefix":
                scores = logits.float().cpu().numpy(); scores -= scores.max(axis=1, keepdims=True)
                bases = np.exp(scores); bases /= bases.sum(axis=1, keepdims=True)
                if args.variant == "B3-dbn" and weight > 0:
                    for index, query in enumerate(chunk):
                        evidence = transition_prior(query["pois"][query["step"] - 1], transitions, prior, args.smoothing)
                        fuse(bases[index], evidence, weight)
            end_fusion = time.perf_counter(); model_times.append(end_model - start_model)
            fusion_times.append(end_fusion - start_fusion); total_times.append(end_fusion - start_total)
            batch_sizes.append(batch_size); total_queries += batch_size
        print(json.dumps({"repeat": repeat + 1, "repeats": args.repeats, "measured_queries": total_queries}), flush=True)
    if not total_times: raise ValueError("benchmark measured zero batches")
    per_query = np.asarray(total_times) / np.asarray(batch_sizes); total_seconds = float(np.sum(total_times))
    timing = {"latency_mean_seconds": float(np.average(per_query, weights=batch_sizes)),
              "latency_p50_seconds": float(np.quantile(per_query, .5)),
              "latency_p95_seconds": float(np.quantile(per_query, .95)),
              "throughput_queries_per_second": float(total_queries / total_seconds),
              "model_seconds": float(np.sum(model_times)), "fusion_seconds": float(np.sum(fusion_times)),
              "total_seconds": total_seconds, "measured_queries": int(total_queries),
              "batch_latency_per_query_seconds": per_query.tolist()}
    memory = {"rss_before_mb": rss_before, "rss_peak_mb": rss_megabytes(),
              "gpu_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
              "gpu_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 2**20) if device.type == "cuda" else None}
    contention = {json.dumps(row, sort_keys=True): row for row in contention_before + gpu_process_snapshot(device)}
    foreign_processes = list(contention.values())
    if foreign_processes:
        print(json.dumps({"warning": "foreign GPU process detected during RQ12 benchmark",
                          "foreign_gpu_processes": foreign_processes}), flush=True)
    hardware = {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                "torch": torch.__version__, "platform": platform.platform(),
                "foreign_gpu_processes": foreign_processes}
    result = {"rq": "RQ12", "variant": args.variant, "protocol": args.protocol, "seed": args.seed,
              "timing_scope": "prepared-input online inference; checkpoint/data loading and warm-up excluded",
              "batch_size": args.batch_size, "repeats": args.repeats, "warmup_batches": args.warmup_batches,
              "query_limit": args.query_limit, "available_queries": available_queries,
              "timed_unique_queries": len(examples) if args.protocol == "last-query" else len(queries),
              "max_batches": args.max_batches, "parameters": int(sum(value.numel() for value in model.parameters())),
              "quality": quality(args.quality_metrics, args.quality_variant), "timing": timing,
              "memory": memory, "hardware": hardware}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "variant": args.variant, **{k: v for k, v in timing.items() if not isinstance(v, list)}}))
    return result


def main():
    parser = argparse.ArgumentParser(description="RQ12 repeatable online efficiency benchmark")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--test-csv", required=True)
    parser.add_argument("--quality-metrics", required=True); parser.add_argument("--quality-variant")
    parser.add_argument("--output", required=True); parser.add_argument("--variant", required=True)
    parser.add_argument("--protocol", choices=["last-query", "all-prefix"], required=True)
    parser.add_argument("--train-csv"); parser.add_argument("--rq7-metrics")
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5); parser.add_argument("--max-batches", type=int)
    parser.add_argument("--query-limit", type=int, help="deterministic timing sample; quality remains full-test")
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothing", type=float, default=1.0); benchmark(parser.parse_args())


if __name__ == "__main__": main()
