from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np


METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")
CANONICAL_CITIES = ("Tokyo", "Nairobi", "NewYork", "Sydney", "CapeTown", "Paris", "Beijing",
                    "Mumbai", "SanFrancisco", "London", "SaoPaulo", "Moscow")
RQ9_VARIANTS = ("memory-true", "memory-shuffled", "memory-random-user", "memory-none",
                "context-shuffled", "context-random-poi", "context-none")


def read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def metric_values(payload):
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS if name in source}


def collect_city(root: Path, city: str, seeds: list[int], random_seeds: list[int], model_slug: str, limit: int, scope: str):
    rows = {}; missing = []
    def require_paths(paths):
        missing.extend(str(path) for path in paths if not path.is_file())
    def capture(label, path, nested=()):
        try:
            payload = read(path)
            for key in nested: payload = payload[key]
            values = metric_values(payload)
            if values: rows[label] = values
        except (FileNotFoundError, KeyError) as exc: missing.append(str(getattr(exc, "filename", None) or path))
    if scope in {"all", "neural"}:
        for seed in seeds:
            for variant in ("E0-ce", "E1-kd", "E2-kd-traj", "E3-kd-vel", "E4-layer", "E5-dual"):
                capture(f"RQ4/{variant}/seed-{seed}", root/city/variant/"correct"/f"seed-{seed}"/"test.metrics.json")
            for order in ("reverse", "random"):
                capture(f"RQ5/E5-dual-{order}/seed-{seed}", root/city/"E5-dual"/order/f"seed-{seed}"/"test.metrics.json")
            for variant in ("E1-kd", "E2-kd-traj", "E3-kd-vel", "E4-layer", "E6-temporal", "E5-dual"):
                capture(f"RQ6/{variant}/seed-{seed}", root/city/variant/"correct"/f"seed-{seed}"/"rq6.metrics.json")
            capture(f"RQ2/teacher/seed-{seed}", root/city/"rq2"/"quantitative-teacher"/f"seed-{seed}"/"rq2.metrics.json")
            for variant in ("none", "gru", "transformer"):
                capture(f"RQ10/student-{variant}/seed-{seed}", root/city/"rq10"/"students"/variant/f"seed-{seed}"/"test.metrics.json")
            capture(f"RQ13/clean/seed-{seed}", root/city/"rq13"/"clean"/f"seed-{seed}"/"rq13.metrics.json")
        capture("RQ2/dbn-data-only", root/city/"rq2"/"dbn-data-only"/f"seed-{seeds[0]}"/"rq2.metrics.json")
        require_paths(
            [root/city/variant/"correct"/f"seed-{seed}"/name
             for seed in seeds for variant in ("E0-ce","E1-kd","E2-kd-traj","E3-kd-vel","E4-layer","E5-dual")
             for name in ("best.pt","test.metrics.json","test.predictions.npz")] +
            [root/city/"E5-dual"/order/f"seed-{seed}"/name
             for seed in seeds for order in ("reverse","random")
             for name in ("best.pt","test.metrics.json","test.predictions.npz")] +
            [root/city/variant/"correct"/f"seed-{seed}"/name
             for seed in seeds for variant in ("E1-kd","E2-kd-traj","E3-kd-vel","E4-layer","E6-temporal","E5-dual")
             for name in ("rq6.metrics.json","rq6.predictions.npz")] +
            [root/city/"rq2"/variant/f"seed-{seed}"/name
             for seed in seeds for variant in ("unigram","markov-bigram","bn-data-only","dbn-data-only","quantitative-teacher")
             for name in ("rq2.metrics.json","test.predictions.npz")] +
            [root/city/"rq10"/group/variant/f"seed-{seed}"/name
             for seed in seeds for group,variants in (("teachers",("gru","transformer")),("students",("none","gru","transformer")))
             for variant in variants for name in ("best.pt","test.metrics.json","test.predictions.npz")] +
            [root/city/"rq13"/variant/f"seed-{seed}"/name
             for seed in seeds for variant in ("clean","gps-drop-25","gps-drop-50","time-noise-30m","time-noise-60m",
                                                "position-noise-200m","position-noise-500m","context-missing-user",
                                                "context-missing-time","context-wrong-user","context-wrong-time",
                                                "context-missing","context-wrong")
             for name in ("rq13.metrics.json","test.predictions.npz")])
    if scope in {"all", "bayesian"}:
        for seed in seeds:
            rq7=root/city/"E5-dual"/"correct"/f"seed-{seed}"/"rq7"/"rq7.metrics.json"
            for variant in ("B0-static", "B3-dbn"):
                capture(f"RQ7/{variant}/seed-{seed}", rq7, ("test_metrics", variant))
            for variant in ("B0-static", "B3-dbn"):
                capture(f"RQ11/{variant}/seed-{seed}", root/city/"rq11"/"bayesian"/variant/f"seed-{seed}"/"rq11.metrics.json",
                        ("metrics", "identity"))
        require_paths(
            [root/city/"E5-dual"/"correct"/f"seed-{seed}"/"rq7"/name
             for seed in seeds for name in ("rq7.metrics.json","B0-static.test.predictions.npz","B1-history.test.predictions.npz",
                                             "B2-sequential.test.predictions.npz","B3-dbn.test.predictions.npz")] +
            [root/city/"rq11"/group/variant/f"seed-{seed}"/name
             for seed in seeds for group,variants in (("distillation",("none","gru","transformer")),
                                                       ("bayesian",("B0-static","B3-dbn")))
             for variant in variants for name in ("rq11.metrics.json","identity.predictions.npz","nll.predictions.npz",
                                                   "brier.predictions.npz","ece.predictions.npz")])
    if scope in {"all", "efficiency"}:
        for seed in seeds:
            for profile in ("batch-1", "batch-256"):
                path=root/city/"rq12"/profile/"neural"/"student-gru"/f"seed-{seed}"/"rq12.metrics.json"
                try:
                    payload=read(path)
                    if payload.get("hardware",{}).get("foreign_gpu_processes"):
                        missing.append(f"GPU_CONTENTION:{path}")
                    rows[f"RQ12/{profile}/student-gru/seed-{seed}"]={
                        "recall@1":float(payload["quality"]["recall@1"]),
                        "throughput_queries_per_second":float(payload["timing"]["throughput_queries_per_second"])}
                except (FileNotFoundError, KeyError): missing.append(str(path))
        require_paths([root/city/"rq12"/profile/group/variant/f"seed-{seed}"/"rq12.metrics.json"
                       for seed in seeds for profile in ("batch-1","batch-256")
                       for group,variants in (("neural",("teacher-gru","teacher-transformer","student-none","student-gru","student-transformer")),
                                              ("bayesian",("B0-static","B3-dbn"))) for variant in variants])
    if scope in {"all", "llm"}:
        rq3=root/city/"rq3"/model_slug/f"limit-{limit}"/f"seed-{seeds[0]}"/"rq3.metrics.json"
        for variant in ("M1-data-only", "M2-llm", "M3-quantitative", "M4-both"):
            capture(f"RQ3/{variant}", rq3, ("test_metrics", variant))
        rq8=root/city/"rq8"/model_slug/f"limit-{limit}"/f"seed-{seeds[0]}"/"rq8.metrics.json"
        for policy in ("never", "always", "entropy", "margin", "random-budget-matched"):
            capture(f"RQ8/{policy}", rq8, ("metrics", policy))
        for variant in RQ9_VARIANTS:
            path=root/city/"rq9"/model_slug/f"limit-{limit}"/variant/"metrics.json"
            try:
                payload=read(path); rows[f"RQ9/{variant}"]={
                    "recall@1":float(payload["acc@1"]),"recall@5":float(payload["acc@5"]),
                    "recall@10":float(payload["acc@10"]),"mrr":float(payload["mrr"])}
            except (FileNotFoundError, KeyError): missing.append(str(path))
        rq3_root=root/city/"rq3"/model_slug/f"limit-{limit}"/f"seed-{seeds[0]}"
        rq8_root=root/city/"rq8"/model_slug/f"limit-{limit}"
        rq9_root=root/city/"rq9"/model_slug/f"limit-{limit}"
        require_paths(
            [rq3_root/"rq3.metrics.json"] + [rq3_root/f"{variant}.test.predictions.npz" for variant in ("M1-data-only","M2-llm","M3-quantitative","M4-both")] +
            [rq8_root/f"seed-{seed}"/name for seed in random_seeds
             for name in ("rq8.metrics.json","never.test.predictions.npz","always.test.predictions.npz",
                          "entropy.test.predictions.npz","margin.test.predictions.npz","random-budget-matched.test.predictions.npz")] +
            [rq9_root/variant/name for variant in RQ9_VARIANTS for name in ("metrics.json","predictions.jsonl")])
    return rows, sorted(set(missing))


def aggregate(cities, city_rows):
    labels=sorted(set.intersection(*(set(city_rows[city]) for city in cities))) if cities else []
    result={}
    for label in labels:
        metrics=sorted(set.intersection(*(set(city_rows[city][label]) for city in cities)))
        result[label]={metric:{"macro_mean":float(np.mean([city_rows[city][label][metric] for city in cities])),
                               "city_population_variance":float(np.var([city_rows[city][label][metric] for city in cities]))}
                       for metric in metrics}
    return result


def summarize_seeds(macro):
    grouped={}
    for label,metrics in macro.items():
        base=re.sub(r"/seed-\d+$","",label)
        for metric,value in metrics.items(): grouped.setdefault(base,{}).setdefault(metric,[]).append(value)
    return {label:{metric:{"mean":float(np.mean([row["macro_mean"] for row in values])),
                           "std":statistics.stdev([row["macro_mean"] for row in values]) if len(values)>1 else None,
                           "city_population_variance_mean":float(np.mean([row["city_population_variance"] for row in values])),
                           "runs":len(values)}
                   for metric,values in metrics.items()} for label,metrics in grouped.items()}


def render(payload):
    lines=["# TIST2015 — Tổng hợp thực nghiệm 12 thành phố","",
           f"> Scope: `{payload['scope']}`. Gate: **{payload['gate']}**.","",
           "## Trạng thái thành phố","","| City | Rows ready | Missing artifacts |","|---|---:|---:|"]
    for city in payload["cities"]:
        lines.append(f"| {city} | {len(payload['per_city'][city])} | {len(payload['missing'][city])} |")
    lines += ["","## Macro metrics qua city và seed","","| Experiment | Metric | Macro mean ± seed std | Mean city variance | Runs |","|---|---|---:|---:|---:|"]
    for label,row in payload["macro_seed_summary"].items():
        for metric,value in row.items():
            mean=f"{value['mean']:.6f}"; std=value["std"]
            if std is not None: mean += f" ± {std:.6f}"
            lines.append(f"| {label} | {metric} | {mean} | {value['city_population_variance_mean']:.6g} | {value['runs']} |")
    if payload["gate"] != "ready-12city":
        lines += ["","## Publication gate","",
                  "Không được gọi đây là 12-city average: còn artifact thiếu hoặc scope chưa đồng nhất.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description="Strict 12-city RQ artifact gate and macro aggregation")
    parser.add_argument("--results-root",type=Path,required=True); parser.add_argument("--cities",nargs="+",required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44]); parser.add_argument("--model-slug",default="qwen2-7b")
    parser.add_argument("--random-seeds",nargs="+",type=int,default=list(range(42,92)))
    parser.add_argument("--limit",type=int,default=200); parser.add_argument("--scope",choices=["all","neural","bayesian","efficiency","llm"],default="all")
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--markdown",type=Path,required=True)
    parser.add_argument("--allow-incomplete",action="store_true"); args=parser.parse_args()
    city_rows={}; missing={}
    for city in args.cities:
        city_rows[city],missing[city]=collect_city(args.results_root,city,args.seeds,args.random_seeds,args.model_slug,args.limit,args.scope)
    macro=aggregate(args.cities,city_rows)
    complete=all(not missing[city] for city in args.cities) and tuple(args.cities)==CANONICAL_CITIES
    payload={"rq":"RQ1-RQ13","scope":args.scope,"cities":args.cities,"seeds":args.seeds,"model_slug":args.model_slug,
             "limit":args.limit,"llm_world_mode":"no-OSM","per_city":city_rows,"missing":missing,"macro":macro,
             "macro_seed_summary":summarize_seeds(macro),
             "gate":"ready-12city" if complete else "incomplete"}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n")
    args.markdown.parent.mkdir(parents=True,exist_ok=True); args.markdown.write_text(render(payload))
    print(json.dumps({"output":str(args.output),"gate":payload["gate"],"missing":sum(map(len,missing.values()))}))
    if not complete and not args.allow_incomplete: raise SystemExit(2)


if __name__ == "__main__": main()
