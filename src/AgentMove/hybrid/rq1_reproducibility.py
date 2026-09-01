from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

QUANTITATIVE = (("teacher-gru", "teachers/gru"), ("teacher-transformer", "teachers/transformer"),
                ("student-ce", "students/none"))
METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")


def summarize(values):
    values = [float(value) for value in values]
    return {"mean": float(np.mean(values)), "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def load_quantitative(root: Path, seeds: list[int]):
    output = {}
    for name, relative in QUANTITATIVE:
        rows = []
        for seed in seeds:
            path = root / relative / f"seed-{seed}" / "test.metrics.json"
            if not path.is_file(): raise FileNotFoundError(f"missing matched quantitative baseline: {path}")
            rows.append(json.loads(path.read_text(encoding="utf-8"))["metrics"])
        output[name] = {metric: summarize([row[metric] for row in rows]) for metric in METRICS}
    return output


def optional_baseline(path: Path, expected_limit: int):
    if not path.is_file(): return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8")); complete = bool(payload.get("is_complete_12_city"))
    limit_ok = payload.get("query_limit") == expected_limit
    return {"status": "ready-bounded" if complete and limit_ok else "incomplete-or-incompatible",
            "path": str(path), "is_complete_12_city": complete, "query_limit": payload.get("query_limit"),
            "macro_average": payload.get("macro_average"), "population_variance_acc1": payload.get("population_variance_acc1"),
            "protocol": payload.get("protocol")}


def render(payload):
    lines = ["# RQ1 — Baseline reproducibility", "",
             "> Báo cáo phân biệt matched Tokyo full-test và baseline TIST2015 bounded 12-city; không so trực tiếp trị tuyệt đối giữa hai protocol.", "",
             "## Quantitative baselines — Tokyo matched full-test", "",
             f"Seeds: {', '.join(map(str, payload['seeds']))}. Cùng preprocessing, split, candidate space và checkpoint selection bằng validation.", "",
             "| Baseline | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in payload["quantitative"].items():
        fmt = lambda key: f"{row[key]['mean']:.6f} ± {row[key]['std']:.6f}"
        lines.append(f"| {name} | {fmt('recall@1')} | {fmt('recall@5')} | {fmt('recall@10')} | {fmt('mrr')} | {fmt('nll')} | {fmt('brier')} | {fmt('ece')} |")
    lines += ["", "## TIST2015 bounded baseline gate", "",
              "| Baseline | Status | 12-city | Limit | Macro metrics |",
              "|---|---|---|---:|---|"]
    for name in ("markov-bigram", "agentmove-original"):
        row = payload["bounded_baselines"][name]; macro = row.get("macro_average")
        rendered = "N/A" if not macro else ", ".join(f"{key}={value:.6f}" for key, value in macro.items())
        lines.append(f"| {name} | {row['status']} | {row.get('is_complete_12_city', False)} | {row.get('query_limit', 'N/A')} | {rendered} |")
    lines += ["", "## Publication gate", "",
              f"- Quantitative matched multi-seed: **{'ready' if payload['gates']['quantitative_matched_ready'] else 'not ready'}**.",
              f"- Markov bounded 12-city: **{payload['bounded_baselines']['markov-bigram']['status']}**.",
              f"- AgentMove bounded 12-city: **{payload['bounded_baselines']['agentmove-original']['status']}**.",
              "- Markov là deterministic baseline nên không pseudo-replicate theo seed.",
              "- AgentMove bounded và quantitative Tokyo full-test không dùng cùng query protocol; không tính paired delta giữa chúng.",
              "- RQ1 chỉ được gọi hoàn thành toàn phần khi AgentMove/quantitative baseline cùng matched sample hoặc được báo cáo thành các protocol riêng như ở đây.",
              "- Việc kiểm tra RQ1 được thực hiện hồi cứu sau các RQ sau; báo cáo phải nêu rõ thay vì khẳng định baseline đã được khóa trước phát triển.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RQ1 baseline reproducibility audit and aggregate")
    parser.add_argument("--rq10-root", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--markov-summary", type=Path, required=True); parser.add_argument("--agentmove-summary", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=200); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True); args = parser.parse_args()
    quantitative = load_quantitative(args.rq10_root, args.seeds)
    bounded = {"markov-bigram": optional_baseline(args.markov_summary, args.query_limit),
               "agentmove-original": optional_baseline(args.agentmove_summary, args.query_limit)}
    payload = {"rq": "RQ1", "seeds": args.seeds, "quantitative": quantitative, "bounded_baselines": bounded,
               "gates": {"quantitative_matched_ready": True,
                         "bounded_12city_ready": all(row["status"] == "ready-bounded" for row in bounded.values()),
                         "direct_cross_protocol_comparison_allowed": False}}
    payload["gate"] = "ready-separated-protocols" if payload["gates"]["bounded_12city_ready"] else "partial-missing-bounded-baselines"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown), "gate": payload["gate"]}))


if __name__ == "__main__": main()
