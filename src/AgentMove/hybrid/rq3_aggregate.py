from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_order_test import METRICS, bootstrap_and_permutation_many, holm_adjust, load_npz, paired_differences
from .rq3_distillation import VARIANTS


COMPARISONS = (("M2-llm", "M1-data-only"), ("M3-quantitative", "M1-data-only"),
               ("M4-both", "M1-data-only"), ("M4-both", "M2-llm"),
               ("M4-both", "M3-quantitative"))


def aggregate(root: Path, iterations: int):
    run = json.loads((root / "rq3.metrics.json").read_text()); rows=[]
    for comparison_index,(left_name,right_name) in enumerate(COMPARISONS):
        left=load_npz(root/f"{left_name}.test.predictions.npz")
        right=load_npz(root/f"{right_name}.test.predictions.npz")
        differences=[np.column_stack([paired_differences(left,right,metric) for metric in METRICS])]
        effects,intervals,p_values=bootstrap_and_permutation_many(differences,iterations,42+comparison_index)
        for metric_index,metric in enumerate(METRICS):
            rows.append({"comparison":f"{left_name}-vs-{right_name}","metric":metric,
                         "effect":float(effects[metric_index]),"ci95":intervals[metric_index].tolist(),
                         "p":float(p_values[metric_index])})
    adjusted=holm_adjust([row["p"] for row in rows])
    for row,value in zip(rows,adjusted): row["holm_p"]=value; row["significant"]=value<.05
    return {"rq":"RQ3","protocol":run["protocol"],"city":run["city"],"limit":run["limit"],
            "selected_weights":run["selected_weights"],"test_metrics":run["test_metrics"],
            "paired_tests":rows,"gate":"ready-bounded-matched"}


def markdown(payload):
    lines=["# RQ3 — LLM knowledge distillation","",
           "> Trọng số fit trên validation; quality và paired tests báo cáo trên bounded matched test.","",
           "| Variant | q-weight | LLM-weight | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        metrics=payload["test_metrics"][variant]; weights=payload["selected_weights"][variant]
        lines.append(f"| {variant} | {weights['quantitative']:.2f} | {weights['llm']:.2f} | "
                     f"{metrics['recall@1']:.6f} | {metrics['recall@5']:.6f} | {metrics['recall@10']:.6f} | "
                     f"{metrics['mrr']:.6f} | {metrics['nll']:.6f} | {metrics['brier']:.6f} | {metrics['ece']:.6f} |")
    lines += ["","## Paired comparisons","",
              "> Positive effect nghĩa là variant đứng trước tốt hơn; NLL/Brier đã đảo dấu. Holm correction áp dụng toàn bộ phép kiểm định.","",
              "| Comparison | Metric | Effect | 95% CI | Holm p | Significant |",
              "|---|---|---:|---:|---:|---|"]
    for row in payload["paired_tests"]:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['effect']:.6f} | "
                     f"{row['ci95'][0]:.6f}–{row['ci95'][1]:.6f} | {row['holm_p']:.6g} | "
                     f"{'yes' if row['significant'] else 'no'} |")
    lines += ["","## Protocol gate","",
              "- Prior data-only fit train; fusion weights chọn bằng validation; test không dùng để tuning.",
              "- Structured LLM evidence được replay từ immutable cache; không gọi lại LLM trong evaluation.",
              "- Cache phải phủ đủ mọi query; evidence ngoài cached quantitative top-k dùng likelihood trung tính.",
              "- Đây là bounded matched experiment và không được gọi là full-query hoặc 12-city result.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--iterations",type=int,default=10000); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--markdown",type=Path,required=True); args=parser.parse_args()
    payload=aggregate(args.root,args.iterations); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n"); args.markdown.parent.mkdir(parents=True,exist_ok=True)
    args.markdown.write_text(markdown(payload)); print(json.dumps({"output":str(args.output),"gate":payload["gate"]}))


if __name__ == "__main__": main()
