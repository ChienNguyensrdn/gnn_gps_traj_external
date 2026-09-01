from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import numpy as np
from .paired_order_test import METRICS,bootstrap_and_permutation_many,holm_adjust,load_npz,paired_differences
from .rq2_data_only import VARIANTS

SUMMARY=("recall@1","recall@5","recall@10","mrr","nll","brier","ece")
def summarize(v): return {"mean":float(np.mean(v)),"std":statistics.stdev(v) if len(v)>1 else None}
def paired_comparisons(root: Path, seeds: list[int], iterations: int):
 tests=[]; deterministic_seed=seeds[0]
 specs=[
  (f"quantitative-teacher-vs-{variant}",
   [(root/"quantitative-teacher"/f"seed-{seed}"/"test.predictions.npz",
     root/variant/f"seed-{deterministic_seed}"/"test.predictions.npz") for seed in seeds])
  for variant in VARIANTS[:-1]
 ]
 specs.append(("dbn-data-only-vs-bn-data-only", [
  (root/"dbn-data-only"/f"seed-{deterministic_seed}"/"test.predictions.npz",
   root/"bn-data-only"/f"seed-{deterministic_seed}"/"test.predictions.npz")
 ]))
 for comparison_index,(comparison,pairs) in enumerate(specs):
  diffs=[]
  for left_path,right_path in pairs:
   left=load_npz(left_path); right=load_npz(right_path)
   diffs.append(np.column_stack([paired_differences(left,right,m) for m in METRICS]))
  eff,ci,pv=bootstrap_and_permutation_many(diffs,iterations,42+comparison_index)
  for j,m in enumerate(METRICS):
   tests.append({"comparison":comparison,"metric":m,"effect":float(eff[j]),
                 "ci95":ci[j].tolist(),"p":float(pv[j]),
                 "replicates":len(pairs)})
 adj=holm_adjust([r["p"] for r in tests])
 for r,x in zip(tests,adj): r["holm_p"]=x; r["significant"]=x<.05
 return tests
def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44]); p.add_argument("--iterations",type=int,default=10000); p.add_argument("--output",type=Path,required=True); p.add_argument("--markdown",type=Path,required=True); a=p.parse_args(); variants={}; tests=[]
 for v in VARIANTS:
  seeds=a.seeds if v=="quantitative-teacher" else [a.seeds[0]]
  rows=[json.loads((a.root/v/f"seed-{s}"/"rq2.metrics.json").read_text())["metrics"] for s in seeds]
  variants[v]={m:summarize([r[m] for r in rows]) for m in SUMMARY}
 tests=paired_comparisons(a.root,a.seeds,a.iterations)
 payload={"rq":"RQ2","seeds":a.seeds,"variants":variants,"paired_tests":tests,"gate":"ready-tokyo-matched-last-query"}
 lines=["# RQ2 — Bayesian student data-only","","> Mọi prior/transition chỉ fit train; test dùng last-query matched. Baseline deterministic chỉ tính một run.","","| Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |","|---|---:|---:|---:|---:|---:|---:|---:|"]
 for v,row in variants.items():
  f=lambda m: f"{row[m]['mean']:.6f}"+(f" ± {row[m]['std']:.6f}" if row[m]['std'] is not None else "")
  lines.append(f"| {v} | {f('recall@1')} | {f('recall@5')} | {f('recall@10')} | {f('mrr')} | {f('nll')} | {f('brier')} | {f('ece')} |")
 lines += ["","## Paired comparisons","","> Positive effect nghĩa là variant đứng trước tốt hơn; NLL/Brier đã đảo dấu. Holm correction áp dụng cho toàn bộ phép kiểm định.","","| Comparison | Metric | Effect favoring first | 95% CI | Holm p | Significant |","|---|---|---:|---:|---:|---|"]
 for r in tests: lines.append(f"| {r['comparison']} | {r['metric']} | {r['effect']:.6f} | {r['ci95'][0]:.6f}–{r['ci95'][1]:.6f} | {r['holm_p']:.6g} | {'yes' if r['significant'] else 'no'} |")
 lines += ["","## Giới hạn","","- BN là geometric fusion của user và target-time empirical priors; DBN bổ sung first-order transition prior.","- DBN-vs-BN dùng một deterministic paired run; không pseudo-replicate theo teacher seeds.","- Đây là categorical POI data-only baseline, không dùng LLM/OSM.","- Kết quả hiện chỉ áp dụng Tokyo; deterministic baselines không pseudo-replicate theo seed.",""]
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n"); a.markdown.parent.mkdir(parents=True,exist_ok=True); a.markdown.write_text("\n".join(lines)); print(json.dumps({"output":str(a.output),"gate":payload["gate"]}))
if __name__=="__main__": main()
