from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from .checkpoint_models import build_checkpoint_model
from .dual_evolution import _device
from .neural_cgm import _batches,_slot,_torch,build_examples
from .rq7_belief_memory import summarize_arrays

VARIANTS=("unigram","markov-bigram","bn-data-only","dbn-data-only","quantitative-teacher")
def normalize(v): v=np.asarray(v,dtype=float); return v/v.sum()
def fit_statistics(frame,user_map,size,alpha):
 global_c=np.full(size,alpha); users=defaultdict(Counter); times=defaultdict(Counter); trans=defaultdict(Counter)
 for _,rows in frame.groupby("trajectory_id",sort=False):
  rows=rows.sort_values("UTC_time",kind="stable"); pois=rows.POI_id.astype(int).tolist(); slots=_slot(rows.UTC_time); user=user_map.get(str(rows.iloc[0].user_id),len(user_map))
  for poi,slot in zip(pois,slots): global_c[poi]+=1; users[user][poi]+=1; times[slot][poi]+=1
  for left,right in zip(pois,pois[1:]): trans[left][right]+=1
 return normalize(global_c),users,times,trans
def sparse_prior(counts,size,alpha,fallback):
 if not counts:return fallback
 row=np.full(size,alpha,dtype=float)
 for key,value in counts.items(): row[key]+=value
 return normalize(row)
def probability_rows(examples,global_p,users,times,trans,alpha):
 size=len(global_p); result={name:[] for name in VARIANTS[:-1]}
 for pois,_,user,target_slot,_ in examples:
  up=sparse_prior(users.get(user),size,alpha,global_p); tp=sparse_prior(times.get(target_slot),size,alpha,global_p); tr=sparse_prior(trans.get(pois[-1]),size,alpha,global_p)
  result["unigram"].append(global_p); result["markov-bigram"].append(tr); result["bn-data-only"].append(normalize(np.sqrt(up*tp))); result["dbn-data-only"].append(normalize(np.cbrt(up*tp*tr)))
 return {k:np.asarray(v) for k,v in result.items()}
def empty_arrays(n):
 return {"labels":np.empty(n,np.int64),"top1":np.empty(n,np.int64),"ranks":np.empty(n,np.int32),"reciprocal_rank":np.empty(n,np.float32),"true_probability":np.empty(n,np.float32),"confidence":np.empty(n,np.float32),"top1_correct":np.empty(n,np.int8),"brier":np.empty(n,np.float32)}
def fill(arrays,index,p,label):
 top=int(np.argmax(p)); true=float(p[label]); rank=1+int(np.count_nonzero(p>true))+int(np.count_nonzero(p[:label]==true)); arrays["labels"][index]=label; arrays["top1"][index]=top; arrays["ranks"][index]=rank; arrays["reciprocal_rank"][index]=1/rank; arrays["true_probability"][index]=true; arrays["confidence"][index]=p[top]; arrays["top1_correct"][index]=top==label; arrays["brier"][index]=np.dot(p,p)-2*true+1
def run(args):
 torch=_torch(); ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model,_=build_checkpoint_model(ckpt); model.load_state_dict(ckpt["model_state"]); device=_device(torch,args.device); model.to(device).eval()
 train=pd.read_csv(args.train_csv); test=pd.read_csv(args.test_csv); examples=build_examples(test,ckpt["user_map"],False); size=int(ckpt["config"]["num_pois"]); gp,users,times,trans=fit_statistics(train,ckpt["user_map"],size,args.alpha); del train,test
 outputs={v:empty_arrays(len(examples)) for v in VARIANTS}; offset=0
 with torch.no_grad():
  for batch in _batches(examples,args.batch_size,False,args.seed):
   poi,slots,lengths,user,target,labels=[v.to(device) for v in batch]; logits=model(poi,slots,lengths,user,target).cpu().numpy(); shifted=logits-logits.max(1,keepdims=True); teacher=np.exp(shifted); teacher/=teacher.sum(1,keepdims=True)
   chunk=examples[offset:offset+len(labels)]; classical=probability_rows(chunk,gp,users,times,trans,args.alpha)
   for local,label in enumerate(labels.cpu().numpy()):
    for variant in VARIANTS[:-1]: fill(outputs[variant],offset+local,classical[variant][local],int(label))
    fill(outputs["quantitative-teacher"],offset+local,teacher[local],int(label))
   offset+=len(labels); print(json.dumps({"processed":offset,"queries":len(examples)}),flush=True)
 root=Path(args.output_root)
 for variant,arrays in outputs.items():
  out=root/variant/f"seed-{args.seed}"; out.mkdir(parents=True,exist_ok=True); payload={"rq":"RQ2","variant":variant,"seed":args.seed,"deterministic":variant!="quantitative-teacher","fit_split":"train","evaluation_split":"test","metrics":summarize_arrays(arrays)}; (out/"rq2.metrics.json").write_text(json.dumps(payload,indent=2)+"\n"); np.savez_compressed(out/"test.predictions.npz",**arrays,query_index=np.arange(len(examples)))
 print(json.dumps({"output":str(root),"queries":len(examples)}))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--train-csv",required=True); p.add_argument("--test-csv",required=True); p.add_argument("--output-root",required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--device",default="auto"); p.add_argument("--alpha",type=float,default=1.0); run(p.parse_args())
if __name__=="__main__":main()
