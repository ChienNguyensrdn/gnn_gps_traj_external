from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from .checkpoint_models import build_checkpoint_model
from .dual_evolution import _device
from .evaluate_student import prediction_arrays, summarize_logits
from .neural_cgm import _batches, _slot, _torch, build_examples

VARIANTS = ("unigram", "markov-bigram", "bn-data-only", "dbn-data-only", "quantitative-teacher")

def normalize(values):
    values=np.asarray(values,dtype=float); return values/values.sum()

def fit_statistics(frame, user_map, size, alpha):
    global_c=np.full(size,alpha); users=defaultdict(lambda:np.full(size,alpha)); times=defaultdict(lambda:np.full(size,alpha)); trans=defaultdict(lambda:np.full(size,alpha))
    for _, rows in frame.groupby("trajectory_id",sort=False):
        rows=rows.sort_values("UTC_time",kind="stable"); pois=rows.POI_id.astype(int).tolist(); slots=_slot(rows.UTC_time)
        user=user_map.get(str(rows.iloc[0].user_id),len(user_map))
        for poi,slot in zip(pois,slots): global_c[poi]+=1; users[user][poi]+=1; times[slot][poi]+=1
        for left,right in zip(pois,pois[1:]): trans[left][right]+=1
    return normalize(global_c),users,times,trans

def probability_rows(examples, global_p, users, times, trans, alpha):
    result={name:[] for name in VARIANTS[:-1]}
    for pois,_,user,target_slot,_ in examples:
        user_p=normalize(users[user]) if user in users else global_p; time_p=normalize(times[target_slot]) if target_slot in times else global_p
        transition=normalize(trans[pois[-1]]) if pois[-1] in trans else global_p
        result["unigram"].append(global_p); result["markov-bigram"].append(transition)
        bn=normalize(np.sqrt(np.clip(user_p*time_p,1e-24,None))); result["bn-data-only"].append(bn)
        result["dbn-data-only"].append(normalize(np.cbrt(np.clip(user_p*time_p*transition,1e-36,None))))
    return {key:np.asarray(value) for key,value in result.items()}

def run(args):
    torch=_torch(); ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model,_=build_checkpoint_model(ckpt); model.load_state_dict(ckpt["model_state"])
    device=_device(torch,args.device); model.to(device).eval(); train=pd.read_csv(args.train_csv); test=pd.read_csv(args.test_csv)
    examples=build_examples(test,ckpt["user_map"],False); labels=np.asarray([row[-1] for row in examples]); size=int(ckpt["config"]["num_pois"])
    global_p,users,times,trans=fit_statistics(train,ckpt["user_map"],size,args.alpha); probs=probability_rows(examples,global_p,users,times,trans,args.alpha)
    logits=[]
    with torch.no_grad():
        for batch in _batches(examples,args.batch_size,False,args.seed):
            poi,slots,lengths,user,target,_=[v.to(device) for v in batch]; logits.append(model(poi,slots,lengths,user,target).cpu().numpy())
    logits=np.concatenate(logits); shifted=logits-logits.max(1,keepdims=True); probs["quantitative-teacher"]=np.exp(shifted); probs["quantitative-teacher"]/=probs["quantitative-teacher"].sum(1,keepdims=True)
    root=Path(args.output_root)
    for variant,values in probs.items():
        output=root/variant/f"seed-{args.seed}"; output.mkdir(parents=True,exist_ok=True); arrays=prediction_arrays(np.log(np.clip(values,1e-12,1)),labels)
        payload={"rq":"RQ2","variant":variant,"seed":args.seed,"deterministic":variant!="quantitative-teacher","fit_split":"train","evaluation_split":"test","metrics":summarize_logits(np.log(np.clip(values,1e-12,1)),labels)}
        (output/"rq2.metrics.json").write_text(json.dumps(payload,indent=2)+"\n"); np.savez_compressed(output/"test.predictions.npz",**arrays,query_index=np.arange(len(labels)))
    print(json.dumps({"output":str(root),"variants":list(probs),"queries":len(labels)}))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--train-csv",required=True); p.add_argument("--test-csv",required=True); p.add_argument("--output-root",required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--device",default="auto"); p.add_argument("--alpha",type=float,default=1.0); run(p.parse_args())
if __name__=="__main__": main()
