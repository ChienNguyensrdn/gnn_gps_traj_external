from __future__ import annotations

import argparse, collections, json
from pathlib import Path
import numpy as np
from .io import read_jsonl, write_json


def _load(path: Path):
    rows=list(read_jsonl(path)); bundle=rows.pop(0)["_bundle"]
    return bundle, rows, np.load(bundle["logits"], mmap_mode="r"), [str(x) for x in json.loads(Path(bundle["candidate_ids"]).read_text())]


def _augment(rows, logits, ids, weight):
    index={x:i for i,x in enumerate(ids)}; output=[]
    for row in rows:
        scores=np.array(logits[int(row["_row_index"])], dtype=np.float32)
        counts=collections.Counter(str(event[-1]) for event in row.get("history",[])+row.get("context",[]) if event)
        for candidate,count in counts.items():
            if candidate in index: scores[index[candidate]] += weight*np.log1p(count)
        output.append(scores)
    return np.stack(output)


def _metrics(rows, matrix, ids):
    index={x:i for i,x in enumerate(ids)}; ranks=[]
    for pos,row in enumerate(rows):
        order=np.argsort(-matrix[pos],kind="stable"); ranks.append(int(np.where(order==index[str(row["true_id"])])[0][0])+1)
    return {f"recall@{k}":float(np.mean([r<=k for r in ranks])) for k in (1,5,10,20,50,100)}


def _write(source_bundle, rows, matrix, output, ids):
    output.mkdir(parents=True,exist_ok=True); logits_path=output/("validation_logits.npy" if rows[0]["query_id"].startswith("validation:") else "test_logits.npy")
    np.save(logits_path,matrix); jsonl=output/("validation.jsonl" if rows[0]["query_id"].startswith("validation:") else "test.jsonl")
    bundle=dict(source_bundle);bundle["logits"]=str(logits_path.resolve())
    with jsonl.open("w") as f:
        f.write(json.dumps({"_bundle":bundle})+"\n")
        for i,row in enumerate(rows): row=dict(row);row["_row_index"]=i;f.write(json.dumps(row,ensure_ascii=False)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--validation",required=True);p.add_argument("--test",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--weights",nargs="+",type=float,default=[0,0.5,1,2,3,5,8,10,15,20]);a=p.parse_args()
    vb,vr,vz,ids=_load(Path(a.validation)); best=None
    for weight in a.weights:
        matrix=_augment(vr,vz,ids,weight); metrics=_metrics(vr,matrix,ids); key=(metrics["recall@1"],metrics["recall@10"])
        if best is None or key>best[0]: best=(key,weight,matrix,metrics)
    tb,tr,tz,test_ids=_load(Path(a.test)); assert ids==test_ids
    test_matrix=_augment(tr,tz,ids,best[1]); test_metrics=_metrics(tr,test_matrix,ids); out=Path(a.output_dir)
    _write(vb,vr,best[2],out,ids);_write(tb,tr,test_matrix,out,ids)
    write_json(out/"cgm_augmentation.json",{"method":"neural logits + validation-tuned log-frequency history prior","weight":best[1],"validation":best[3],"test":test_metrics})
    print(json.dumps({"weight":best[1],"validation":best[3],"test":test_metrics},indent=2))
if __name__=="__main__":main()
