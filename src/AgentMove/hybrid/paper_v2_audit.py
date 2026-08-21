from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .io import read_jsonl, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--test",required=True);p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--min-recall",type=float,default=.45);p.add_argument("--min-osm-coverage",type=float,default=.9);p.add_argument("--output")
    a=p.parse_args(); rows=list(read_jsonl(a.test));bundle=rows.pop(0)["_bundle"]
    logits=np.load(bundle["logits"],mmap_mode="r");ids=[str(x) for x in json.loads(Path(bundle["candidate_ids"]).read_text())];ix={x:i for i,x in enumerate(ids)}
    hits=0
    for row in rows:
        top=np.argpartition(-logits[int(row["_row_index"])],min(a.top_k,len(ids))-1)[:a.top_k]
        hits+=ix[str(row["true_id"])] in set(top.tolist())
    recall=hits/len(rows)
    metadata=json.loads(Path(bundle["candidate_metadata"]).read_text());covered=sum(bool((metadata.get(i,{}).get("address") or "").strip()) for i in ids);coverage=covered/len(ids)
    report={"queries":len(rows),"top_k":a.top_k,"candidate_recall":recall,"minimum_recall":a.min_recall,
            "osm":{"covered":covered,"total":len(ids),"coverage":coverage,"minimum":a.min_osm_coverage},
            "pass":recall>=a.min_recall and coverage>=a.min_osm_coverage}
    if a.output:write_json(a.output,report)
    print(json.dumps(report,indent=2))
    if recall<a.min_recall: print("BLOCKED: improve/retrain CGM or tune k before LLM extraction.")
    if coverage<a.min_osm_coverage: print("BLOCKED: enrich candidate metadata with OSM before the paper run.")
    raise SystemExit(0 if report["pass"] else 2)
if __name__=="__main__":main()
