from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from .io import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--metadata",required=True);a=p.parse_args()
    frame=pd.read_csv(a.csv,dtype={"venue_id":str});metadata=json.loads(Path(a.metadata).read_text());updated=set()
    columns=["poi","street","subdistrict","admin"]
    for venue_id,rows in frame.groupby("venue_id",sort=False):
        if str(venue_id) not in metadata:continue
        row=rows.iloc[0];parts=[str(row.get(c,"")) for c in columns if pd.notna(row.get(c)) and str(row.get(c,""))]
        item=metadata[str(venue_id)];item["address"]="; ".join(parts)
        item["osm_category"]=str(row.get("osm_category","") or "");item["osm_type"]=str(row.get("osm_type","") or "")
        updated.add(str(venue_id))
    write_json(a.metadata,metadata);covered=sum(bool((v.get("address") or "").strip()) for v in metadata.values())
    print(json.dumps({"matched":len(updated),"covered":covered,"total":len(metadata),"coverage":covered/len(metadata)},indent=2))
if __name__=="__main__":main()
