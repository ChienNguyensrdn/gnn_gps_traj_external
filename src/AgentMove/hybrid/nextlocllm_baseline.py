from __future__ import annotations

import argparse, hashlib, json, math, random, urllib.request
from urllib.error import HTTPError
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def torch_module():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required") from exc


def load_csv(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing prepared split: {path}")
    frame = pd.read_csv(path)
    needed = {"user_id", "POI_id", "POI_catname", "latitude", "longitude", "trajectory_id", "norm_in_day_time"}
    if needed - set(frame): raise ValueError(f"{path}: missing {sorted(needed-set(frame))}")
    return frame.sort_values(["trajectory_id", "UTC_time"], kind="stable")


def ollama_embed(texts, model, base_url, batch_size=32):
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"): endpoint = endpoint[:-3]

    def post(url, payload):
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    vectors = []
    try:
        for start in range(0, len(texts), batch_size):
            body = post(endpoint + "/api/embed", {"model": model, "input": texts[start:start+batch_size]})
            vectors.extend(body.get("embeddings", []))
        if len(vectors) != len(texts):
            raise ValueError("batch endpoint returned an incomplete embedding list")
        api = "api/embed"
    except (HTTPError, ValueError) as modern_error:
        # Ollama before 0.1.26 exposes only the singular legacy endpoint.
        print(f"embedding_api=/api/embed unavailable ({modern_error}); falling back to /api/embeddings", flush=True)
        vectors = []
        try:
            for text in texts:
                body = post(endpoint + "/api/embeddings", {"model": model, "prompt": text})
                vector = body.get("embedding")
                if not vector: raise ValueError("legacy endpoint returned no embedding")
                vectors.append(vector)
            api = "api/embeddings"
        except (HTTPError, ValueError) as legacy_error:
            print(f"embedding_api=/api/embeddings unavailable ({legacy_error}); falling back to /v1/embeddings", flush=True)
            vectors = []
            try:
                for start in range(0, len(texts), batch_size):
                    body = post(endpoint + "/v1/embeddings", {"model": model, "input": texts[start:start+batch_size]})
                    data = sorted(body.get("data", []), key=lambda row: row.get("index", 0))
                    vectors.extend(row["embedding"] for row in data)
                api = "v1/embeddings"
            except Exception as openai_error:
                raise RuntimeError(
                    f"No supported Ollama embedding endpoint at {endpoint} for model {model}; "
                    f"modern={modern_error}; legacy={legacy_error}; openai={openai_error}. "
                    "Install an embedding-capable Ollama model if qwen2:7b is rejected."
                ) from openai_error
    if len(vectors) != len(texts): raise ValueError(f"Expected {len(texts)} embeddings, received {len(vectors)}")
    print(f"embedding_api={api} vectors={len(vectors)}", flush=True)
    array = np.asarray(vectors, dtype=np.float32)
    array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)
    return array


def semantic_cache(frames, path, model, base_url):
    categories = sorted(set(pd.concat([f.POI_catname.fillna("").astype(str) for f in frames]).tolist()))
    destination = Path(path)
    signature = hashlib.sha256((model + "\n" + "\n".join(categories)).encode()).hexdigest()
    if destination.is_file():
        payload = json.loads(destination.read_text())
        if payload.get("signature") == signature:
            return {key: np.asarray(value, dtype=np.float32) for key, value in payload["vectors"].items()}
    print(f"embedding_categories={len(categories)} model={model} endpoint={base_url}", flush=True)
    values = ollama_embed(categories, model, base_url)
    payload = {"model": model, "signature": signature, "vectors": {key: value.tolist() for key, value in zip(categories, values)}}
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(payload) + "\n")
    return {key: value for key, value in zip(categories, values)}


def poi_features(frames, count, semantic):
    frame = pd.concat(frames).drop_duplicates("POI_id", keep="first")
    coords = np.zeros((count, 2), dtype=np.float32); dim = len(next(iter(semantic.values())))
    semantics = np.zeros((count, dim), dtype=np.float32)
    for row in frame.itertuples(index=False):
        poi = int(row.POI_id)
        if 0 <= poi < count:
            coords[poi] = (float(row.latitude), float(row.longitude)); semantics[poi] = semantic[str(row.POI_catname) if not pd.isna(row.POI_catname) else ""]
    valid = np.any(coords != 0, axis=1)
    if not valid.all(): raise ValueError(f"Missing coordinates for {int((~valid).sum())} candidates")
    low, high = coords.min(0), coords.max(0); scale = np.maximum(high-low, 1e-6)
    return (coords-low)/scale, semantics, low, scale


def examples(frame, all_prefixes):
    result=[]
    for _, group in frame.groupby("trajectory_id", sort=False):
        pois=group.POI_id.astype(int).tolist(); times=group.norm_in_day_time.astype(float).fillna(0).tolist()
        if len(pois)<2: continue
        stops=range(1,len(pois)) if all_prefixes else [len(pois)-1]
        for stop in stops: result.append((pois[:stop],times[:stop],pois[stop]))
    return result


@dataclass
class Config:
    semantic_dim:int; model_dim:int=128; layers:int=2; heads:int=4; dropout:float=.2


def build_model(config, coords, semantics):
    torch=torch_module()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer("coords",torch.tensor(coords)); self.register_buffer("semantics",torch.tensor(semantics))
            self.semantic=torch.nn.Linear(config.semantic_dim,config.model_dim); self.spatial=torch.nn.Linear(2,config.model_dim); self.time=torch.nn.Linear(2,config.model_dim)
            layer=torch.nn.TransformerEncoderLayer(config.model_dim,config.heads,config.model_dim*4,config.dropout,batch_first=True,norm_first=True)
            self.encoder=torch.nn.TransformerEncoder(layer,config.layers); self.head=torch.nn.Sequential(torch.nn.LayerNorm(config.model_dim),torch.nn.Linear(config.model_dim,config.model_dim),torch.nn.GELU(),torch.nn.Linear(config.model_dim,2),torch.nn.Sigmoid())
        def forward(self,poi,times,lengths):
            angle=times*2*math.pi; x=self.semantic(self.semantics[poi])+self.spatial(self.coords[poi])+self.time(torch.stack((torch.sin(angle),torch.cos(angle)),-1))
            pos=torch.arange(poi.shape[1],device=poi.device)[None,:]; x=self.encoder(x,src_key_padding_mask=pos>=lengths[:,None])
            return self.head(x[torch.arange(x.shape[0],device=x.device),lengths-1])
    return Model()


def batches(rows,batch_size,shuffle,seed,device):
    torch=torch_module(); order=list(range(len(rows)))
    if shuffle: random.Random(seed).shuffle(order)
    for start in range(0,len(order),batch_size):
        selected=[rows[i] for i in order[start:start+batch_size]]; lengths=torch.tensor([len(r[0]) for r in selected]); width=int(lengths.max())
        poi=torch.zeros((len(selected),width),dtype=torch.long); times=torch.zeros((len(selected),width))
        for i,row in enumerate(selected): n=len(row[0]); poi[i,:n]=torch.tensor(row[0]); times[i,:n]=torch.tensor(row[1])
        yield poi.to(device),times.to(device),lengths.to(device),torch.tensor([r[2] for r in selected],device=device)


def evaluate(model,rows,candidate_coords,args,device):
    torch=torch_module(); model.eval(); candidates=torch.tensor(candidate_coords,device=device); n=h1=h5=h10=0; rr=0.
    with torch.no_grad():
        for poi,times,lengths,labels in batches(rows,args.batch_size,False,args.seed,device):
            predicted=model(poi,times,lengths); distances=torch.cdist(predicted,candidates); order=torch.argsort(distances,dim=1)
            ranks=(order==labels[:,None]).nonzero()[:,1]+1; n+=len(labels); h1+=int((ranks<=1).sum()); h5+=int((ranks<=5).sum()); h10+=int((ranks<=10).sum()); rr+=float((1/ranks.float()).sum())
    return {"count":n,"acc@1":h1/n,"acc@5":h5/n,"acc@10":h10/n,"mrr":rr/n}


def run(args):
    torch=torch_module(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train,val,test=map(load_csv,(args.train_csv,args.validation_csv,args.test_csv)); ids=json.loads(Path(args.candidate_ids).read_text()); count=len(ids)
    semantic=semantic_cache((train,val,test),args.embedding_cache,args.embedding_model,args.ollama_base_url)
    coords,semantics,low,scale=poi_features((train,val,test),count,semantic); config=Config(semantics.shape[1],args.model_dim,args.layers,args.heads,args.dropout)
    device_name="cuda" if args.device=="auto" and torch.cuda.is_available() else ("mps" if args.device=="auto" and torch.backends.mps.is_available() else ("cpu" if args.device=="auto" else args.device)); device=torch.device(device_name)
    model=build_model(config,coords,semantics).to(device); train_rows=examples(train,True); val_rows=examples(val,False); test_rows=examples(test,False)
    if args.train_limit: train_rows=train_rows[:args.train_limit]
    if args.validation_limit: val_rows=val_rows[:args.validation_limit]
    if args.test_limit: test_rows=test_rows[:args.test_limit]
    print(f"device={device} train={len(train_rows)} validation={len(val_rows)} test={len(test_rows)} pois={count} semantic_dim={config.semantic_dim}",flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4); best=-1.; state=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for poi,times,lengths,labels in batches(train_rows,args.batch_size,True,args.seed+epoch,device):
            optimizer.zero_grad(set_to_none=True); loss=torch.nn.functional.smooth_l1_loss(model(poi,times,lengths),model.coords[labels]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
        scores=evaluate(model,val_rows,coords,args,device); scores.update(epoch=epoch,train_loss=float(np.mean(losses))); history.append(scores); print(json.dumps(scores),flush=True)
        score=scores["acc@1"]+scores["acc@10"]
        if score>best: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); result=evaluate(model,test_rows,coords,args,device); result.update(method="NextLocLLM reproduction",dataset=args.dataset,city=args.city,seed=args.seed,device=device_name,test_limit=args.test_limit,candidate_count=count,embedding_model=args.embedding_model,duration="unavailable",protocol="coordinate prediction plus full-space nearest-POI retrieval",validation_history=history)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); torch.save({"model_state":state,"config":asdict(config),"coordinate_min":low,"coordinate_scale":scale},output/"best.pt"); (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n"); print(f"metrics={output/'metrics.json'}\n{json.dumps(result,indent=2)}")


def aggregate(args):
    rows=[json.loads((Path(args.root)/city/"metrics.json").read_text()) for city in args.cities]; keys=("acc@1","acc@5","acc@10","mrr")
    result={"method":"NextLocLLM reproduction","cities":args.cities,"city_count":len(rows),"macro_average":{k:float(np.mean([r[k] for r in rows])) for k in keys},"city_metrics":{r["city"]:{k:r[k] for k in keys} for r in rows}}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


def parser():
    root=argparse.ArgumentParser(); sub=root.add_subparsers(dest="command",required=True); p=sub.add_parser("run")
    for name in ("train_csv","validation_csv","test_csv","candidate_ids","embedding_cache","output","dataset","city"): p.add_argument("--"+name.replace("_","-"),required=True)
    p.add_argument("--embedding-model",default="qwen2:7b"); p.add_argument("--ollama-base-url",default="http://127.0.0.1:11434/v1"); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--seed",type=int,default=42); p.add_argument("--model-dim",type=int,default=128); p.add_argument("--layers",type=int,default=2); p.add_argument("--heads",type=int,default=4); p.add_argument("--dropout",type=float,default=.2); p.add_argument("--device",default="auto"); p.add_argument("--train-limit",type=int,default=0); p.add_argument("--validation-limit",type=int,default=0); p.add_argument("--test-limit",type=int,default=200)
    a=sub.add_parser("aggregate"); a.add_argument("--root",required=True); a.add_argument("--cities",nargs="+",required=True); a.add_argument("--output",required=True); return root


def main():
    args=parser().parse_args(); run(args) if args.command=="run" else aggregate(args)
if __name__=="__main__": main()
