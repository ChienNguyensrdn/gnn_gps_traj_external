from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required in the project virtual environment") from exc


@dataclass
class Config:
    num_pois: int
    num_users: int
    num_categories: int
    model_dim: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2


def build_model(config: Config):
    torch = _torch()

    class SparseGETNext(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.poi = torch.nn.Embedding(config.num_pois, config.model_dim)
            self.user = torch.nn.Embedding(config.num_users + 1, config.model_dim)
            self.category = torch.nn.Embedding(config.num_categories + 1, config.model_dim)
            self.time = torch.nn.Linear(2, config.model_dim)
            layer = torch.nn.TransformerEncoderLayer(
                config.model_dim, config.heads, config.model_dim * 4,
                config.dropout, batch_first=True, norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, config.layers)
            self.norm = torch.nn.LayerNorm(config.model_dim)
            self.output = torch.nn.Linear(config.model_dim, config.num_pois)
            self.flow_weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, poi, category, times, lengths, users, flow_targets, flow_values):
            positions = torch.arange(poi.shape[1], device=poi.device)[None, :]
            padding = positions >= lengths[:, None]
            angle = times * (2.0 * math.pi)
            x = self.poi(poi) + self.category(category) + self.time(torch.stack([torch.sin(angle), torch.cos(angle)], -1))
            x = self.encoder(x, src_key_padding_mask=padding)
            last = x[torch.arange(x.shape[0], device=x.device), lengths - 1] + self.user(users)
            logits = self.output(self.norm(last))
            if flow_targets.numel():
                logits.scatter_add_(1, flow_targets, self.flow_weight * flow_values)
            return logits

    return SparseGETNext()


def load_frame(path: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"GETNext split not found: {path}. Run the dataset prepare step "
            "or copy data/hybrid from the machine where preprocessing completed."
        )
    frame = pd.read_csv(path)
    required = {"user_id", "POI_id", "POI_catid_code", "trajectory_id", "norm_in_day_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame.sort_values(["trajectory_id", "UTC_time"], kind="stable")


def mappings(train: pd.DataFrame):
    users = {str(value): i for i, value in enumerate(sorted(train.user_id.astype(str).unique()))}
    categories = {str(value): i + 1 for i, value in enumerate(sorted(train.POI_catid_code.astype(str).unique()))}
    return users, categories


def examples(frame: pd.DataFrame, users, categories, all_prefixes: bool):
    rows = []
    for _, group in frame.groupby("trajectory_id", sort=False):
        poi = group.POI_id.astype(int).tolist()
        cat = [categories.get(str(value), 0) for value in group.POI_catid_code]
        times = group.norm_in_day_time.astype(float).fillna(0).tolist()
        if len(poi) < 2:
            continue
        stops = range(1, len(poi)) if all_prefixes else [len(poi) - 1]
        user = users.get(str(group.iloc[0].user_id), len(users))
        for stop in stops:
            rows.append((poi[:stop], cat[:stop], times[:stop], user, poi[stop]))
    return rows


def build_flow(train: pd.DataFrame, max_neighbors: int, num_pois: int):
    counts = defaultdict(Counter)
    for _, group in train.groupby("trajectory_id", sort=False):
        seq = group.POI_id.astype(int).tolist()
        for source, target in zip(seq, seq[1:]):
            counts[source][target] += 1
    flow = {}
    for source, targets in counts.items():
        total = sum(targets.values())
        # Log-lift over a uniform candidate prior. Frequent observed transitions
        # receive positive evidence without materialising a POI x POI matrix.
        flow[source] = [(target, math.log((count + 1) * num_pois / (total + num_pois))) for target, count in targets.most_common(max_neighbors)]
    return flow


def batches(rows, flow, batch_size, shuffle, seed, device):
    torch = _torch()
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    width_flow = max((len(value) for value in flow.values()), default=1)
    for start in range(0, len(order), batch_size):
        selected = [rows[i] for i in order[start:start + batch_size]]
        lengths = torch.tensor([len(row[0]) for row in selected], dtype=torch.long)
        width = int(lengths.max())
        poi = torch.zeros((len(selected), width), dtype=torch.long)
        cat = torch.zeros_like(poi)
        times = torch.zeros((len(selected), width), dtype=torch.float32)
        targets = torch.zeros((len(selected), width_flow), dtype=torch.long)
        values = torch.zeros((len(selected), width_flow), dtype=torch.float32)
        for i, row in enumerate(selected):
            n = len(row[0]); poi[i, :n] = torch.tensor(row[0]); cat[i, :n] = torch.tensor(row[1]); times[i, :n] = torch.tensor(row[2])
            neighbors = flow.get(row[0][-1], [])
            for j, (target, value) in enumerate(neighbors): targets[i, j] = target; values[i, j] = value
        yield tuple(value.to(device) for value in (poi, cat, times, lengths, torch.tensor([r[3] for r in selected]), targets, values, torch.tensor([r[4] for r in selected])))


def metric_state(logits, labels):
    torch = _torch()
    order = torch.argsort(logits, dim=1, descending=True)
    positions = (order == labels[:, None]).nonzero()[:, 1] + 1
    return {
        "count": int(labels.numel()), "hits@1": int((positions <= 1).sum()),
        "hits@5": int((positions <= 5).sum()), "hits@10": int((positions <= 10).sum()),
        "rr_sum": float((1.0 / positions.float()).sum()),
    }


def evaluate(model, rows, flow, args, device):
    torch = _torch(); model.eval(); total = {"count": 0, "hits@1": 0, "hits@5": 0, "hits@10": 0, "rr_sum": 0.0}
    with torch.no_grad():
        for batch in batches(rows, flow, args.batch_size, False, args.seed, device):
            *inputs, labels = batch
            state = metric_state(model(*inputs), labels)
            for key in total: total[key] += state[key]
    n = total["count"]
    return {"count": n, "acc@1": total["hits@1"] / n, "acc@5": total["hits@5"] / n,
            "acc@10": total["hits@10"] / n, "mrr": total["rr_sum"] / n}


def run(args):
    torch = _torch(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = load_frame(args.train_csv); val = load_frame(args.validation_csv); test = load_frame(args.test_csv)
    users, cats = mappings(train)
    candidate_ids = json.loads(Path(args.candidate_ids).read_text())
    config = Config(len(candidate_ids), len(users), len(cats), args.model_dim, args.layers, args.heads, args.dropout)
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        if frame.POI_id.min() < 0 or frame.POI_id.max() >= config.num_pois:
            raise ValueError(f"{name}: POI_id must be contiguous in [0, {config.num_pois - 1}]")
    device = torch.device(device_name); model = build_model(config).to(device); flow = build_flow(train, args.max_neighbors, config.num_pois)
    train_rows = examples(train, users, cats, True); val_rows = examples(val, users, cats, False); test_rows = examples(test, users, cats, False)
    if args.train_limit: train_rows = train_rows[:args.train_limit]
    if args.validation_limit: val_rows = val_rows[:args.validation_limit]
    if args.test_limit: test_rows = test_rows[:args.test_limit]
    if not train_rows or not val_rows or not test_rows: raise ValueError("A split produced zero GETNext examples")
    print(f"device={device} train={len(train_rows)} validation={len(val_rows)} test={len(test_rows)} pois={config.num_pois} sparse_edges={sum(map(len, flow.values()))}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4); loss_fn = torch.nn.CrossEntropyLoss()
    best_score = -1.0; best_state = None; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in batches(train_rows, flow, args.batch_size, True, args.seed + epoch, device):
            *inputs, labels = batch; optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(*inputs), labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scores = evaluate(model, val_rows, flow, args, device); scores.update(epoch=epoch, train_loss=float(np.mean(losses)))
        history.append(scores); print(json.dumps(scores), flush=True)
        score = scores["acc@1"] + scores["acc@10"]
        if score > best_score: best_score = score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); result = evaluate(model, test_rows, flow, args, device)
    result.update({"method": "GETNext sparse reproduction", "dataset": args.dataset, "city": args.city,
                   "seed": args.seed, "device": device_name, "protocol": "full-candidate next-POI; deterministic first-N test trajectories",
                   "test_limit": args.test_limit, "config": asdict(config), "validation_history": history})
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "config": asdict(config), "users": users, "categories": cats}, output / "best.pt")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"metrics={output / 'metrics.json'}\n{json.dumps(result, indent=2)}")


def aggregate(args):
    rows = [json.loads((Path(args.root) / city / "metrics.json").read_text()) for city in args.cities]
    keys = ["acc@1", "acc@5", "acc@10", "mrr"]
    result = {"method": "GETNext sparse reproduction", "cities": args.cities, "city_count": len(rows),
              "macro_average": {key: float(np.mean([row[key] for row in rows])) for key in keys},
              "city_metrics": {row["city"]: {key: row[key] for key in keys} for row in rows}}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser():
    root = argparse.ArgumentParser(description="Memory-safe sparse GETNext reproduction")
    sub = root.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    for name in ("train_csv", "validation_csv", "test_csv", "candidate_ids", "output", "dataset", "city"):
        run_p.add_argument("--" + name.replace("_", "-"), required=True)
    run_p.add_argument("--epochs", type=int, default=10); run_p.add_argument("--batch-size", type=int, default=32)
    run_p.add_argument("--learning-rate", type=float, default=1e-3); run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--model-dim", type=int, default=128); run_p.add_argument("--layers", type=int, default=2)
    run_p.add_argument("--heads", type=int, default=4); run_p.add_argument("--dropout", type=float, default=0.2)
    run_p.add_argument("--max-neighbors", type=int, default=64); run_p.add_argument("--device", default="auto")
    run_p.add_argument("--train-limit", type=int, default=0); run_p.add_argument("--validation-limit", type=int, default=0)
    run_p.add_argument("--test-limit", type=int, default=200)
    agg = sub.add_parser("aggregate"); agg.add_argument("--root", required=True); agg.add_argument("--cities", nargs="+", required=True); agg.add_argument("--output", required=True)
    return root


def main():
    args = parser().parse_args(); run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__": main()
