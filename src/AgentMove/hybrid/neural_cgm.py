from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    num_pois: int
    num_users: int
    poi_dim: int = 64
    user_dim: int = 32
    time_dim: int = 16
    hidden_dim: int = 128


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required: .venv/bin/python -m pip install torch") from exc


def build_model(config: ModelConfig):
    torch = _torch()

    class NeuralCGM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.poi = torch.nn.Embedding(config.num_pois, config.poi_dim)
            self.time = torch.nn.Embedding(48, config.time_dim)
            self.user = torch.nn.Embedding(config.num_users + 1, config.user_dim)
            self.gru = torch.nn.GRU(config.poi_dim + config.time_dim, config.hidden_dim, batch_first=True)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_dim + config.user_dim + config.time_dim, config.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim, config.num_pois),
            )

        def forward(self, poi_ids, time_slots, lengths, user_ids, target_slots, return_states=False):
            embedded = torch.cat([self.poi(poi_ids), self.time(time_slots)], dim=-1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            packed_output, hidden = self.gru(packed)
            features = torch.cat([hidden[-1], self.user(user_ids), self.time(target_slots)], dim=-1)
            head_state = self.head[1](self.head[0](features))
            logits = self.head[2](head_state)
            if not return_states:
                return logits
            temporal, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=poi_ids.shape[1]
            )
            return {
                "logits": logits,
                "depth_states": [hidden[-1], head_state],
                "temporal_states": temporal,
                "lengths": lengths,
            }

    return NeuralCGM()


def _slot(values: pd.Series) -> List[int]:
    parsed = pd.to_datetime(values, errors="coerce")
    return (parsed.dt.hour * 2 + parsed.dt.minute // 30).fillna(0).astype(int).clip(0, 47).tolist()


def build_examples(frame: pd.DataFrame, user_map: Dict[str, int], all_prefixes: bool) -> List[Tuple[List[int], List[int], int, int, int]]:
    examples = []
    for _, rows in frame.groupby("trajectory_id", sort=False):
        rows = rows.sort_values("UTC_time")
        pois = rows["POI_id"].astype(int).tolist()
        slots = _slot(rows["UTC_time"])
        if len(pois) < 2:
            continue
        user = user_map.get(str(rows.iloc[0]["user_id"]), len(user_map))
        stops = range(1, len(pois)) if all_prefixes else [len(pois) - 1]
        for stop in stops:
            examples.append((pois[:stop], slots[:stop], user, slots[stop], pois[stop]))
    return examples


def _batches(examples, batch_size: int, shuffle: bool, seed: int):
    torch = _torch()
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        rows = [examples[index] for index in indices[start:start + batch_size]]
        lengths = torch.tensor([len(row[0]) for row in rows], dtype=torch.long)
        width = int(lengths.max())
        poi = torch.zeros((len(rows), width), dtype=torch.long)
        slots = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            poi[index, :len(row[0])] = torch.tensor(row[0])
            slots[index, :len(row[1])] = torch.tensor(row[1])
        yield poi, slots, lengths, torch.tensor([row[2] for row in rows]), torch.tensor([row[3] for row in rows]), torch.tensor([row[4] for row in rows])


def _recall(logits, labels, k: int) -> float:
    torch = _torch()
    return float((torch.topk(logits, min(k, logits.shape[1]), dim=1).indices == labels[:, None]).any(dim=1).float().mean())


def train(args) -> Dict[str, float]:
    torch = _torch()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_frame = pd.read_csv(args.train_csv)
    validation_frame = pd.read_csv(args.validation_csv)
    candidate_ids = json.loads(Path(args.candidate_ids).read_text(encoding="utf-8"))
    user_values = sorted(train_frame["user_id"].astype(str).unique())
    user_map = {value: index for index, value in enumerate(user_values)}
    config = ModelConfig(len(candidate_ids), len(user_map), args.poi_dim, args.user_dim, args.time_dim, args.hidden_dim)
    model = build_model(config)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device))
    model.to(device)
    train_examples = build_examples(train_frame, user_map, all_prefixes=True)
    validation_examples = build_examples(validation_frame, user_map, all_prefixes=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best, best_state, best_metrics = -1.0, None, {}
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in _batches(train_examples, args.batch_size, True, args.seed + epoch):
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(poi, slots, lengths, users, targets)
            loss = criterion(logits, labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in _batches(validation_examples, args.batch_size, False, args.seed):
                poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
                all_logits.append(model(poi, slots, lengths, users, targets).cpu()); all_labels.append(labels.cpu())
        logits = torch.cat(all_logits); labels = torch.cat(all_labels)
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), "recall@1": _recall(logits, labels, 1), "recall@5": _recall(logits, labels, 5), "recall@10": _recall(logits, labels, 10)}
        print(json.dumps(metrics), flush=True)
        score = metrics["recall@1"] + metrics["recall@10"]
        if score > best:
            best, best_metrics = score, metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "config": asdict(config), "user_map": user_map, "candidate_ids": candidate_ids, "metrics": best_metrics, "seed": args.seed}, destination)
    print(f"checkpoint={destination} best={json.dumps(best_metrics)}")
    return best_metrics


def _trajectory_users(csv_paths: Sequence[str]) -> Dict[str, str]:
    result = {}
    for path in csv_paths:
        frame = pd.read_csv(path, usecols=["trajectory_id", "user_id"])
        for row in frame.drop_duplicates("trajectory_id").itertuples(index=False):
            result[str(row.trajectory_id)] = str(row.user_id)
    return result


def export(args) -> None:
    torch = _torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["config"]); model = build_model(config)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    candidate_index = {str(value): index for index, value in enumerate(checkpoint["candidate_ids"])}
    trajectory_users = _trajectory_users(args.getnext_csv)
    from .io import read_jsonl
    queries = [row for row in read_jsonl(args.input) if "_bundle" not in row]
    logits_rows = []
    with torch.no_grad():
        for query in queries:
            context = query.get("context", [])
            context_ids = [candidate_index[str(row[-1])] for row in context if str(row[-1]) in candidate_index]
            if not context_ids:
                context_ids = [0]
            context_slots = []
            for row in context[-len(context_ids):]:
                stamp = pd.to_datetime(row[0], errors="coerce")
                context_slots.append(int(stamp.hour * 2 + stamp.minute // 30) if not pd.isna(stamp) else 0)
            trajectory = str(query.get("metadata", {}).get("trajectory_id", str(query["query_id"]).split(":", 1)[-1]))
            encoded_user = trajectory_users.get(trajectory, "")
            user = checkpoint["user_map"].get(encoded_user, len(checkpoint["user_map"]))
            target = pd.to_datetime(query.get("target_time"), errors="coerce")
            target_slot = int(target.hour * 2 + target.minute // 30) if not pd.isna(target) else 0
            poi = torch.tensor([context_ids]); slots = torch.tensor([context_slots]); lengths = torch.tensor([len(context_ids)])
            row = model(poi, slots, lengths, torch.tensor([user]), torch.tensor([target_slot]))[0].numpy().astype(np.float32)
            logits_rows.append(row)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.stack(logits_rows))
    print(f"exported={len(logits_rows)} shape={np.stack(logits_rows).shape} output={output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export a learned neural candidate generator")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train-csv", required=True); train_parser.add_argument("--validation-csv", required=True)
    train_parser.add_argument("--candidate-ids", required=True); train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--epochs", type=int, default=10); train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3); train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--poi-dim", type=int, default=64); train_parser.add_argument("--user-dim", type=int, default=32)
    train_parser.add_argument("--time-dim", type=int, default=16); train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--device", default="auto")
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True); export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--getnext-csv", nargs="+", required=True); export_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args) if args.command == "train" else export(args)


if __name__ == "__main__":
    main()
