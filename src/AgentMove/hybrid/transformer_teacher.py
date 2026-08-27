from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .neural_cgm import _batches, _recall, _torch, build_examples


def _device(torch, name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TransformerConfig:
    num_pois: int
    num_users: int
    poi_dim: int = 64
    user_dim: int = 32
    time_dim: int = 16
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1


def build_model(config: TransformerConfig):
    torch = _torch()

    class TransformerTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.poi = torch.nn.Embedding(config.num_pois, config.poi_dim)
            self.time = torch.nn.Embedding(48, config.time_dim)
            self.user = torch.nn.Embedding(config.num_users + 1, config.user_dim)
            self.input_projection = torch.nn.Linear(config.poi_dim + config.time_dim, config.hidden_dim)
            self.layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(
                config.hidden_dim, config.heads, config.hidden_dim * 4, config.dropout,
                batch_first=True, norm_first=True) for _ in range(config.layers)])
            self.norm = torch.nn.LayerNorm(config.hidden_dim)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_dim + config.user_dim + config.time_dim, config.hidden_dim),
                torch.nn.ReLU(), torch.nn.Linear(config.hidden_dim, config.num_pois))

        @staticmethod
        def positional(width, hidden, device, dtype):
            position = torch.arange(width, device=device, dtype=dtype)[:, None]
            scale = torch.exp(torch.arange(0, hidden, 2, device=device, dtype=dtype) * (-math.log(10000.0) / hidden))
            result = torch.zeros((width, hidden), device=device, dtype=dtype)
            result[:, 0::2] = torch.sin(position * scale)
            result[:, 1::2] = torch.cos(position * scale[:result[:, 1::2].shape[1]])
            return result

        def forward(self, poi_ids, time_slots, lengths, user_ids, target_slots, return_states=False):
            hidden = self.input_projection(torch.cat([self.poi(poi_ids), self.time(time_slots)], dim=-1))
            hidden = hidden + self.positional(hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype)
            mask = torch.arange(hidden.shape[1], device=lengths.device)[None, :] >= lengths[:, None]
            first = None
            for index, layer in enumerate(self.layers):
                hidden = layer(hidden, src_key_padding_mask=mask)
                if index == 0: first = hidden
            hidden = self.norm(hidden); positions = (lengths - 1).clamp_min(0)
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), positions]
            first_pooled = first[torch.arange(len(first), device=first.device), positions]
            features = torch.cat([pooled, self.user(user_ids), self.time(target_slots)], dim=-1)
            head_state = self.head[1](self.head[0](features)); logits = self.head[2](head_state)
            if not return_states: return logits
            return {"logits": logits, "depth_states": [first_pooled, head_state],
                    "temporal_states": hidden.masked_fill(mask.unsqueeze(-1), 0), "lengths": lengths}

    return TransformerTeacher()


def train(args):
    torch = _torch(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_frame = pd.read_csv(args.train_csv); validation_frame = pd.read_csv(args.validation_csv)
    candidate_ids = json.loads(Path(args.candidate_ids).read_text()); users = sorted(train_frame["user_id"].astype(str).unique())
    user_map = {value: index for index, value in enumerate(users)}
    config = TransformerConfig(len(candidate_ids), len(user_map), args.poi_dim, args.user_dim, args.time_dim,
                               args.hidden_dim, args.heads, args.layers, args.dropout)
    model = build_model(config); device = _device(torch, args.device); model.to(device)
    train_examples = build_examples(train_frame, user_map, True); validation_examples = build_examples(validation_frame, user_map, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_score, best_state, best_metrics, history = -1.0, None, {}, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in _batches(train_examples, args.batch_size, True, args.seed + epoch):
            poi, slots, lengths, user_ids, targets, labels = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True); logits = model(poi, slots, lengths, user_ids, targets)
            loss = torch.nn.functional.cross_entropy(logits, labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); logits, labels = [], []
        with torch.no_grad():
            for batch in _batches(validation_examples, args.batch_size, False, args.seed):
                poi, slots, lengths, user_ids, targets, batch_labels = [value.to(device) for value in batch]
                logits.append(model(poi, slots, lengths, user_ids, targets).cpu()); labels.append(batch_labels.cpu())
        logits, labels = torch.cat(logits), torch.cat(labels)
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   **{f"recall@{k}": _recall(logits, labels, k) for k in (1, 5, 10)}}
        history.append(metrics); print(json.dumps(metrics), flush=True); score = metrics["recall@1"] + metrics["recall@10"]
        if score > best_score:
            best_score, best_metrics = score, metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "transformer", "model_state": best_state, "config": asdict(config),
                "user_map": user_map, "candidate_ids": candidate_ids, "metrics": best_metrics,
                "history": history, "seed": args.seed}, output)
    output.with_suffix(".metrics.json").write_text(json.dumps({"best": best_metrics, "history": history}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train a matched Transformer next-location teacher")
    parser.add_argument("--train-csv", required=True); parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--candidate-ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poi-dim", type=int, default=64); parser.add_argument("--user-dim", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4); parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1); parser.add_argument("--device", default="auto")
    train(parser.parse_args())


if __name__ == "__main__": main()
