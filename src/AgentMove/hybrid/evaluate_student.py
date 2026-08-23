from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dual_evolution import _device, corrupt_examples
from .metrics import expected_calibration_error
from .neural_cgm import ModelConfig, _batches, _torch, build_examples, build_model


def summarize_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-logits, axis=1, kind="stable")
    ranks = np.empty(len(labels), dtype=int)
    for index, label in enumerate(labels):
        ranks[index] = int(np.where(order[index] == label)[0][0]) + 1
    confidence = probabilities.max(axis=1)
    correct = (order[:, 0] == labels).astype(int)
    true_probability = probabilities[np.arange(len(labels)), labels]
    one_hot = np.zeros_like(probabilities); one_hot[np.arange(len(labels)), labels] = 1.0
    return {
        "queries": int(len(labels)),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "nll": float(-np.mean(np.log(np.clip(true_probability, 1e-12, 1.0)))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(confidence, correct),
    }


def resolve_order_mode(checkpoint: dict, requested: str) -> str:
    stored = checkpoint.get("distillation", {}).get("order_mode", "correct")
    if stored not in {"correct", "reverse", "random"}:
        raise ValueError(f"checkpoint has unsupported order mode: {stored}")
    if requested == "auto":
        return stored
    if requested != stored:
        raise ValueError(f"requested order mode {requested!r} does not match checkpoint order mode {stored!r}")
    return requested


def evaluate(args) -> dict[str, float]:
    torch = _torch(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**checkpoint["config"])); model.load_state_dict(checkpoint["model_state"])
    device = _device(torch, args.device); model.to(device).eval()
    order_mode = resolve_order_mode(checkpoint, args.order_mode)
    examples = corrupt_examples(
        build_examples(pd.read_csv(args.test_csv), checkpoint["user_map"], all_prefixes=False),
        order_mode,
        args.seed,
    )
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in _batches(examples, args.batch_size, False, args.seed):
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            all_logits.append(model(poi, slots, lengths, users, targets).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if not all_logits:
        raise ValueError("test split produced zero examples")
    metrics = summarize_logits(np.concatenate(all_logits), np.concatenate(all_labels))
    metrics.update({"device": str(device), "checkpoint": str(Path(args.checkpoint).resolve()),
                    "order_mode": order_mode, "seed": args.seed})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics)); return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen BeliefMove student on a held-out test CSV")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order-mode", choices=["auto", "correct", "reverse", "random"], default="auto")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
