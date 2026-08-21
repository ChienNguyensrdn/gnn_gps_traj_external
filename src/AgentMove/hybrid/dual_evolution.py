from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .neural_cgm import ModelConfig, _batches, _recall, _torch, build_examples, build_model


def corrupt_examples(examples: Iterable[tuple], mode: str, seed: int) -> List[tuple]:
    """Change only input order; labels, users, target slots and splits remain fixed."""
    if mode not in {"correct", "reverse", "random"}:
        raise ValueError(f"unsupported order mode: {mode}")
    rng = random.Random(seed)
    output = []
    for pois, slots, user, target_slot, label in examples:
        pairs = list(zip(pois, slots))
        if mode == "reverse":
            pairs.reverse()
        elif mode == "random":
            rng.shuffle(pairs)
        ordered_pois = [item[0] for item in pairs]
        ordered_slots = [item[1] for item in pairs]
        output.append((ordered_pois, ordered_slots, user, target_slot, label))
    return output


def masked_temporal_mse(student, teacher, lengths):
    torch = _torch()
    width = student.shape[1]
    mask = torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]
    if width < 2:
        return student.sum() * 0.0
    delta_student = student[:, 1:] - student[:, :-1]
    delta_teacher = teacher[:, 1:] - teacher[:, :-1]
    transition_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
    count = transition_mask.sum().clamp_min(1) * student.shape[-1]
    return ((delta_student - delta_teacher).pow(2) * transition_mask).sum() / count


def distillation_losses(student, teacher, labels, projections, temperature: float):
    torch = _torch()
    functional = torch.nn.functional
    ce = functional.cross_entropy(student["logits"], labels)
    kd = functional.kl_div(
        functional.log_softmax(student["logits"] / temperature, dim=-1),
        functional.softmax(teacher["logits"] / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    projected_student = [layer(state) for layer, state in zip(projections, student["depth_states"])]
    projected_teacher = teacher["depth_states"]
    trajectory = sum(functional.mse_loss(left, right) for left, right in zip(projected_student, projected_teacher)) / len(projected_teacher)
    if len(projected_teacher) > 1:
        velocity = sum(
            functional.mse_loss(projected_student[i + 1] - projected_student[i], projected_teacher[i + 1] - projected_teacher[i])
            for i in range(len(projected_teacher) - 1)
        ) / (len(projected_teacher) - 1)
    else:
        velocity = ce * 0.0
    temporal = masked_temporal_mse(student["temporal_states"], teacher["temporal_states"], student["lengths"])
    return {"ce": ce, "kd": kd, "trajectory": trajectory, "velocity": velocity, "temporal": temporal}


def _device(torch, name: str):
    return torch.device("mps" if name == "auto" and torch.backends.mps.is_available() else ("cpu" if name == "auto" else name))


def train(args) -> Dict[str, float]:
    torch = _torch()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_config = ModelConfig(**teacher_checkpoint["config"])
    teacher = build_model(teacher_config)
    teacher.load_state_dict(teacher_checkpoint["model_state"])
    student_config = ModelConfig(
        teacher_config.num_pois, teacher_config.num_users, args.poi_dim,
        args.user_dim, args.time_dim, teacher_config.hidden_dim,
    )
    # The shared latent dimension makes state/velocity subtraction well-defined.
    student = build_model(student_config)
    projections = torch.nn.ModuleList([
        torch.nn.Linear(teacher_config.hidden_dim, teacher_config.hidden_dim),
        torch.nn.Linear(teacher_config.hidden_dim, teacher_config.hidden_dim),
    ])
    device = _device(torch, args.device)
    teacher.to(device).eval(); student.to(device); projections.to(device)
    frame_train = pd.read_csv(args.train_csv)
    frame_validation = pd.read_csv(args.validation_csv)
    user_map = teacher_checkpoint["user_map"]
    train_examples = corrupt_examples(build_examples(frame_train, user_map, True), args.order_mode, args.seed)
    validation_examples = corrupt_examples(build_examples(frame_validation, user_map, False), args.order_mode, args.seed)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projections.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    weights = {name: getattr(args, f"lambda_{name}") for name in ("kd", "trajectory", "velocity", "temporal")}
    best_score, best_state, best_metrics = -1.0, None, {}
    history = []
    for epoch in range(1, args.epochs + 1):
        student.train(); epoch_losses = []
        for batch in _batches(train_examples, args.batch_size, True, args.seed + epoch):
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_output = teacher(poi, slots, lengths, users, targets, return_states=True)
            student_output = student(poi, slots, lengths, users, targets, return_states=True)
            terms = distillation_losses(student_output, teacher_output, labels, projections, args.temperature)
            loss = terms["ce"] + sum(weights[name] * terms[name] for name in weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); optimizer.step()
            epoch_losses.append({name: float(value.detach().cpu()) for name, value in terms.items()} | {"total": float(loss.detach().cpu())})
        student.eval(); all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in _batches(validation_examples, args.batch_size, False, args.seed):
                poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
                all_logits.append(student(poi, slots, lengths, users, targets).cpu()); all_labels.append(labels.cpu())
        logits, labels = torch.cat(all_logits), torch.cat(all_labels)
        metrics = {"epoch": epoch, "order_mode": args.order_mode, "train_loss": float(np.mean([row["total"] for row in epoch_losses]))}
        metrics.update({f"recall@{k}": _recall(logits, labels, k) for k in (1, 5, 10)})
        metrics["loss_terms"] = {name: float(np.mean([row[name] for row in epoch_losses])) for name in epoch_losses[0]}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        score = metrics["recall@1"] + metrics["recall@10"]
        if score > best_score:
            best_score, best_metrics = score, metrics
            best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "config": asdict(student_config), "user_map": user_map,
                "candidate_ids": teacher_checkpoint["candidate_ids"], "metrics": best_metrics,
                "history": history, "distillation": {"temperature": args.temperature, "weights": weights,
                "order_mode": args.order_mode, "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve())}, "seed": args.seed}, destination)
    destination.with_suffix(".metrics.json").write_text(json.dumps({"best": best_metrics, "history": history}, indent=2), encoding="utf-8")
    return best_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-axis representation evolution distillation for Neural-CGM")
    parser.add_argument("--teacher-checkpoint", required=True); parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poi-dim", type=int, default=32); parser.add_argument("--user-dim", type=int, default=16)
    parser.add_argument("--time-dim", type=int, default=8); parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0); parser.add_argument("--lambda-trajectory", type=float, default=1.0)
    parser.add_argument("--lambda-velocity", type=float, default=1.0); parser.add_argument("--lambda-temporal", type=float, default=1.0)
    parser.add_argument("--order-mode", choices=["correct", "reverse", "random"], default="correct")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
