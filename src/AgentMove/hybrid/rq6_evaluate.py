from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dual_evolution import _device
from .evaluate_student import prediction_arrays, summarize_logits
from .evo_metrics import linear_cka, transition_cosine
from .neural_cgm import ModelConfig, _batches, _torch, build_examples, build_model


def length_thresholds(lengths: np.ndarray) -> tuple[int, int]:
    if len(lengths) == 0:
        raise ValueError("validation split produced zero sequence lengths")
    try:
        values = np.quantile(lengths, (1 / 3, 2 / 3), method="lower")
    except TypeError:  # NumPy < 1.22 compatibility.
        values = np.quantile(lengths, (1 / 3, 2 / 3), interpolation="lower")
    return int(values[0]), int(values[1])


def bucket_masks(lengths: np.ndarray, thresholds: tuple[int, int]) -> dict[str, np.ndarray]:
    short_max, medium_max = thresholds
    return {"short": lengths <= short_max,
            "medium": (lengths > short_max) & (lengths <= medium_max),
            "long": lengths > medium_max}


def masked_transition_sum(student: np.ndarray, teacher: np.ndarray, lengths: np.ndarray,
                          rotation: np.ndarray | None = None) -> tuple[float, int]:
    left = np.diff(student, axis=1); right = np.diff(teacher, axis=1)
    if rotation is not None: left = left @ rotation
    positions = np.arange(left.shape[1])[None, :]
    mask = positions < np.maximum(lengths[:, None] - 1, 0)
    left = left[mask]; right = right[mask]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 0
    if not np.any(valid): return 0.0, 0
    values = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return float(values.sum()), int(len(values))


class AlignmentAccumulator:
    def __init__(self, width: int):
        self.count = 0; self.student_sum = np.zeros(width); self.teacher_sum = np.zeros(width)
        self.cross = np.zeros((width, width))

    def add(self, student: np.ndarray, teacher: np.ndarray) -> None:
        if student.shape != teacher.shape or student.ndim != 2:
            raise ValueError("alignment states must have equal [samples, hidden] shape")
        self.count += len(student); self.student_sum += student.sum(axis=0); self.teacher_sum += teacher.sum(axis=0)
        self.cross += student.T @ teacher

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count < 2: raise ValueError("at least two validation states are required for alignment")
        student_mean = self.student_sum / self.count; teacher_mean = self.teacher_sum / self.count
        centered_cross = self.cross - self.count * np.outer(student_mean, teacher_mean)
        left, _, right = np.linalg.svd(centered_cross, full_matrices=False)
        return left @ right, student_mean, teacher_mean


def fit_alignments(student, teacher, examples: list, batch_size: int, device, seed: int):
    width = student.gru.hidden_size
    depth = [AlignmentAccumulator(width), AlignmentAccumulator(width)]; temporal = AlignmentAccumulator(width)
    torch = _torch()
    with torch.no_grad():
        for batch in _batches(examples, batch_size, False, seed):
            poi, slots, lengths, users, targets, _ = [value.to(device) for value in batch]
            student_output = student(poi, slots, lengths, users, targets, return_states=True)
            teacher_output = teacher(poi, slots, lengths, users, targets, return_states=True)
            for index in range(2):
                depth[index].add(student_output["depth_states"][index].cpu().numpy(),
                                 teacher_output["depth_states"][index].cpu().numpy())
            student_temporal = student_output["temporal_states"].cpu().numpy()
            teacher_temporal = teacher_output["temporal_states"].cpu().numpy()
            batch_lengths = lengths.cpu().numpy(); positions = np.arange(student_temporal.shape[1])[None, :]
            mask = positions < batch_lengths[:, None]
            temporal.add(student_temporal[mask], teacher_temporal[mask])
    return [item.solve() for item in depth], temporal.solve()


def evaluate(args) -> dict:
    torch = _torch(); device = _device(torch, args.device)
    student_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    student = build_model(ModelConfig(**student_checkpoint["config"])); student.load_state_dict(student_checkpoint["model_state"])
    teacher = build_model(ModelConfig(**teacher_checkpoint["config"])); teacher.load_state_dict(teacher_checkpoint["model_state"])
    student.to(device).eval(); teacher.to(device).eval()
    user_map = student_checkpoint["user_map"]
    validation = build_examples(pd.read_csv(args.validation_csv), user_map, all_prefixes=False)
    examples = build_examples(pd.read_csv(args.test_csv), user_map, all_prefixes=False)
    thresholds = length_thresholds(np.asarray([len(row[0]) for row in validation], dtype=int))
    depth_alignments, temporal_alignment = fit_alignments(student, teacher, validation, args.batch_size, device, args.seed)
    all_logits, all_labels, all_lengths = [], [], []
    student_depth = [[], []]; teacher_depth = [[], []]
    temporal_sum = 0.0; temporal_count = 0
    with torch.no_grad():
        for batch in _batches(examples, args.batch_size, False, args.seed):
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            student_output = student(poi, slots, lengths, users, targets, return_states=True)
            teacher_output = teacher(poi, slots, lengths, users, targets, return_states=True)
            all_logits.append(student_output["logits"].cpu().numpy()); all_labels.append(labels.cpu().numpy())
            batch_lengths = lengths.cpu().numpy(); all_lengths.append(batch_lengths)
            for index in range(2):
                student_depth[index].append(student_output["depth_states"][index].cpu().numpy())
                teacher_depth[index].append(teacher_output["depth_states"][index].cpu().numpy())
            total, count = masked_transition_sum(student_output["temporal_states"].cpu().numpy(),
                                                 teacher_output["temporal_states"].cpu().numpy(), batch_lengths,
                                                 temporal_alignment[0])
            temporal_sum += total; temporal_count += count
    if not all_logits: raise ValueError("test split produced zero examples")
    logits = np.concatenate(all_logits); labels = np.concatenate(all_labels); lengths = np.concatenate(all_lengths)
    overall = summarize_logits(logits, labels)
    cka_layers = [linear_cka(np.concatenate(student_depth[i]), np.concatenate(teacher_depth[i])) for i in range(2)]
    aligned_layers = []
    for index, values in enumerate(student_depth):
        rotation, student_mean, teacher_mean = depth_alignments[index]
        aligned_layers.append((np.concatenate(values) - student_mean) @ rotation + teacher_mean)
    student_layers = np.stack(aligned_layers, axis=1)
    teacher_layers = np.stack([np.concatenate(values) for values in teacher_depth], axis=1)
    overall.update({"cka": float(np.mean(cka_layers)), "cka_layer0": cka_layers[0], "cka_layer1": cka_layers[1],
                    "layer_transition_cosine": transition_cosine(student_layers, teacher_layers),
                    "transition_cosine": temporal_sum / temporal_count if temporal_count else 0.0})
    buckets = {}
    for name, mask in bucket_masks(lengths, thresholds).items():
        buckets[name] = summarize_logits(logits[mask], labels[mask]) if np.any(mask) else {"queries": 0}
    result = {"metrics": overall, "length_thresholds": {"short_max": thresholds[0], "medium_max": thresholds[1]},
              "length_buckets": buckets, "seed": args.seed, "device": str(device),
              "checkpoint": str(Path(args.checkpoint).resolve()), "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
              "alignment_fit_split": "validation", "alignment_method": "centered orthogonal Procrustes"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        predictions = Path(args.predictions_output); predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions, **prediction_arrays(logits, labels), lengths=lengths.astype(np.int32),
                            query_index=np.arange(len(labels), dtype=np.int64))
    print(json.dumps({"output": str(output), "metrics": overall, "length_thresholds": result["length_thresholds"]}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RQ6 dual-axis evolution and trajectory-length buckets")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--validation-csv", required=True); parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predictions-output")
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())


if __name__ == "__main__": main()
