from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint_models import build_checkpoint_model
from .dual_evolution import _device
from .neural_cgm import _batches, _torch, build_examples
from .evaluate_student import prediction_arrays, summarize_logits

VARIANTS = ("clean", "gps-drop-25", "gps-drop-50", "time-noise-30m", "time-noise-60m",
            "position-noise-200m", "position-noise-500m",
            "context-missing-user", "context-missing-time", "context-wrong-user", "context-wrong-time",
            "context-missing", "context-wrong")


def position_mapping(frame: pd.DataFrame, sigma_m: float, seed: int) -> dict[int, int]:
    required = {"POI_id", "latitude", "longitude"}
    if not required.issubset(frame.columns):
        raise ValueError(f"position noise requires columns: {sorted(required)}")
    coordinates = frame.groupby("POI_id", sort=True)[["latitude", "longitude"]].mean().dropna()
    ids = coordinates.index.to_numpy(dtype=int); points = coordinates.to_numpy(dtype=float)
    rng = np.random.default_rng(seed); latitude_scale = 111_320.0
    noisy = points.copy(); noisy[:, 0] += rng.normal(0, sigma_m / latitude_scale, len(points))
    longitude_scale = latitude_scale * np.maximum(np.cos(np.radians(points[:, 0])), 0.1)
    noisy[:, 1] += rng.normal(0, sigma_m / longitude_scale, len(points))
    scale = np.array([latitude_scale, float(np.mean(longitude_scale))])
    mapped = []
    for start in range(0, len(points), 256):
        distance = ((noisy[start:start + 256, None, :] - points[None, :, :]) * scale) ** 2
        mapped.extend(ids[np.argmin(distance.sum(axis=2), axis=1)].tolist())
    return dict(zip(ids.tolist(), mapped))


def perturb_examples(examples, variant: str, seed: int, num_users: int, poi_mapping=None):
    if variant not in VARIANTS: raise ValueError(f"unsupported RQ13 variant: {variant}")
    rng = np.random.default_rng(seed); output = []; changed = total = 0
    for pois, slots, user, target_slot, label in examples:
        pois = list(pois); slots = list(slots); original = (pois.copy(), slots.copy(), user, target_slot)
        if variant.startswith("gps-drop-"):
            fraction = int(variant.rsplit("-", 1)[1]) / 100
            keep = rng.random(len(pois)) >= fraction; keep[-1] = True
            pois = [value for value, selected in zip(pois, keep) if selected]
            slots = [value for value, selected in zip(slots, keep) if selected]
        elif variant.startswith("time-noise-"):
            minutes = int(variant.split("-")[2][:-1]); steps = max(1, minutes // 30)
            slots = [int((value + rng.integers(-steps, steps + 1)) % 48) for value in slots]
            target_slot = int((target_slot + rng.integers(-steps, steps + 1)) % 48)
        elif variant.startswith("position-noise-"):
            if poi_mapping is None: raise ValueError("position-noise requires a POI mapping")
            pois = [poi_mapping.get(value, value) for value in pois]
        elif variant == "context-missing-user":
            user = num_users
        elif variant == "context-missing-time":
            target_slot = 0
        elif variant == "context-wrong-user":
            user = (user + 1) % max(num_users, 1)
        elif variant == "context-wrong-time":
            target_slot = (target_slot + 24) % 48
        elif variant == "context-missing":
            user = num_users; target_slot = 0
        elif variant == "context-wrong":
            user = (user + 1) % max(num_users, 1); target_slot = (target_slot + 24) % 48
        changed += int((pois, slots, user, target_slot) != original); total += 1
        output.append((pois, slots, user, target_slot, label))
    return output, float(changed / total) if total else 0.0


def evaluate(args):
    torch = _torch(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_checkpoint_model(checkpoint); model.load_state_dict(checkpoint["model_state"])
    device = _device(torch, args.device); model.to(device).eval(); frame = pd.read_csv(args.test_csv)
    examples = build_examples(frame, checkpoint["user_map"], all_prefixes=False); mapping = None
    if args.variant.startswith("position-noise-"):
        meters = float(args.variant.split("-")[2][:-1]); mapping = position_mapping(frame, meters, args.seed)
    perturbed, changed_rate = perturb_examples(examples, args.variant, args.seed, len(checkpoint["user_map"]), mapping)
    logits_rows, labels_rows = [], []
    with torch.no_grad():
        for batch in _batches(perturbed, args.batch_size, False, args.seed):
            poi, slots, lengths, users, targets, labels = [value.to(device) for value in batch]
            logits_rows.append(model(poi, slots, lengths, users, targets).cpu().numpy()); labels_rows.append(labels.cpu().numpy())
    if not logits_rows: raise ValueError("RQ13 test split produced zero examples")
    logits = np.concatenate(logits_rows); labels = np.concatenate(labels_rows); metrics = summarize_logits(logits, labels)
    result = {"rq": "RQ13", "variant": args.variant, "seed": args.seed, "device": str(device),
              "checkpoint": str(Path(args.checkpoint).resolve()), "protocol": "frozen E5-dual; last-query; context-only perturbation; target unchanged",
              "changed_query_rate": changed_rate, "metrics": metrics}
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "rq13.metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "test.predictions.npz", **prediction_arrays(logits, labels),
                        query_index=np.arange(len(labels), dtype=np.int64))
    print(json.dumps({"output": str(output), "variant": args.variant, "changed_query_rate": changed_rate, **metrics}))
    return result


def main():
    parser = argparse.ArgumentParser(description="RQ13 deterministic input-robustness evaluation")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42); evaluate(parser.parse_args())


if __name__ == "__main__": main()
