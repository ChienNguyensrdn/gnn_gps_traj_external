from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import TemperatureScaler
from .io import read_jsonl, write_json
from .metrics import expected_calibration_error


def _bundle(path: str):
    rows = list(read_jsonl(path)); bundle = rows.pop(0)["_bundle"]
    ids = [str(value) for value in json.loads(Path(bundle["candidate_ids"]).read_text(encoding="utf-8"))]
    index = {value: position for position, value in enumerate(ids)}
    labels = np.asarray([index[str(row["true_id"])] for row in rows], dtype=int)
    row_indices = np.asarray([int(row["_row_index"]) for row in rows], dtype=int)
    return np.load(bundle["logits"], mmap_mode="r")[row_indices], labels


def _metrics(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled); probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    confidence = probabilities[np.arange(len(labels)), order[:, 0]]
    correct = (ranks == 1).astype(int)
    true_probability = probabilities[np.arange(len(labels)), labels]
    brier = np.sum(probabilities * probabilities, axis=1) - 2.0 * true_probability + 1.0
    return {
        "queries": len(labels), "temperature": temperature,
        "acc@1": float(np.mean(ranks <= 1)), "acc@5": float(np.mean(ranks <= 5)),
        "acc@10": float(np.mean(ranks <= 10)), "mrr": float(np.mean(1.0 / ranks)),
        "ece": expected_calibration_error(confidence, correct),
        "nll": float(np.mean(-np.log(np.maximum(true_probability, 1e-12)))),
        "brier": float(np.mean(brier)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-space Stage-1 logits before/after temperature scaling")
    parser.add_argument("--validation", required=True); parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validation_logits, validation_labels = _bundle(args.validation)
    temperature = TemperatureScaler().fit(validation_logits, validation_labels).temperature
    test_logits, test_labels = _bundle(args.test)
    result = {"uncalibrated": _metrics(test_logits, test_labels, 1.0), "temperature_scaled": _metrics(test_logits, test_labels, temperature)}
    write_json(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
