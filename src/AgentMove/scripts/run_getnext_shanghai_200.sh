#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LIMIT="${QUERY_LIMIT:-200}"; EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"
exec "$PYTHON_BIN" -m hybrid.getnext_baseline run \
  --dataset ISP-Shanghai --city Shanghai \
  --train-csv data/hybrid/Shanghai/getnext/train.csv \
  --validation-csv data/hybrid/Shanghai/getnext/val.csv \
  --test-csv data/hybrid/Shanghai/getnext/test.csv \
  --candidate-ids data/hybrid/Shanghai/candidate_ids.json \
  --output "results/getnext/ISP-Shanghai/limit-$LIMIT" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --test-limit "$LIMIT" \
  --train-limit "${TRAIN_LIMIT:-0}" --validation-limit "${VALIDATION_LIMIT:-0}" --device "${DEVICE:-auto}"
