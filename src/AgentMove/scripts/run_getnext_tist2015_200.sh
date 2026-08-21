#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; LIMIT="${QUERY_LIMIT:-200}"; EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"
CITIES=(Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow)
run_city() {
  local city="$1" base="data/hybrid/TIST2015/$1" out="results/getnext/TIST2015/limit-$LIMIT/$1"
  [[ -f "$out/metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip completed $city"; return; }
  "$PYTHON_BIN" -m hybrid.getnext_baseline run --dataset TIST2015 --city "$city" \
    --train-csv "$base/getnext/train.csv" --validation-csv "$base/getnext/val.csv" --test-csv "$base/getnext/test.csv" \
    --candidate-ids "$base/candidate_ids.json" --output "$out" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --test-limit "$LIMIT" --train-limit "${TRAIN_LIMIT:-0}" --validation-limit "${VALIDATION_LIMIT:-0}" --device "${DEVICE:-auto}"
}
case "${1:-pending}" in
  pending) for city in "${CITIES[@]}"; do run_city "$city"; done ;;
  aggregate) "$PYTHON_BIN" -m hybrid.getnext_baseline aggregate --root "results/getnext/TIST2015/limit-$LIMIT" --cities "${CITIES[@]}" --output "results/getnext/TIST2015/limit-$LIMIT/macro_average.json" ;;
  audit) for city in "${CITIES[@]}"; do [[ -f "results/getnext/TIST2015/limit-$LIMIT/$city/metrics.json" ]] && echo "done $city" || echo "pending $city"; done ;;
  *) run_city "$1" ;;
esac
