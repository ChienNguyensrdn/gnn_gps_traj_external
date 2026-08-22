#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; LIMIT="${QUERY_LIMIT:-200}"; EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"
CITIES=(Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow)
run_city() {
  local city="$1" base="data/hybrid/TIST2015/$1" out="results/getnext/TIST2015/limit-$LIMIT/$1"
  [[ -f "$out/metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip completed $city"; return; }
  if [[ ! -f "$base/getnext/train.csv" || ! -f "$base/getnext/val.csv" || ! -f "$base/getnext/test.csv" || ! -f "$base/candidate_ids.json" ]]; then
    local input=""
    [[ -f "data/input_trajectories_clean/${city}_filtered.csv" ]] && input="data/input_trajectories_clean/${city}_filtered.csv"
    [[ -z "$input" && -f "data/input_trajectories/${city}_filtered.csv" ]] && input="data/input_trajectories/${city}_filtered.csv"
    if [[ -z "$input" ]]; then
      echo "Missing GETNext and normalized data for $city. Copy data/hybrid/TIST2015/$city or data/input_trajectories/${city}_filtered.csv to this machine." >&2
      return 2
    fi
    echo "Preparing GETNext CSVs for $city from $input"
    "$PYTHON_BIN" -m hybrid.prepare_dataset --dataset tist2015 --input "$input" --city "$city" --output-dir "$base"
  fi
  "$PYTHON_BIN" -m hybrid.getnext_baseline run --dataset TIST2015 --city "$city" \
    --train-csv "$base/getnext/train.csv" --validation-csv "$base/getnext/val.csv" --test-csv "$base/getnext/test.csv" \
    --candidate-ids "$base/candidate_ids.json" --output "$out" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --test-limit "$LIMIT" --train-limit "${TRAIN_LIMIT:-0}" --validation-limit "${VALIDATION_LIMIT:-0}" --device "${DEVICE:-auto}"
}
case "${1:-pending}" in
  pending) for city in "${CITIES[@]}"; do run_city "$city"; done ;;
  aggregate) QUERY_LIMIT="$LIMIT" ./scripts/aggregate_getnext_results.sh aggregate ;;
  audit) for city in "${CITIES[@]}"; do [[ -f "results/getnext/TIST2015/limit-$LIMIT/$city/metrics.json" ]] && echo "done $city" || echo "pending $city"; done ;;
  *) run_city "$1" ;;
esac
