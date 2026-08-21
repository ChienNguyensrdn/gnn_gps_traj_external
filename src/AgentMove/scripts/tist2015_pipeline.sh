#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/tist2015_common.sh"
cd "$(tist2015_agentmove_root)"

ACTION="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
MODEL_SLUG="$(tist2015_model_slug "$OLLAMA_MODEL")"
RAW_DIR="data/dataset_tist2015"
LOG_DIR="${LOG_DIR:-results/logs/tist2015/$MODEL_SLUG}"
mkdir -p "$LOG_DIR"

raw_files=(dataset_TIST2015_Checkins.txt dataset_TIST2015_POIs.txt dataset_TIST2015_Cities.txt)

audit() {
  local file city missing=0 status metrics
  echo "TIST2015 protocol audit model=$OLLAMA_MODEL endpoint=http://127.0.0.1:11434/v1"
  [[ -x "$PYTHON_BIN" ]] && echo "python=ready ($PYTHON_BIN)" || { echo "python=missing ($PYTHON_BIN)"; missing=1; }
  for file in "${raw_files[@]}"; do
    [[ -f "$RAW_DIR/$file" ]] && echo "raw=ready $RAW_DIR/$file" || { echo "raw=missing $RAW_DIR/$file"; missing=1; }
  done
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && echo "ollama=ready" || echo "ollama=offline (required only for run)"
  for city in "${TIST2015_CITIES[@]}"; do
    status="prepared"; [[ -f "data/hybrid/TIST2015/$city/validation.jsonl" ]] || status="unprepared"
    [[ -f "data/hybrid/TIST2015/$city/neural_cgm/best.pt" ]] && status="$status,trained" || status="$status,untrained"
    metrics="results/tist2015-hybrid/$MODEL_SLUG/limit-$QUERY_LIMIT/no-osm/$city/full/metrics.json"
    [[ -f "$metrics" ]] && status="$status,complete-no-osm" || status="$status,pending"
    printf '%-14s %s\n' "$city" "$status"
  done
  return "$missing"
}

download() {
  echo "Dataset download explicitly requested; preserving existing raw files."
  "$PYTHON_BIN" -m processing.download --download_mode data --data_name tist2015
}

prepare() { PYTHON_BIN="$PYTHON_BIN" ./scripts/prepare_tist2015_hybrid.sh; }

train() {
  local city base output
  for city in "${TIST2015_CITIES[@]}"; do
    base="data/hybrid/TIST2015/$city"; output="$base/neural_cgm/best.pt"
    [[ -f "$output" && "${FORCE_TRAIN:-0}" != "1" ]] && { echo "skip trained $city"; continue; }
    for file in "$base/getnext/train.csv" "$base/getnext/val.csv" "$base/candidate_ids.json"; do
      [[ -f "$file" ]] || { echo "Missing prepared input: $file" >&2; exit 2; }
    done
    mkdir -p "$base/neural_cgm"
    "$PYTHON_BIN" -m hybrid.neural_cgm train --train-csv "$base/getnext/train.csv" \
      --validation-csv "$base/getnext/val.csv" --candidate-ids "$base/candidate_ids.json" \
      --output "$output" --epochs "${EPOCHS:-10}" --batch-size "${BATCH_SIZE:-64}" \
      --learning-rate "${LEARNING_RATE:-0.001}" --seed "${SEED:-42}" 2>&1 | tee "$LOG_DIR/train-$city.log"
    for split in validation test; do
      "$PYTHON_BIN" -m hybrid.neural_cgm export --checkpoint "$output" --input "$base/$split.jsonl" \
        --getnext-csv "$base/getnext/train.csv" "$base/getnext/val.csv" "$base/getnext/test.csv" \
        --output "$base/neural_cgm/${split}_logits.npy" 2>&1 | tee "$LOG_DIR/export-$city-$split.log"
      "$PYTHON_BIN" -m hybrid.cgm_adapter --logits "$base/neural_cgm/${split}_logits.npy" \
        --metadata "$base/${split}_metadata.jsonl" --candidate-ids "$base/candidate_ids.json" \
        --candidate-metadata "$base/candidate_metadata.json" --output "$base/neural_cgm/$split.jsonl"
    done
  done
}

run_all() {
  ./scripts/start_ollama.sh
  local city
  for city in ${CITIES:-${TIST2015_CITIES[*]}}; do
    tist2015_is_city "$city" || { echo "Invalid city: $city" >&2; exit 2; }
    QUERY_LIMIT="${TEST_LIMIT:-$QUERY_LIMIT}" VALIDATION_LIMIT="${VALIDATION_LIMIT:-$QUERY_LIMIT}" \
      OLLAMA_MODEL="$OLLAMA_MODEL" ./scripts/run_tist2015_city_200.sh "$city" 2>&1 | tee "$LOG_DIR/run-$city.log"
  done
}

aggregate_results() {
  QUERY_LIMIT="$QUERY_LIMIT" OLLAMA_MODEL="$OLLAMA_MODEL" ./scripts/complete_tist2015_table2.sh aggregate
  "$PYTHON_BIN" -m hybrid.aggregate_runs --results-root results \
    --output "results/tist2015-hybrid/$MODEL_SLUG/run_index.json"
}

case "$ACTION" in
  audit) audit ;;
  download) download ;;
  prepare) prepare ;;
  train) train ;;
  run) run_all ;;
  aggregate) aggregate_results ;;
  *) echo "Usage: $0 <audit|download|prepare|train|run|aggregate>" >&2; exit 2 ;;
esac
