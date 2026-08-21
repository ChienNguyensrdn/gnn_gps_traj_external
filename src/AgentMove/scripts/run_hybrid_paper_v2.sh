#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
COMMAND="${1:-audit}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"; MODEL_SLUG="${MODEL//[:\/]/-}"
TOP_K="${TOP_K:-10}"
DATA_DIR="${V2_DATA_DIR:-data/hybrid/Shanghai/neural_cgm/paper-v2-agentmove-200}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hybrid/paper-v2-agentmove-200/${MODEL_SLUG}}"

prepare() {
  "$PYTHON_BIN" -m hybrid.augment_cgm \
    --validation data/hybrid/Shanghai/neural_cgm/sample-50-seed42/validation.jsonl \
    --test data/hybrid/Shanghai/neural_cgm/agentmove-200/test.jsonl --output-dir "$DATA_DIR"
}
audit() {
  "$PYTHON_BIN" -m hybrid.paper_v2_audit --test "$DATA_DIR/test.jsonl" --top-k "$TOP_K" \
    --min-recall "${MIN_CANDIDATE_RECALL:-0.45}" --min-osm-coverage "${MIN_OSM_COVERAGE:-0.9}" \
    --output "$DATA_DIR/protocol_audit.json"
}

if [[ "$COMMAND" == "osm" ]]; then
  USE_OSM=1 DATASET=isp CITY=Shanghai NOMINATIM_URL="${NOMINATIM_URL:-http://127.0.0.1:8080}" ./scripts/hybrid_pipeline.sh osm
  "$PYTHON_BIN" -m hybrid.refresh_osm_metadata --csv data/input_trajectories_clean/Shanghai_filtered.csv \
    --metadata data/hybrid/Shanghai/candidate_metadata.json
  exit 0
fi
prepare
if [[ "$COMMAND" == "prepare" ]]; then exit 0; fi
audit
if [[ "$COMMAND" == "audit" ]]; then exit 0; fi
[[ "$COMMAND" == "run" ]] || { echo "Usage: $0 osm|prepare|audit|run" >&2; exit 2; }

export DATASET=isp CITY=Shanghai OLLAMA_MODEL="$MODEL" OUTPUT_DIR
export VALIDATION_FILE="$DATA_DIR/validation.jsonl" TEST_FILE="$DATA_DIR/test.jsonl"
export TOP_K TOP_M="${TOP_M:-5}" LLM_BATCH_SIZE="$TOP_K" LLM_RETRIES="${LLM_RETRIES:-2}"
export LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}" COMPACT_EVIDENCE=1
export HYBRID_VARIANTS="${HYBRID_VARIANTS:-full no_temperature no_link_calibration stage1_only stage1_uncalibrated}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}" OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
./scripts/hybrid_pipeline.sh run
