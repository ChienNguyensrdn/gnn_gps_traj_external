#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${OLLAMA_MODEL//[:\/]/-}"
SAMPLE_DIR="data/hybrid/Shanghai/neural_cgm/sample-50-seed${SAMPLE_SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hybrid/shanghai-neural-cgm-50-seed${SAMPLE_SEED}/${MODEL_SLUG}}"

checkpoint="data/hybrid/Shanghai/neural_cgm/best.pt"
[[ -f "$checkpoint" ]] || {
  echo "Missing neural CGM checkpoint: $checkpoint" >&2
  echo "Train it with: .venv/bin/python -m hybrid.neural_cgm train ..." >&2
  exit 2
}
[[ -f "$SAMPLE_DIR/validation.jsonl" && -f "$SAMPLE_DIR/test.jsonl" ]] || {
  echo "Missing neural CGM sample files in $SAMPLE_DIR" >&2; exit 2;
}

export DATASET=isp CITY=Shanghai OLLAMA_MODEL OUTPUT_DIR
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export VALIDATION_FILE="$SAMPLE_DIR/validation.jsonl"
export TEST_FILE="$SAMPLE_DIR/test.jsonl"
export TOP_K="${TOP_K:-10}" TOP_M="${TOP_M:-5}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-3}" LLM_RETRIES="${LLM_RETRIES:-2}"
export LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}"
export OLLAMA_RESTART_ATTEMPTS="${OLLAMA_RESTART_ATTEMPTS:-5}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

echo "Shanghai Neural-CGM 50% RQ pipeline"
echo "checkpoint=$checkpoint model=$OLLAMA_MODEL output=$OUTPUT_DIR"
echo "Two resumable evidence caches will be built: embedding and frequency memory."

if [[ "${REPORT_ONLY:-0}" != "1" ]]; then
  ./scripts/hybrid_pipeline.sh run
fi
if [[ "${HYBRID_DRY_RUN:-0}" == "1" ]]; then exit 0; fi

"$PYTHON_BIN" -m hybrid.rq_report \
  --results "$OUTPUT_DIR" --manifest "$SAMPLE_DIR/sample_manifest.json"
echo "Report: $OUTPUT_DIR/RQ_REPORT.md"
