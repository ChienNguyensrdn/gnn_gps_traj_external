#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_DIR="${SAMPLE_DIR:-data/hybrid/Shanghai/sample-50-seed${SAMPLE_SEED}}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${OLLAMA_MODEL//[:\/]/-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hybrid/shanghai-50-seed${SAMPLE_SEED}/${MODEL_SLUG}}"

base_validation="data/hybrid/Shanghai/validation.jsonl"
base_test="data/hybrid/Shanghai/test.jsonl"
[[ -f "$base_validation" && -f "$base_test" ]] || {
  echo "Missing prepared Shanghai data. Run: DATASET=isp CITY=Shanghai ./scripts/hybrid_pipeline.sh prepare" >&2
  exit 2
}

if [[ "${RESAMPLE:-0}" == "1" || ! -f "$SAMPLE_DIR/sample_manifest.json" ]]; then
  "$PYTHON_BIN" -m hybrid.sample_split \
    --validation "$base_validation" --test "$base_test" \
    --output-dir "$SAMPLE_DIR" --fraction 0.5 --seed "$SAMPLE_SEED"
fi

export DATASET=isp
export CITY=Shanghai
export OLLAMA_MODEL
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export VALIDATION_FILE="$SAMPLE_DIR/validation.jsonl"
export TEST_FILE="$SAMPLE_DIR/test.jsonl"
export OUTPUT_DIR
export TOP_K="${TOP_K:-10}"
export TOP_M="${TOP_M:-5}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-3}"
export LLM_RETRIES="${LLM_RETRIES:-2}"
export LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}"
export OLLAMA_RESTART_ATTEMPTS="${OLLAMA_RESTART_ATTEMPTS:-3}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

echo "Shanghai 50% RQ pipeline"
echo "sample=$SAMPLE_DIR seed=$SAMPLE_SEED model=$OLLAMA_MODEL"
echo "output=$OUTPUT_DIR"

if [[ "${REPORT_ONLY:-0}" != "1" ]]; then
  ./scripts/hybrid_pipeline.sh run
fi

if [[ "${HYBRID_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run complete; report generation skipped."
  exit 0
fi

"$PYTHON_BIN" -m hybrid.rq_report \
  --results "$OUTPUT_DIR" --manifest "$SAMPLE_DIR/sample_manifest.json"
echo "Report: $OUTPUT_DIR/RQ_REPORT.md"
