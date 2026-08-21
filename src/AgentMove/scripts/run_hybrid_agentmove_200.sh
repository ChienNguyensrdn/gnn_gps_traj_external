#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${MODEL//[:\/]/-}"
DATA_DIR="${DATA_DIR:-data/hybrid/Shanghai/neural_cgm/agentmove-200}"
BASE_RESULTS="${BASE_RESULTS:-results/hybrid/shanghai-neural-cgm-50-seed42/qwen2-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hybrid/agentmove-faithful-shanghai-200/${MODEL_SLUG}}"

"$PYTHON_BIN" -m hybrid.agentmove_protocol \
  --validation data/hybrid/Shanghai/neural_cgm/sample-50-seed42/validation.jsonl \
  --test data/hybrid/Shanghai/neural_cgm/test.jsonl \
  --output-dir "$DATA_DIR" --test-users "${AGENTMOVE_SAMPLE:-200}" \
  --seed-cache-from "$BASE_RESULTS" --result-dir "$OUTPUT_DIR"

export DATASET=isp CITY=Shanghai OLLAMA_MODEL="$MODEL"
export VALIDATION_FILE="$DATA_DIR/validation.jsonl" TEST_FILE="$DATA_DIR/test.jsonl"
export OUTPUT_DIR TOP_K="${TOP_K:-10}" TOP_M="${TOP_M:-5}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-3}" LLM_RETRIES="${LLM_RETRIES:-2}"
export LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
./scripts/hybrid_pipeline.sh run
