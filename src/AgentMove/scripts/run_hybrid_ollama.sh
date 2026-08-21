#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
CITY="${CITY:-Shanghai}"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"

.venv/bin/python -m hybrid.cli \
  --validation "data/hybrid/$CITY/validation.jsonl" \
  --test "data/hybrid/$CITY/test.jsonl" \
  --output-dir "results/hybrid/$CITY-ollama-${MODEL//[:\/]/-}" \
  --top-k "${TOP_K:-10}" \
  --top-m "${TOP_M:-5}" \
  --extractor llm \
  --platform Ollama \
  --model-name "$MODEL" \
  --llm-batch-size "${LLM_BATCH_SIZE:-3}" \
  --llm-retries "${LLM_RETRIES:-2}"
