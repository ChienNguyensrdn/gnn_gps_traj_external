#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Approximately 50% of the prepared ISP-Shanghai queries:
# validation: ceil(1042 / 2) = 521
# test:       ceil(2403 / 2) = 1202
export DATASET="${DATASET:-isp}"
export CITY="${CITY:-Shanghai}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"

# Keep the paper/full-run candidate and memory settings. Only query count is
# reduced, making this run directly extensible to the full experiment.
export TOP_K="${TOP_K:-10}"
export TOP_M="${TOP_M:-5}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-3}"
export LLM_RETRIES="${LLM_RETRIES:-2}"
export LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}"
export VALIDATION_LIMIT="${VALIDATION_LIMIT:-521}"
export TEST_LIMIT="${TEST_LIMIT:-1202}"

export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"

echo "Running approximately 50% of the ISP-Shanghai Hybrid experiment"
echo "validation=$VALIDATION_LIMIT/1042 test=$TEST_LIMIT/2403"
echo "model=$OLLAMA_MODEL top_k=$TOP_K top_m=$TOP_M batch=$LLM_BATCH_SIZE"

./scripts/hybrid_pipeline.sh run
