#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Small, stable Ollama experiment for Apple Silicon.
# Override any setting before the command when a larger run is desired.
export DATASET="${DATASET:-isp}"
export CITY="${CITY:-Shanghai}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"

# Reduce both LLM context size and the number of model calls.
export TOP_K="${TOP_K:-5}"
export TOP_M="${TOP_M:-3}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-1}"
export LLM_RETRIES="${LLM_RETRIES:-2}"
export LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}"

# Only a small subset is loaded for the experiment.
export VALIDATION_LIMIT="${VALIDATION_LIMIT:-5}"
export TEST_LIMIT="${TEST_LIMIT:-5}"

# Ollama resource controls for Apple Silicon.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"

echo "Running small Hybrid experiment on Apple Silicon"
echo "dataset=$DATASET city=$CITY model=$OLLAMA_MODEL"
echo "validation=$VALIDATION_LIMIT test=$TEST_LIMIT top_k=$TOP_K top_m=$TOP_M"

./scripts/hybrid_pipeline.sh run
