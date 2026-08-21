#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${OLLAMA_MODEL//[:\/]/-}"
TEST_FILE="${TEST_FILE:-data/hybrid/Shanghai/neural_cgm/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/llm-only/agentmove-faithful-shanghai-200/${MODEL_SLUG}}"
RESTARTS="${OLLAMA_RESTART_ATTEMPTS:-5}"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"
cmd=("$PYTHON_BIN" -m hybrid.llm_only --test "$TEST_FILE" --output-dir "$OUTPUT_DIR"
     --platform Ollama --model-name "$OLLAMA_MODEL" --retries "${LLM_RETRIES:-2}"
     --agentmove-sample "${TEST_LIMIT:-${AGENTMOVE_SAMPLE:-200}}")
if [[ "${HYBRID_DRY_RUN:-0}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi

./scripts/start_ollama.sh

attempt=0
while :; do
  set +e; "${cmd[@]}"; status=$?; set -e
  [[ "$status" -eq 0 ]] && break
  [[ "$status" -eq 75 ]] || exit "$status"
  attempt=$((attempt + 1)); [[ "$attempt" -le "$RESTARTS" ]] || exit 75
  ./scripts/start_ollama.sh
done
echo "LLM-only results: $OUTPUT_DIR"
