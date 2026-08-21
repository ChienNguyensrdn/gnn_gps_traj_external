#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tist2015_common.sh"

# Resumable AgentMove-faithful LLM-ZS baseline for TIST2015 Table II.
# Usage:
#   ./scripts/run_tist2015_llm_only_200.sh Tokyo
#   ./scripts/run_tist2015_llm_only_200.sh all
#   ./scripts/run_tist2015_llm_only_200.sh pending
#   ./scripts/run_tist2015_llm_only_200.sh audit
#   ./scripts/run_tist2015_llm_only_200.sh aggregate

TARGET="${1:-audit}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
LLM_RETRIES="${LLM_RETRIES:-2}"
LLM_BASELINE="${LLM_BASELINE:-llm-zs}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="$(tist2015_agentmove_root)"
cd "$ROOT"

tist2015_require_positive_integer QUERY_LIMIT "$QUERY_LIMIT"
tist2015_require_python "$PYTHON_BIN"

MODEL_SLUG="$(tist2015_model_slug "$OLLAMA_MODEL")"
case "$LLM_BASELINE" in
  llm-zs) PROMPT_TYPE="llmzs" ;;
  llm-mob) PROMPT_TYPE="llmmob" ;;
  *) echo "LLM_BASELINE must be llm-zs or llm-mob: $LLM_BASELINE" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-results/tist2015-llm-only/$MODEL_SLUG/limit-$QUERY_LIMIT/$LLM_BASELINE}"
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"

run_city() {
  local city="$1"
  local test_file="data/hybrid/TIST2015/$city/neural_cgm/test.jsonl"
  local output_dir="$OUTPUT_ROOT/$city"
  [[ -f "$test_file" ]] || { echo "Missing TIST2015 test split: $test_file" >&2; return 2; }
  echo "$LLM_BASELINE TIST2015 city=$city limit=$QUERY_LIMIT model=$OLLAMA_MODEL"
  echo "output=$output_dir"
  "$PYTHON_BIN" -m hybrid.llm_only \
    --test "$test_file" \
    --output-dir "$output_dir" \
    --platform Ollama \
    --model-name "$OLLAMA_MODEL" \
    --prompt-type "$PROMPT_TYPE" \
    --retries "$LLM_RETRIES" \
    --agentmove-sample 0 \
    --limit "$QUERY_LIMIT"
}

audit() {
  local city metrics predictions count
  echo "TIST2015 $LLM_BASELINE audit"
  echo "model=$OLLAMA_MODEL limit=$QUERY_LIMIT ollama=http://127.0.0.1:11434/v1"
  for city in "${TIST2015_CITIES[@]}"; do
    metrics="$OUTPUT_ROOT/$city/metrics.json"
    predictions="$OUTPUT_ROOT/$city/predictions.jsonl"
    if [[ -f "$metrics" ]]; then
      count="$(wc -l < "$predictions" 2>/dev/null || echo 0)"
      echo "complete $city predictions=$count"
    elif [[ -f "$predictions" ]]; then
      count="$(wc -l < "$predictions")"
      echo "partial  $city predictions=$count/$QUERY_LIMIT"
    else
      echo "pending  $city"
    fi
  done
}

aggregate() {
  "$PYTHON_BIN" -m hybrid.tist2015_llm_only_aggregate \
    --input-root "$OUTPUT_ROOT" \
    --model "$OLLAMA_MODEL" \
    --baseline "$LLM_BASELINE" \
    --query-limit "$QUERY_LIMIT"
  local file_slug="${LLM_BASELINE//-/_}"
  echo "Summary: $OUTPUT_ROOT/tist2015_${file_slug}_summary.json"
  echo "Table II cells: $OUTPUT_ROOT/tist2015_${file_slug}_table2_cells.tex"
}

case "$TARGET" in
  audit) audit ;;
  aggregate) aggregate ;;
  all)
    ./scripts/start_ollama.sh
    for city in "${TIST2015_CITIES[@]}"; do run_city "$city"; done
    aggregate
    ;;
  pending)
    ./scripts/start_ollama.sh
    for city in "${TIST2015_CITIES[@]}"; do
      [[ -f "$OUTPUT_ROOT/$city/metrics.json" ]] || run_city "$city"
    done
    aggregate
    ;;
  *)
    if ! tist2015_is_city "$TARGET"; then
      echo "Usage: $0 <audit|aggregate|pending|all|city>" >&2
      echo "Cities: ${TIST2015_CITIES[*]}" >&2
      exit 2
    fi
    ./scripts/start_ollama.sh
    run_city "$TARGET"
    ;;
esac
