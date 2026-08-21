#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/tist2015_common.sh"

TARGET="${1:-audit}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="$(tist2015_agentmove_root)"
cd "$ROOT"

tist2015_require_positive_integer QUERY_LIMIT "$QUERY_LIMIT"
MODEL_SLUG="$(tist2015_model_slug "$OLLAMA_MODEL")"
HYBRID_ROOT="results/tist2015-hybrid/$MODEL_SLUG/limit-$QUERY_LIMIT/no-osm"
LLM_ROOT="results/tist2015-llm-only/$MODEL_SLUG/limit-$QUERY_LIMIT/llm-zs"
TABLE_ROOT="results/tist2015-table2/$MODEL_SLUG/limit-$QUERY_LIMIT"

audit() {
  local city hybrid_metrics llm_metrics llm_cache count
  echo "Table II completion audit"
  echo "model=$OLLAMA_MODEL limit=$QUERY_LIMIT ollama=http://127.0.0.1:11434/v1"
  for city in "${TIST2015_CITIES[@]}"; do
    hybrid_metrics="$HYBRID_ROOT/$city/full/metrics.json"
    llm_metrics="$LLM_ROOT/$city/metrics.json"
    llm_cache="$LLM_ROOT/$city/predictions.jsonl"
    if [[ -f "$llm_metrics" ]]; then
      count="$(wc -l < "$llm_cache" 2>/dev/null || echo 0)"
      printf '%-14s hybrid=%-8s llm-zs=complete(%s)\n' "$city" \
        "$([[ -f "$hybrid_metrics" ]] && echo complete || echo pending)" "$count"
    elif [[ -f "$llm_cache" ]]; then
      count="$(wc -l < "$llm_cache")"
      printf '%-14s hybrid=%-8s llm-zs=partial(%s/%s)\n' "$city" \
        "$([[ -f "$hybrid_metrics" ]] && echo complete || echo pending)" "$count" "$QUERY_LIMIT"
    else
      printf '%-14s hybrid=%-8s llm-zs=pending\n' "$city" \
        "$([[ -f "$hybrid_metrics" ]] && echo complete || echo pending)"
    fi
  done
}

run_hybrid_pending() {
  local city
  for city in "${TIST2015_CITIES[@]}"; do
    if [[ ! -f "$HYBRID_ROOT/$city/full/metrics.json" ]]; then
      QUERY_LIMIT="$QUERY_LIMIT" OLLAMA_MODEL="$OLLAMA_MODEL" \
        ./scripts/run_tist2015_city_200.sh "$city"
    fi
  done
}

run_llmzs_pending() {
  QUERY_LIMIT="$QUERY_LIMIT" OLLAMA_MODEL="$OLLAMA_MODEL" \
    ./scripts/run_tist2015_llm_only_200.sh pending
}

run_llmmob_pending() {
  QUERY_LIMIT="$QUERY_LIMIT" OLLAMA_MODEL="$OLLAMA_MODEL" \
    ./scripts/run_tist2015_llm_mob_200.sh pending
}

aggregate() {
  "$PYTHON_BIN" -m hybrid.tist2015_table2_aggregate \
    --hybrid-root "$HYBRID_ROOT" \
    --llm-root "$LLM_ROOT" \
    --output-dir "$TABLE_ROOT" \
    --model "$OLLAMA_MODEL" \
    --query-limit "$QUERY_LIMIT"
  echo "Summary: $TABLE_ROOT/tist2015_table2_summary.json"
  echo "LaTeX rows: $TABLE_ROOT/tist2015_table2_rows.tex"
}

case "$TARGET" in
  audit) audit ;;
  run-hybrid-pending) run_hybrid_pending ;;
  run-llmzs-pending) run_llmzs_pending ;;
  run-llmmob-pending) run_llmmob_pending ;;
  aggregate) aggregate ;;
  all)
    run_hybrid_pending
    run_llmzs_pending
    aggregate
    ;;
  *)
    echo "Usage: $0 <audit|run-hybrid-pending|run-llmzs-pending|run-llmmob-pending|aggregate|all>" >&2
    exit 2
    ;;
esac
