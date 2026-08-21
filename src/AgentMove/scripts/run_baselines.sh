#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
TARGET="${1:-audit}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
LIMIT="${QUERY_LIMIT:-200}"
LOG_DIR="${LOG_DIR:-results/logs/baselines/${MODEL//[:\/]/-}}"
mkdir -p "$LOG_DIR"

run_logged() {
  local name="$1"; shift
  echo "[$(date -u +%FT%TZ)] start $name" | tee -a "$LOG_DIR/manifest.log"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "[$(date -u +%FT%TZ)] complete $name" | tee -a "$LOG_DIR/manifest.log"
}

case "$TARGET" in
  audit)
    QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/tist2015_pipeline.sh audit
    QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/run_tist2015_llm_only_200.sh audit
    QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/run_tist2015_agentmove_200.sh audit
    ;;
  llm-zs) run_logged llm-zs env QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/run_tist2015_llm_only_200.sh pending ;;
  llm-mob) run_logged llm-mob env QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/run_tist2015_llm_mob_200.sh pending ;;
  agentmove) run_logged agentmove env QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/run_tist2015_agentmove_200.sh pending ;;
  hybrid) run_logged hybrid env QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/tist2015_pipeline.sh run ;;
  all)
    "$0" llm-zs; "$0" llm-mob; "$0" agentmove; "$0" hybrid
    ;;
  aggregate)
    .venv/bin/python -m hybrid.aggregate_runs --results-root results --output results/run_index.json
    ;;
  *) echo "Usage: $0 <audit|llm-zs|llm-mob|agentmove|hybrid|all|aggregate>" >&2; exit 2 ;;
esac
