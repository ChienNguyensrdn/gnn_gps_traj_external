#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"
LIMIT="${RQ1_LIMIT:-200}"; RQ10_ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq10"
MARKOV_SUMMARY="${MARKOV_SUMMARY:-results/tist2015-markov/limit-$LIMIT/tist2015_markov_summary.json}"
AGENTMOVE_SUMMARY="${AGENTMOVE_SUMMARY:-results/tist2015-agentmove-original-no-osm/limit-$LIMIT/tist2015_agentmove_summary.json}"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; return 2; }; }
quantitative_status() {
  local seed group variant path missing=0
  for seed in ${RQ1_SEEDS:-42 43 44}; do
    for group in teachers students; do
      [[ "$group" == teachers ]] && variants="gru transformer" || variants="none"
      for variant in $variants; do
        path="$RQ10_ROOT/$group/$variant/seed-$seed/test.metrics.json"
        [[ -f "$path" ]] && echo "ready   $path" || { echo "missing $path"; missing=$((missing + 1)); }
      done
    done
  done
  (( missing == 0 ))
}
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  echo "RQ1 city=$CITY seeds=${RQ1_SEEDS:-42 43 44} bounded_limit=$LIMIT"
  quantitative_status || true
  [[ -f "$MARKOV_SUMMARY" ]] && echo "ready   $MARKOV_SUMMARY" || echo "missing $MARKOV_SUMMARY"
  [[ -f "$AGENTMOVE_SUMMARY" ]] && echo "ready   $AGENTMOVE_SUMMARY" || echo "missing $AGENTMOVE_SUMMARY"
  echo "Protocols remain separated: Tokyo full-test quantitative vs bounded 12-city baselines."
}
run_bounded() {
  QUERY_LIMIT="$LIMIT" ./scripts/run_tist2015_markov_200.sh pending
  QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}" ./scripts/run_tist2015_agentmove_200.sh pending
}
aggregate() {
  quantitative_status || { echo "RQ1 quantitative gate incomplete; run RQ10 first." >&2; exit 2; }
  "$PYTHON_BIN" -m hybrid.rq1_reproducibility --rq10-root "$RQ10_ROOT" --seeds ${RQ1_SEEDS:-42 43 44} \
    --markov-summary "$MARKOV_SUMMARY" --agentmove-summary "$AGENTMOVE_SUMMARY" --query-limit "$LIMIT" \
    --output results/beliefmove-evo/aggregated/rq1_summary.json --markdown ../../ideas/results_rq1.md
}
case "$ACTION" in
  audit|status) audit ;; run-bounded) run_bounded ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|status|run-bounded|aggregate>" >&2; exit 2 ;;
esac
