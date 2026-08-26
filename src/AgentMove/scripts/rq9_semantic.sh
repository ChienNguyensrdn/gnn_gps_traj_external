#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"; MODEL_SLUG="${MODEL//[:\/]/-}"; LIMIT="${RQ9_LIMIT:-200}"; SEED="${SEED:-42}"
BASE="data/hybrid/TIST2015/$CITY/neural_cgm"
HYBRID_RUN_DIR="${HYBRID_RUN_DIR:-results/tist2015-hybrid/$MODEL_SLUG/limit-$LIMIT/no-osm/$CITY}"
ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq9/$MODEL_SLUG/limit-$LIMIT"
VARIANTS=(memory-true memory-shuffled memory-random-user memory-none context-shuffled context-random-poi context-none)

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  require_file "$BASE/test.jsonl"; require_file "$HYBRID_RUN_DIR/calibration.json"
  echo "city=$CITY model=$MODEL limit=$LIMIT seed=$SEED"; echo "output=$ROOT"
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null && echo "ollama=ready" || echo "ollama=offline (required for collect)"
}
collect() {
  audit; ./scripts/start_ollama.sh
  OLLAMA_BASE_URL="http://127.0.0.1:11434/v1" ./scripts/test_ollama.sh "$MODEL"
  export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1" OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  local variant
  for variant in "${VARIANTS[@]}"; do
    "$PYTHON_BIN" -m hybrid.rq9_semantic_rerank --test "$BASE/test.jsonl" \
      --calibration "$HYBRID_RUN_DIR/calibration.json" --output-dir "$ROOT/$variant" \
      --variant "$variant" --model-name "$MODEL" --limit "$LIMIT" --seed "$SEED" \
      --top-k "${TOP_K:-10}" --retries "${LLM_RETRIES:-2}"
  done
}
aggregate() {
  "$PYTHON_BIN" -m hybrid.rq9_aggregate --root "$ROOT" --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" \
    --output results/beliefmove-evo/aggregated/rq9_summary.json --markdown ../../ideas/results_rq9.md
}
case "$ACTION" in audit) audit ;; collect) collect ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|collect|aggregate>" >&2; exit 2 ;; esac
