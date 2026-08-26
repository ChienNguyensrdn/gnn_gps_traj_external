#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CITY="${CITY:-Tokyo}"; SEED="${SEED:-42}"; LIMIT="${RQ8_LIMIT:-200}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"; MODEL_SLUG="${MODEL//[:\/]/-}"
BASE="data/hybrid/TIST2015/$CITY/neural_cgm"
HYBRID_RUN_DIR="${HYBRID_RUN_DIR:-results/tist2015-hybrid/$MODEL_SLUG/limit-$LIMIT/no-osm/$CITY}"
ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq8/$MODEL_SLUG/limit-$LIMIT"
ALWAYS="$ROOT/always-cache"; OUT="$ROOT/seed-$SEED"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  for file in "$BASE/validation.jsonl" "$BASE/test.jsonl" \
    "$HYBRID_RUN_DIR/evidence_cache.jsonl" "$HYBRID_RUN_DIR/calibration.json"; do require_file "$file"; done
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null && echo "ollama=ready" || echo "ollama=offline (required for collect)"
  echo "city=$CITY model=$MODEL limit=$LIMIT seed=$SEED"
  echo "hybrid_run=$HYBRID_RUN_DIR"
  echo "always_cache=$ALWAYS"; echo "output=$OUT"
}
collect() {
  audit; mkdir -p "$ALWAYS/validation" "$ALWAYS/test"
  ./scripts/start_ollama.sh
  OLLAMA_BASE_URL="http://127.0.0.1:11434/v1" ./scripts/test_ollama.sh "$MODEL"
  export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
  export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  for split in validation test; do
    "$PYTHON_BIN" -m hybrid.free_text_rerank --test "$BASE/$split.jsonl" \
      --evidence-cache "$HYBRID_RUN_DIR/evidence_cache.jsonl" --calibration "$HYBRID_RUN_DIR/calibration.json" \
      --output-dir "$ALWAYS/$split" --model-name "$MODEL" --platform Ollama \
      --top-k "${TOP_K:-10}" --top-m "${TOP_M:-5}" --retries "${LLM_RETRIES:-2}" --limit "$LIMIT"
  done
}
evaluate() {
  audit
  require_file "$ALWAYS/validation/predictions.jsonl"; require_file "$ALWAYS/test/predictions.jsonl"
  "$PYTHON_BIN" -m hybrid.rq8_routing --validation "$BASE/validation.jsonl" --test "$BASE/test.jsonl" \
    --validation-always "$ALWAYS/validation/predictions.jsonl" --test-always "$ALWAYS/test/predictions.jsonl" \
    --calibration "$HYBRID_RUN_DIR/calibration.json" --output-dir "$OUT" --limit "$LIMIT" --seed "$SEED"
}

aggregate() {
  "$PYTHON_BIN" -m hybrid.rq8_aggregate --root "$ROOT" --seeds ${RQ8_SEEDS:-42 43 44} \
    --output "results/beliefmove-evo/aggregated/rq8_summary.json" --markdown ../../ideas/results_rq8.md
}

case "$ACTION" in audit) audit ;; collect) collect ;; evaluate) evaluate ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|collect|evaluate|aggregate>" >&2; exit 2 ;; esac
