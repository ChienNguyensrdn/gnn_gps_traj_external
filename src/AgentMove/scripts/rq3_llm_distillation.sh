#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-audit}"; PY="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"
SEED="${SEED:-42}"; LIMIT="${RQ3_LIMIT:-200}"; MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${MODEL//[:\/]/-}"; BASE="data/hybrid/TIST2015/$CITY"
NEURAL="$BASE/neural_cgm"; HYBRID="${HYBRID_RUN_DIR:-results/tist2015-hybrid/$MODEL_SLUG/limit-$LIMIT/no-osm/$CITY}"
ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq3/$MODEL_SLUG/limit-$LIMIT/seed-$SEED"
req(){ [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit(){
  [[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }
  req "$BASE/getnext/train.csv"; req "$NEURAL/validation.jsonl"; req "$NEURAL/test.jsonl"
  req "$HYBRID/evidence_cache.jsonl"; req "$HYBRID/calibration.json"
  echo "RQ3 city=$CITY model=$MODEL limit=$LIMIT seed=$SEED"
  echo "protocol=bounded matched last-query fit=train,validation eval=test"
  echo "hybrid_cache=$HYBRID"; echo "output=$ROOT"
}
evaluate(){
  audit
  [[ -f "$ROOT/rq3.metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $ROOT/rq3.metrics.json"; return; }
  "$PY" -m hybrid.rq3_distillation --train-csv "$BASE/getnext/train.csv" \
    --validation "$NEURAL/validation.jsonl" --test "$NEURAL/test.jsonl" \
    --evidence-cache "$HYBRID/evidence_cache.jsonl" --calibration "$HYBRID/calibration.json" \
    --output-dir "$ROOT" --city "$CITY" --seed "$SEED" --limit "$LIMIT" \
    --weight-grid ${RQ3_WEIGHT_GRID:-0 0.25 0.5 0.75 1.0}
}
status(){
  local missing=0 variant
  req "$ROOT/rq3.metrics.json"
  for variant in M1-data-only M2-llm M3-quantitative M4-both; do
    [[ -f "$ROOT/$variant.test.predictions.npz" ]] || { echo "missing $ROOT/$variant.test.predictions.npz"; missing=$((missing+1)); }
  done
  ((missing==0)) && echo "RQ3 complete: metrics and paired predictions are ready."
  ((missing==0))
}
aggregate(){
  status
  "$PY" -m hybrid.rq3_aggregate --root "$ROOT" --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" \
    --output results/beliefmove-evo/aggregated/rq3_summary.json --markdown ../../ideas/results_rq3.md
}
case "$ACTION" in audit) audit;; evaluate) evaluate;; status) status;; aggregate) aggregate;;
  *) echo "Usage: $0 <audit|evaluate|status|aggregate>" >&2; exit 2;; esac
