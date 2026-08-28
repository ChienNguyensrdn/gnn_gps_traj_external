#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"
SEED="${SEED:-42}"; BASE="data/hybrid/TIST2015/$CITY"; ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq11"
RQ10_ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq10"
RQ7_ROOT="results/beliefmove-evo/artifacts/full/$CITY/E5-dual/correct/seed-$SEED"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  require_file "$BASE/getnext/train.csv"; require_file "$BASE/getnext/val.csv"; require_file "$BASE/getnext/test.csv"
  require_file "configs/beliefmove_evo/calibration.json"
  echo "city=$CITY seed=$SEED output=$ROOT"
  echo "calibration_fit=validation evaluation=test protocols=last-query,all-prefix"
}
evaluate_distillation() {
  audit; local variant checkpoint output
  for variant in none gru transformer; do
    checkpoint="$RQ10_ROOT/students/$variant/seed-$SEED/best.pt"; output="$ROOT/distillation/$variant/seed-$SEED"
    require_file "$checkpoint"
    if [[ -f "$output/rq11.metrics.json" && -f "$output/before.predictions.npz" && -f "$output/after.predictions.npz" && "${FORCE:-0}" != 1 ]]; then
      echo "skip existing $output"; continue
    fi
    "$PYTHON_BIN" -m hybrid.rq11_calibration --checkpoint "$checkpoint" --validation-csv "$BASE/getnext/val.csv" \
      --test-csv "$BASE/getnext/test.csv" --output-dir "$output" --variant "$variant" --protocol last-query \
      --batch-size "${BATCH_SIZE:-128}" --device "${DEVICE:-auto}" --seed "$SEED"
  done
}
evaluate_bayesian() {
  audit; local checkpoint="$RQ7_ROOT/best.pt" rq7_metrics="$RQ7_ROOT/rq7/rq7.metrics.json" variant output
  require_file "$checkpoint"; require_file "$rq7_metrics"
  for variant in B0-static B3-dbn; do
    output="$ROOT/bayesian/$variant/seed-$SEED"
    if [[ -f "$output/rq11.metrics.json" && -f "$output/before.predictions.npz" && -f "$output/after.predictions.npz" && "${FORCE:-0}" != 1 ]]; then
      echo "skip existing $output"; continue
    fi
    "$PYTHON_BIN" -m hybrid.rq11_calibration --checkpoint "$checkpoint" --train-csv "$BASE/getnext/train.csv" \
      --validation-csv "$BASE/getnext/val.csv" --test-csv "$BASE/getnext/test.csv" --output-dir "$output" \
      --variant "$variant" --protocol all-prefix --rq7-metrics "$rq7_metrics" \
      --batch-size "${BATCH_SIZE:-128}" --device "${DEVICE:-auto}" --seed "$SEED"
  done
}
status() {
  local seed group variant path missing=0
  echo "RQ11 status city=$CITY root=$ROOT seeds=${RQ11_SEEDS:-42 43 44}"
  for seed in ${RQ11_SEEDS:-42 43 44}; do
    for group in distillation bayesian; do
      if [[ "$group" == distillation ]]; then variants="none gru transformer"; else variants="B0-static B3-dbn"; fi
      for variant in $variants; do
        for path in "$ROOT/$group/$variant/seed-$seed/rq11.metrics.json" \
                    "$ROOT/$group/$variant/seed-$seed/before.predictions.npz" \
                    "$ROOT/$group/$variant/seed-$seed/after.predictions.npz"; do
          if [[ -f "$path" ]]; then echo "ready   $path"; else echo "missing $path"; missing=$((missing + 1)); fi
        done
      done
    done
  done
  if (( missing > 0 )); then
    echo "RQ11 incomplete: $missing artifact(s) missing." >&2
    echo "Resume with: CITY=$CITY DEVICE=${DEVICE:-cuda} BATCH_SIZE=${BATCH_SIZE:-128} $0 run-seeds" >&2
    return 2
  fi
  echo "RQ11 complete: calibration metrics and paired predictions are ready."
}
run_seeds() {
  local seed
  for seed in ${RQ11_SEEDS:-42 43 44}; do
    CITY="$CITY" SEED="$seed" "$0" evaluate-distillation
    CITY="$CITY" SEED="$seed" "$0" evaluate-bayesian
  done
}
aggregate() {
  status || { echo "Aggregation stopped by RQ11 publication gate." >&2; exit 2; }
  "$PYTHON_BIN" -m hybrid.rq11_aggregate --root "$ROOT" --seeds ${RQ11_SEEDS:-42 43 44} \
    --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" --ece-iterations "${ECE_BOOTSTRAP_ITERATIONS:-1000}" \
    --output results/beliefmove-evo/aggregated/rq11_summary.json --markdown ../../ideas/results_rq11.md
}
case "$ACTION" in
  audit) audit ;; status) status ;; evaluate-distillation) evaluate_distillation ;;
  evaluate-bayesian) evaluate_bayesian ;; run-seeds) run_seeds ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|status|evaluate-distillation|evaluate-bayesian|run-seeds|aggregate>" >&2; exit 2 ;;
esac
