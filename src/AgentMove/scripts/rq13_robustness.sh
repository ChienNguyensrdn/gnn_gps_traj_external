#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"; SEED="${SEED:-42}"
BASE="data/hybrid/TIST2015/$CITY"; ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq13"
CHECKPOINT_ROOT="results/beliefmove-evo/artifacts/full/$CITY/E5-dual/correct"
VARIANTS=(clean gps-drop-25 gps-drop-50 time-noise-30m time-noise-60m position-noise-200m position-noise-500m context-missing context-wrong)

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  require_file "$BASE/getnext/test.csv"; require_file "configs/beliefmove_evo/robustness.json"
  require_file "$CHECKPOINT_ROOT/seed-$SEED/best.pt"
  echo "city=$CITY seed=$SEED output=$ROOT"
  echo "checkpoint=frozen-E5-dual protocol=last-query target=unchanged variants=${#VARIANTS[@]}"
}
evaluate_seed() {
  audit; local variant output
  for variant in "${VARIANTS[@]}"; do
    output="$ROOT/$variant/seed-$SEED"
    if [[ -f "$output/rq13.metrics.json" && -f "$output/test.predictions.npz" && "${FORCE:-0}" != 1 ]]; then
      echo "skip existing $output"; continue
    fi
    "$PYTHON_BIN" -m hybrid.rq13_robustness --checkpoint "$CHECKPOINT_ROOT/seed-$SEED/best.pt" \
      --test-csv "$BASE/getnext/test.csv" --output-dir "$output" --variant "$variant" \
      --batch-size "${BATCH_SIZE:-256}" --device "${DEVICE:-auto}" --seed "$SEED"
  done
}
run_seeds() {
  local seed
  for seed in ${RQ13_SEEDS:-42 43 44}; do CITY="$CITY" SEED="$seed" "$0" evaluate-seed; done
}
status() {
  local seed variant path missing=0
  echo "RQ13 status city=$CITY root=$ROOT seeds=${RQ13_SEEDS:-42 43 44}"
  for seed in ${RQ13_SEEDS:-42 43 44}; do
    for variant in "${VARIANTS[@]}"; do
      for path in "$ROOT/$variant/seed-$seed/rq13.metrics.json" "$ROOT/$variant/seed-$seed/test.predictions.npz"; do
        [[ -f "$path" ]] && echo "ready   $path" || { echo "missing $path"; missing=$((missing + 1)); }
      done
    done
  done
  (( missing == 0 )) || { echo "RQ13 incomplete: $missing artifact(s) missing." >&2; return 2; }
  echo "RQ13 complete: all robustness metrics and paired predictions are ready."
}
aggregate() {
  status || { echo "Aggregation stopped by RQ13 publication gate." >&2; exit 2; }
  "$PYTHON_BIN" -m hybrid.rq13_aggregate --root "$ROOT" --seeds ${RQ13_SEEDS:-42 43 44} \
    --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" \
    --output results/beliefmove-evo/aggregated/rq13_summary.json --markdown ../../ideas/results_rq13.md
}
case "$ACTION" in
  audit) audit ;; evaluate-seed) evaluate_seed ;; run-seeds) run_seeds ;; status) status ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|evaluate-seed|run-seeds|status|aggregate>" >&2; exit 2 ;;
esac
