#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"; SEED="${SEED:-42}"
PROFILE="${PROFILE:-batch-256}"
case "$PROFILE" in
  batch-1) PROFILE_BATCH=1; PROFILE_QUERY_LIMIT="${QUERY_LIMIT:-2000}" ;;
  batch-256) PROFILE_BATCH=256; PROFILE_QUERY_LIMIT="${QUERY_LIMIT:-}" ;;
  *) echo "Invalid PROFILE=$PROFILE (expected batch-1 or batch-256)" >&2; exit 2 ;;
esac
if [[ -n "${BATCH_SIZE:-}" && "$BATCH_SIZE" != "$PROFILE_BATCH" ]]; then
  echo "PROFILE=$PROFILE requires BATCH_SIZE=$PROFILE_BATCH, got $BATCH_SIZE" >&2; exit 2
fi
BASE="data/hybrid/TIST2015/$CITY"; SCOPE=full; [[ -n "${MAX_BATCHES:-}" ]] && SCOPE=smoke
ROOT="results/beliefmove-evo/artifacts/$SCOPE/$CITY/rq12/$PROFILE"; FULL_ROOT="results/beliefmove-evo/artifacts/full/$CITY"
RQ8_SUMMARY="${RQ8_SUMMARY:-results/beliefmove-evo/aggregated/rq8_summary.json}"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  require_file "$BASE/getnext/test.csv"; require_file "configs/beliefmove_evo/efficiency.json"
  echo "city=$CITY seed=$SEED scope=$SCOPE profile=$PROFILE output=$ROOT"
  echo "batch=$PROFILE_BATCH query_limit=${PROFILE_QUERY_LIMIT:-full} repeats=${BENCHMARK_REPEATS:-5} warmup=${WARMUP_BATCHES:-10} device=${DEVICE:-auto}"
}
run_one() {
  local group="$1" variant="$2" checkpoint="$3" metrics="$4" quality_variant="${5:-}" output
  output="$ROOT/$group/$variant/seed-$SEED/rq12.metrics.json"
  require_file "$checkpoint"; require_file "$metrics"
  [[ -f "$output" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $output"; return; }
  local extra=(--batch-size "$PROFILE_BATCH" --warmup-batches "${WARMUP_BATCHES:-10}"
    --repeats "${BENCHMARK_REPEATS:-5}" --device "${DEVICE:-auto}" --seed "$SEED")
  [[ -n "$PROFILE_QUERY_LIMIT" ]] && extra+=(--query-limit "$PROFILE_QUERY_LIMIT")
  [[ -n "${MAX_BATCHES:-}" ]] && extra+=(--max-batches "$MAX_BATCHES")
  local args=(--checkpoint "$checkpoint" --test-csv "$BASE/getnext/test.csv" --quality-metrics "$metrics"
    --output "$output" --variant "$variant" --protocol last-query)
  [[ -n "$quality_variant" ]] && args+=(--quality-variant "$quality_variant")
  "$PYTHON_BIN" -m hybrid.rq12_efficiency "${args[@]}" "${extra[@]}"
}
benchmark_neural() {
  audit
  run_one neural teacher-gru "$FULL_ROOT/rq10/teachers/gru/seed-$SEED/best.pt" "$FULL_ROOT/rq10/teachers/gru/seed-$SEED/test.metrics.json"
  run_one neural teacher-transformer "$FULL_ROOT/rq10/teachers/transformer/seed-$SEED/best.pt" "$FULL_ROOT/rq10/teachers/transformer/seed-$SEED/test.metrics.json"
  local variant
  for variant in none gru transformer; do
    run_one neural "student-$variant" "$FULL_ROOT/rq10/students/$variant/seed-$SEED/best.pt" "$FULL_ROOT/rq10/students/$variant/seed-$SEED/test.metrics.json"
  done
}
benchmark_bayesian() {
  audit; require_file "$BASE/getnext/train.csv"
  local checkpoint="$FULL_ROOT/E5-dual/correct/seed-$SEED/best.pt" metrics="$FULL_ROOT/E5-dual/correct/seed-$SEED/rq7/rq7.metrics.json" variant output
  require_file "$checkpoint"; require_file "$metrics"
  local extra=(--batch-size "$PROFILE_BATCH" --warmup-batches "${WARMUP_BATCHES:-10}"
    --repeats "${BENCHMARK_REPEATS:-5}" --device "${DEVICE:-auto}" --seed "$SEED")
  [[ -n "$PROFILE_QUERY_LIMIT" ]] && extra+=(--query-limit "$PROFILE_QUERY_LIMIT")
  [[ -n "${MAX_BATCHES:-}" ]] && extra+=(--max-batches "$MAX_BATCHES")
  for variant in B0-static B3-dbn; do
    output="$ROOT/bayesian/$variant/seed-$SEED/rq12.metrics.json"
    [[ -f "$output" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $output"; continue; }
    "$PYTHON_BIN" -m hybrid.rq12_efficiency --checkpoint "$checkpoint" --train-csv "$BASE/getnext/train.csv" \
      --test-csv "$BASE/getnext/test.csv" --rq7-metrics "$metrics" --quality-metrics "$metrics" \
      --quality-variant "$variant" --output "$output" --variant "$variant" --protocol all-prefix "${extra[@]}"
  done
}
status() {
  local seed profile variant path missing=0
  echo "RQ12 status city=$CITY root=results/beliefmove-evo/artifacts/full/$CITY/rq12 seeds=${RQ12_SEEDS:-42 43 44}"
  for profile in batch-1 batch-256; do
    for seed in ${RQ12_SEEDS:-42 43 44}; do
      for variant in teacher-gru teacher-transformer student-none student-gru student-transformer; do
        path="results/beliefmove-evo/artifacts/full/$CITY/rq12/$profile/neural/$variant/seed-$seed/rq12.metrics.json"
        [[ -f "$path" ]] && echo "ready   $path" || { echo "missing $path"; missing=$((missing + 1)); }
      done
      for variant in B0-static B3-dbn; do
        path="results/beliefmove-evo/artifacts/full/$CITY/rq12/$profile/bayesian/$variant/seed-$seed/rq12.metrics.json"
        [[ -f "$path" ]] && echo "ready   $path" || { echo "missing $path"; missing=$((missing + 1)); }
      done
    done
  done
  [[ -f "$RQ8_SUMMARY" ]] && echo "ready   $RQ8_SUMMARY" || { echo "missing $RQ8_SUMMARY"; missing=$((missing + 1)); }
  (( missing == 0 )) || { echo "RQ12 incomplete: $missing artifact(s) missing." >&2; return 2; }
  echo "RQ12 complete: full neural/Bayesian benchmarks and bounded RQ8 source are ready."
}
run_seeds() {
  local seed
  for seed in ${RQ12_SEEDS:-42 43 44}; do
    CITY="$CITY" SEED="$seed" "$0" benchmark-neural
    CITY="$CITY" SEED="$seed" "$0" benchmark-bayesian
  done
}
run_profiles() {
  local profile
  for profile in batch-1 batch-256; do
    CITY="$CITY" PROFILE="$profile" "$0" run-seeds
  done
}
aggregate() {
  status || { echo "Aggregation stopped by RQ12 gate." >&2; exit 2; }
  local extra=(); [[ "${ALLOW_GPU_CONTENTION:-0}" == 1 ]] && extra+=(--allow-contention)
  "$PYTHON_BIN" -m hybrid.rq12_aggregate --root "results/beliefmove-evo/artifacts/full/$CITY/rq12" \
    --seeds ${RQ12_SEEDS:-42 43 44} --rq8-summary "$RQ8_SUMMARY" \
    --output results/beliefmove-evo/aggregated/rq12_summary.json --markdown ../../ideas/results_rq12.md "${extra[@]}"
}
case "$ACTION" in
  audit) audit ;; status) status ;; benchmark-neural) benchmark_neural ;; benchmark-bayesian) benchmark_bayesian ;;
  run-seeds) run_seeds ;; run-profiles) run_profiles ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|status|benchmark-neural|benchmark-bayesian|run-seeds|run-profiles|aggregate>" >&2; exit 2 ;;
esac
