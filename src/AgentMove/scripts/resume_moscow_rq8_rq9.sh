#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-status}"
CITY="Moscow"
LIMIT="${LLM_LIMIT:-200}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${MODEL//[:\/]/-}"
RQ8_ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq8/$MODEL_SLUG/limit-$LIMIT"
RQ9_ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq9/$MODEL_SLUG/limit-$LIMIT"
HYBRID_ROOT="results/tist2015-hybrid/$MODEL_SLUG/limit-$LIMIT/no-osm/$CITY"
RANDOM_SEEDS="${RQ8_RANDOM_SEEDS:-$(seq 42 91)}"
RQ9_VARIANTS="memory-true memory-shuffled memory-random-user memory-none context-shuffled context-random-poi context-none"
LOG_DIR="${LOG_DIR:-results/logs/tist2015/$MODEL_SLUG}"

rq8_seed_complete() {
  local seed="$1" name
  for name in rq8.metrics.json never.test.predictions.npz always.test.predictions.npz \
              entropy.test.predictions.npz margin.test.predictions.npz \
              random-budget-matched.test.predictions.npz; do
    [[ -f "$RQ8_ROOT/seed-$seed/$name" ]] || return 1
  done
}

rq9_variant_complete() {
  local variant="$1"
  [[ -f "$RQ9_ROOT/$variant/metrics.json" && -f "$RQ9_ROOT/$variant/predictions.jsonl" ]]
}

audit_inputs() {
  local path missing=0
  for path in .venv/bin/python \
    "data/hybrid/TIST2015/$CITY/neural_cgm/validation.jsonl" \
    "data/hybrid/TIST2015/$CITY/neural_cgm/test.jsonl" \
    "$HYBRID_ROOT/evidence_cache.jsonl" "$HYBRID_ROOT/calibration.json"; do
    [[ -f "$path" || -x "$path" ]] && echo "ready   $path" || { echo "missing $path"; missing=$((missing + 1)); }
  done
  (( missing == 0 )) || return 2
}

status() {
  local seed variant missing8=0 missing9=0 complete8=0 complete9=0
  for seed in $RANDOM_SEEDS; do
    if rq8_seed_complete "$seed"; then complete8=$((complete8 + 1));
    else echo "RQ8 missing/incomplete seed-$seed"; missing8=$((missing8 + 1)); fi
  done
  for variant in $RQ9_VARIANTS; do
    if rq9_variant_complete "$variant"; then complete9=$((complete9 + 1));
    else echo "RQ9 missing/incomplete $variant"; missing9=$((missing9 + 1)); fi
  done
  echo "RQ8 complete=$complete8/$(wc -w <<<"$RANDOM_SEEDS" | tr -d ' ') missing_seeds=$missing8"
  echo "RQ9 complete=$complete9/7 missing_variants=$missing9"
  (( missing8 == 0 && missing9 == 0 )) || return 2
}

run_rq8() {
  local seed
  if [[ ! -f "$RQ8_ROOT/always-cache/validation/predictions.jsonl" || \
        ! -f "$RQ8_ROOT/always-cache/test/predictions.jsonl" ]]; then
    CITY="$CITY" RQ8_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" HYBRID_RUN_DIR="$HYBRID_ROOT" \
      ./scripts/rq8_routing.sh collect
  else
    echo "skip existing RQ8 always-cache"
  fi
  for seed in $RANDOM_SEEDS; do
    if rq8_seed_complete "$seed"; then
      echo "skip complete RQ8 seed-$seed"
    else
      echo "resume RQ8 seed-$seed"
      CITY="$CITY" SEED="$seed" RQ8_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" \
        HYBRID_RUN_DIR="$HYBRID_ROOT" ./scripts/rq8_routing.sh evaluate
    fi
  done
}

run_rq9() {
  local pending=0 variant
  for variant in $RQ9_VARIANTS; do rq9_variant_complete "$variant" || pending=$((pending + 1)); done
  if (( pending == 0 )); then echo "skip complete RQ9"; return 0; fi
  echo "resume RQ9 pending_variants=$pending"
  CITY="$CITY" SEED="${RQ9_SEED:-42}" RQ9_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" \
    HYBRID_RUN_DIR="$HYBRID_ROOT" ./scripts/rq9_semantic.sh collect
}

run_all() {
  audit_inputs
  mkdir -p "$LOG_DIR"
  run_rq8
  run_rq9
  status
  echo "Moscow RQ8/RQ9 complete. Refresh 12-city status with:"
  echo "LLM_LIMIT=$LIMIT OLLAMA_MODEL=$MODEL ./scripts/run_all_cities_rqs.sh status"
}

case "$ACTION" in
  audit) audit_inputs; status || true ;;
  rq8) audit_inputs; run_rq8 ;;
  rq9) audit_inputs; run_rq9 ;;
  run) mkdir -p "$LOG_DIR"; run_all 2>&1 | tee -a "$LOG_DIR/resume-Moscow-rq8-rq9.log" ;;
  status) status ;;
  *) echo "Usage: $0 <audit|rq8|rq9|run|status>" >&2; exit 2 ;;
esac
