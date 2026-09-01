#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-audit}"; PY="${PYTHON_BIN:-.venv/bin/python}"
CANONICAL_CITIES="Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow"
CITY_LIST="${CITIES:-$CANONICAL_CITIES}"; SEEDS="${RQ_SEEDS:-42 43 44}"
DEVICE="${DEVICE:-auto}"; BATCH_SIZE="${BATCH_SIZE:-128}"; LIMIT="${LLM_LIMIT:-200}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"; MODEL_SLUG="${MODEL//[:\/]/-}"

log(){ printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
require_python(){ [[ -x "$PY" ]] || { echo "Missing required Python: $PY" >&2; exit 2; }; }
validate_cities(){
  local city canonical
  for city in $CITY_LIST; do
    canonical=0
    for expected in $CANONICAL_CITIES; do [[ "$city" == "$expected" ]] && canonical=1; done
    ((canonical==1)) || { echo "Invalid city: $city" >&2; exit 2; }
  done
}
run_student(){
  local city="$1" seed="$2" variant="$3" order="$4"
  local root="results/beliefmove-evo/artifacts/full/$city/$variant/$order/seed-$seed"
  if [[ ! -f "$root/best.pt" ]]; then
    CITY="$city" SEED="$seed" VARIANT="$variant" ORDER_MODE="$order" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" \
      EPOCHS="${EPOCHS:-10}" ./scripts/beliefmove_evo.sh train-student
  else echo "skip checkpoint $root/best.pt"; fi
}
evaluate_student(){
  local city="$1" seed="$2" variant="$3" order="$4" rq="$5"
  local root="results/beliefmove-evo/artifacts/full/$city/$variant/$order/seed-$seed"
  if [[ ! -f "$root/test.metrics.json" || ! -f "$root/test.predictions.npz" ]]; then
    CITY="$city" SEED="$seed" VARIANT="$variant" ORDER_MODE="$order" EVALUATION_RQ="$rq" \
      DEVICE="$DEVICE" BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" ./scripts/beliefmove_evo.sh evaluate-student
  else echo "skip evaluation $root"; fi
}
evaluate_rq6(){
  local city="$1" seed="$2" variant="$3"
  local root="results/beliefmove-evo/artifacts/full/$city/$variant/correct/seed-$seed"
  if [[ ! -f "$root/rq6.metrics.json" || ! -f "$root/rq6.predictions.npz" ]]; then
    CITY="$city" SEED="$seed" VARIANT="$variant" DEVICE="$DEVICE" BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" \
      ./scripts/beliefmove_evo.sh evaluate-rq6
  else echo "skip RQ6 evaluation $root"; fi
}

audit(){
  require_python; validate_cities; ./scripts/tist2015_pipeline.sh audit || true
  local city missing=0 base file
  for city in $CITY_LIST; do
    base="data/hybrid/TIST2015/$city"
    for file in candidate_ids.json getnext/train.csv getnext/val.csv getnext/test.csv neural_cgm/validation.jsonl neural_cgm/test.jsonl; do
      [[ -f "$base/$file" ]] && echo "ready   $base/$file" || { echo "missing $base/$file"; missing=$((missing+1)); }
    done
  done
  echo "audit cities=$(wc -w <<<"$CITY_LIST" | tr -d ' ') missing=$missing"
  ((missing==0))
}

neural(){
  require_python; validate_cities
  CITIES="$CITY_LIST" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" ./scripts/tist2015_pipeline.sh train
  local city seed variant order
  for city in $CITY_LIST; do
    log "neural city=$city RQ10"
    CITY="$city" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" RQ10_SEEDS="$SEEDS" \
      ./scripts/rq10_teacher_robustness.sh run-seeds
    log "neural city=$city RQ2"
    CITY="$city" DEVICE="$DEVICE" BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" RQ2_SEEDS="$SEEDS" \
      ./scripts/rq2_data_only.sh run-seeds
    log "neural city=$city RQ4"
    for seed in $SEEDS; do
      for variant in E0-ce E1-kd E2-kd-traj E3-kd-vel E4-layer E5-dual; do
        run_student "$city" "$seed" "$variant" correct
        evaluate_student "$city" "$seed" "$variant" correct RQ4
      done
    done
    log "neural city=$city RQ5"
    for seed in $SEEDS; do
      evaluate_student "$city" "$seed" E5-dual correct RQ5
      for order in reverse random; do
        run_student "$city" "$seed" E5-dual "$order"
        evaluate_student "$city" "$seed" E5-dual "$order" RQ5
      done
    done
    log "neural city=$city RQ6"
    for seed in $SEEDS; do
      run_student "$city" "$seed" E6-temporal correct
      for variant in E1-kd E2-kd-traj E3-kd-vel E4-layer E6-temporal E5-dual; do
        evaluate_rq6 "$city" "$seed" "$variant"
      done
    done
    log "neural city=$city RQ13"
    CITY="$city" DEVICE="$DEVICE" BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" RQ13_SEEDS="$SEEDS" \
      ./scripts/rq13_robustness.sh run-seeds
  done
}

bayesian(){
  require_python; validate_cities
  local city seed root
  for city in $CITY_LIST; do
    log "bayesian city=$city RQ7"
    for seed in $SEEDS; do
      root="results/beliefmove-evo/artifacts/full/$city/E5-dual/correct/seed-$seed/rq7"
      if [[ ! -f "$root/rq7.metrics.json" ]]; then
        CITY="$city" SEED="$seed" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" ./scripts/beliefmove_evo.sh evaluate-rq7
      else echo "skip RQ7 $root/rq7.metrics.json"; fi
    done
    log "bayesian city=$city RQ11"
    CITY="$city" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" RQ11_SEEDS="$SEEDS" \
      ./scripts/rq11_calibration.sh run-seeds
  done
}

efficiency(){
  require_python; validate_cities
  local city
  for city in $CITY_LIST; do
    log "efficiency city=$city RQ12; requires uncontended GPU"
    CITY="$city" DEVICE="$DEVICE" RQ12_SEEDS="$SEEDS" BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-5}" \
      ./scripts/rq12_efficiency.sh run-profiles
  done
}

llm_bounded(){
  require_python; validate_cities
  log "build/resume bounded Hybrid evidence caches"
  CITIES="$CITY_LIST" TEST_LIMIT="$LIMIT" VALIDATION_LIMIT="$LIMIT" QUERY_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" \
    ./scripts/tist2015_pipeline.sh run
  RQ1_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/rq1_reproducibility.sh run-bounded
  local city
  for city in $CITY_LIST; do
    log "LLM bounded city=$city RQ3"
    CITY="$city" RQ3_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/rq3_llm_distillation.sh evaluate
    log "LLM bounded city=$city RQ8"
    CITY="$city" RQ8_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/rq8_routing.sh collect
    CITY="$city" RQ8_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" RQ8_RANDOM_SEEDS="${RQ8_RANDOM_SEEDS:-$(seq 42 91)}" \
      ./scripts/rq8_routing.sh evaluate-random
    log "LLM bounded city=$city RQ9"
    CITY="$city" RQ9_LIMIT="$LIMIT" OLLAMA_MODEL="$MODEL" ./scripts/rq9_semantic.sh collect
  done
}

summary(){
  local allow=(); [[ "$1" == status ]] && allow+=(--allow-incomplete)
  "$PY" -m hybrid.all_cities_summary --results-root results/beliefmove-evo/artifacts/full \
    --cities $CANONICAL_CITIES --seeds $SEEDS --model-slug "$MODEL_SLUG" --limit "$LIMIT" \
    --random-seeds ${RQ8_RANDOM_SEEDS:-$(seq 42 91)} \
    --scope "${AGGREGATE_SCOPE:-all}" --output results/beliefmove-evo/aggregated/12city/summary.json \
    --markdown ../../ideas/results_12city.md "${allow[@]}"
}

all(){ neural; bayesian; efficiency; llm_bounded; summary aggregate; }
case "$ACTION" in
  audit) audit;; neural) neural;; bayesian) bayesian;; efficiency) efficiency;;
  llm-bounded) llm_bounded;; status) summary status;; aggregate) summary aggregate;; all) all;;
  *) echo "Usage: $0 <audit|neural|bayesian|efficiency|llm-bounded|status|aggregate|all>" >&2; exit 2;;
esac
