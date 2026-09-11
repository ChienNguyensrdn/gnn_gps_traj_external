#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-audit}"
PY="${PYTHON_BIN:-.venv/bin/python}"
SEEDS="${RQ_SEEDS:-42 43 44}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
ITERATIONS="${SIGNIFICANCE_ITERATIONS:-10000}"
TIST_CITIES="${CITIES:-Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow}"
DATASETS="${CORE_DATASETS:-tist2015 www2019 yjmob100k}"

log(){ printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
need_python(){ [[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }; }

configure(){
  local dataset="$1" unit="$2"
  case "$dataset" in
    tist2015)
      CORE_CITY="$unit"; CORE_BASE="data/hybrid/TIST2015/$unit"
      CORE_OUT="results/beliefmove-evo"; CORE_LABEL="TIST2015-$unit" ;;
    www2019)
      CORE_CITY=Shanghai; CORE_BASE="${WWW2019_DATA_BASE:-data/hybrid/WWW2019/Shanghai}"
      CORE_OUT="${WWW2019_RESULTS_ROOT:-results/beliefmove-evo-www2019}"
      CORE_LABEL=WWW2019-Shanghai-ISP ;;
    yjmob100k)
      CORE_CITY=YJMob; CORE_BASE="${YJMOB_DATA_BASE:-data/hybrid/YJMob100K/YJMob}"
      CORE_OUT="${YJMOB_RESULTS_ROOT:-results/beliefmove-evo-yjmob100k}"
      CORE_LABEL="YJMob100K-Dataset${YJMOB_DATASET:-1}" ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  export CITY="$CORE_CITY" DATA_BASE="$CORE_BASE" BELIEFMOVE_OUT="$CORE_OUT" DATASET_LABEL="$CORE_LABEL"
}

units(){
  case "$1" in tist2015) echo "$TIST_CITIES";; www2019) echo Shanghai;; yjmob100k) echo YJMob;; esac
}

delegate(){
  local dataset="$1" action="$2"
  case "$dataset" in
    tist2015)
      [[ "$action" == train-teacher ]] && action=train
      CITIES="$TIST_CITIES" ./scripts/tist2015_pipeline.sh "$action" ;;
    www2019) ./scripts/www2019_pipeline.sh "$action" ;;
    yjmob100k) ./scripts/yjmob100k_pipeline.sh "$action" ;;
  esac
}

audit(){
  need_python
  local dataset
  for dataset in $DATASETS; do log "audit dataset=$dataset"; delegate "$dataset" audit || true; done
}

prepare(){
  local dataset
  for dataset in $DATASETS; do
    log "prepare dataset=$dataset"; delegate "$dataset" prepare
    log "teacher dataset=$dataset"; delegate "$dataset" train-teacher
  done
}

variants(){
  if [[ "${FULL_ABLATION:-0}" == 1 ]]; then
    echo "E0-ce E1-kd E2-kd-traj E3-kd-vel E4-layer E6-temporal E5-dual"
  else
    echo "E0-ce E1-kd E5-dual"
  fi
}

train_eval(){
  local variant="$1" seed="$2" root
  root="$CORE_OUT/artifacts/full/$CORE_CITY/$variant/correct/seed-$seed"
  if [[ ! -f "$root/best.pt" ]]; then
    SEED="$seed" VARIANT="$variant" ORDER_MODE=correct EPOCHS="${STUDENT_EPOCHS:-10}" \
      BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh train-student
  else echo "skip checkpoint $root/best.pt"; fi
  if [[ ! -f "$root/test.metrics.json" || ! -f "$root/test.predictions.npz" ]]; then
    SEED="$seed" VARIANT="$variant" ORDER_MODE=correct EVALUATION_RQ=RQ4 \
      BATCH_SIZE="$EVAL_BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh evaluate-student
  else echo "skip evaluation $root"; fi
}

neural(){
  need_python
  local dataset unit seed variant
  for dataset in $DATASETS; do
    for unit in $(units "$dataset"); do
      configure "$dataset" "$unit"; log "neural dataset=$CORE_LABEL"
      [[ -f "$CORE_BASE/neural_cgm/best.pt" ]] || { echo "Missing teacher: $CORE_BASE/neural_cgm/best.pt" >&2; exit 2; }
      for seed in $SEEDS; do for variant in $(variants); do train_eval "$variant" "$seed"; done; done
    done
  done
}

temporal(){
  need_python
  local dataset unit seed mode checkpoint output
  for dataset in $DATASETS; do for unit in $(units "$dataset"); do
    configure "$dataset" "$unit"; log "frozen temporal control dataset=$CORE_LABEL"
    for seed in $SEEDS; do
      checkpoint="$CORE_OUT/artifacts/full/$CORE_CITY/E5-dual/correct/seed-$seed/best.pt"
      [[ -f "$checkpoint" ]] || { echo "Missing E5 checkpoint: $checkpoint" >&2; exit 2; }
      for mode in correct reverse random; do
        output="$CORE_OUT/artifacts/full/$CORE_CITY/E5-dual/frozen-$mode/seed-$seed"
        [[ -f "$output/test.metrics.json" && -f "$output/test.predictions.npz" && "${FORCE:-0}" != 1 ]] && { echo "skip $output"; continue; }
        mkdir -p "$output"
        "$PY" -m hybrid.evaluate_student --checkpoint "$checkpoint" --test-csv "$CORE_BASE/getnext/test.csv" \
          --output "$output/test.metrics.json" --predictions-output "$output/test.predictions.npz" \
          --batch-size "$EVAL_BATCH_SIZE" --device "$DEVICE" --seed "$seed" \
          --order-mode correct --input-order-mode "$mode"
      done
    done
  done; done
}

belief(){
  need_python
  local dataset unit seed checkpoint output
  for dataset in $DATASETS; do for unit in $(units "$dataset"); do
    configure "$dataset" "$unit"; log "belief dataset=$CORE_LABEL"
    for seed in $SEEDS; do
      checkpoint="$CORE_OUT/artifacts/full/$CORE_CITY/E5-dual/correct/seed-$seed/best.pt"
      output="$CORE_OUT/artifacts/full/$CORE_CITY/E5-dual/correct/seed-$seed/rq7"
      [[ -f "$output/rq7.metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip $output"; continue; }
      "$PY" -m hybrid.rq7_belief_memory --checkpoint "$checkpoint" \
        --train-csv "$CORE_BASE/getnext/train.csv" --validation-csv "$CORE_BASE/getnext/val.csv" \
        --test-csv "$CORE_BASE/getnext/test.csv" --output-dir "$output" \
        --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$seed"
    done
  done; done
}

teacher_robustness(){
  local dataset unit
  for dataset in $DATASETS; do for unit in $(units "$dataset"); do
    configure "$dataset" "$unit"; log "teacher robustness dataset=$CORE_LABEL"
    RQ10_SEEDS="$SEEDS" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" \
      ./scripts/rq10_teacher_robustness.sh run-seeds
  done; done
}

calibration(){
  local dataset unit
  for dataset in $DATASETS; do for unit in $(units "$dataset"); do
    configure "$dataset" "$unit"; log "calibration dataset=$CORE_LABEL"
    RQ11_SEEDS="$SEEDS" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" \
      ./scripts/rq11_calibration.sh run-seeds
  done; done
}

aggregate_one(){
  local dataset="$1" unit="$2" slug output report
  configure "$dataset" "$unit"; slug="${dataset}-${unit}"; output="results/publication-core/$slug.json"
  report="../../ideas/publication_core_${slug}.md"
  "$PY" -m hybrid.www2019_summary --root "$CORE_OUT/artifacts/full/$CORE_CITY" --seeds $SEEDS \
    --iterations "$ITERATIONS" --dataset-label "$CORE_LABEL" --report-title "$CORE_LABEL — Publication core" \
    --output "$output" --markdown "$report" --frozen-only "${@:3}"
}

aggregate(){
  need_python; local dataset unit
  for dataset in $DATASETS; do for unit in $(units "$dataset"); do aggregate_one "$dataset" "$unit"; done; done
}
status(){
  need_python; local dataset unit
  for dataset in $DATASETS; do for unit in $(units "$dataset"); do aggregate_one "$dataset" "$unit" --allow-incomplete || true; done; done
}
report(){
  ./scripts/aggregate_publication_core_markdown.sh
}

case "$ACTION" in
  audit) audit;; prepare) prepare;; neural) neural;; temporal) temporal;; belief) belief;;
  teacher-robustness) teacher_robustness;; calibration) calibration;; aggregate) aggregate;; status) status;; report) report;;
  core) prepare; neural; temporal; belief; aggregate;;
  *) echo "Usage: $0 <audit|prepare|neural|temporal|belief|teacher-robustness|calibration|status|aggregate|report|core>" >&2; exit 2;;
esac
