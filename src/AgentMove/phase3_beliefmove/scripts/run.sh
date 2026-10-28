#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase3_beliefmove/scripts/common.sh

ACTION="${1:-audit}"

run_cli() {
  "$PYTHON_BIN" -m hybrid.beliefmove.experiment "$@"
}

audit_unit() {
  beliefmove_audit_base
  run_cli audit \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT"
}

prepare_unit() {
  beliefmove_audit_base
  run_cli prepare \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --region-cell-meters "$REGION_CELL_METERS"
}

teacher_unit() {
  local tag="$1" train_limit="$2" val_limit="$3" query_root="${4:-}"
  local -a command=(teacher-cache
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --model "$OLLAMA_MODEL" --cache-tag "$tag" \
    --model-digest "$OLLAMA_MODEL_DIGEST" \
    --train-limit "$train_limit" --validation-limit "$val_limit" \
    --require-gate)
  if [[ -n "$query_root" ]]; then command+=(--query-root "$query_root"); fi
  run_cli "${command[@]}"
}

is_pilot_unit() {
  [[ "$BM_DATASET:$BM_UNIT" == "tist2015:Tokyo" || \
     "$BM_DATASET:$BM_UNIT" == "www2019:Shanghai" || \
     "$BM_DATASET:$BM_UNIT" == "yjmob100k:YJMob" ]]
}

smoke_unit() {
  is_pilot_unit || return 0
  teacher_unit smoke 20 20
}

pilot_unit() {
  is_pilot_unit || return 0
  teacher_unit pilot 500 200
  local repeat
  for repeat in $(seq 1 "$PILOT_REPEAT_COUNT"); do
    teacher_unit "pilot-repeat-$repeat" 0 "$PILOT_REPEAT_VAL_LIMIT"
  done
  run_cli pilot-evaluate \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --model "$OLLAMA_MODEL" --model-digest "$OLLAMA_MODEL_DIGEST" \
    --repeat-count "$PILOT_REPEAT_COUNT"
}

teacher_full_unit() { teacher_unit official "$TEACHER_TRAIN_LIMIT" "$TEACHER_VAL_LIMIT"; }

fit_unit() {
  run_cli fit \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --model "$OLLAMA_MODEL" --seeds $BELIEFMOVE_SEEDS
}

train_unit() {
  run_cli train \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --model "$OLLAMA_MODEL" --seeds $BELIEFMOVE_SEEDS
}

test_bank_unit() {
  run_cli test-bank \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --model "$OLLAMA_MODEL" --limit "$TEACHER_TEST_PANEL_LIMIT"
}

status_unit() {
  run_cli status \
    --dataset "$BM_DATASET" --unit "$BM_UNIT" \
    --data-base "$BM_DATA_BASE" --output-root "$BM_OUTPUT" \
    --model "$OLLAMA_MODEL"
}

case "$ACTION" in
  audit) beliefmove_for_each_unit audit_unit ;;
  prepare) beliefmove_for_each_unit prepare_unit ;;
  smoke)
    beliefmove_for_each_unit smoke_unit
    run_cli smoke-report --results-root "$BELIEFMOVE_ROOT" \
      --model "$OLLAMA_MODEL" --model-digest "$OLLAMA_MODEL_DIGEST" \
      --require-ready ;;
  smoke-report)
    run_cli smoke-report --results-root "$BELIEFMOVE_ROOT" \
      --model "$OLLAMA_MODEL" --model-digest "$OLLAMA_MODEL_DIGEST" ;;
  pilot)
    # This is intentionally the first pilot action.  No pilot request may be
    # sent unless all three representative units pass the strict smoke gate.
    run_cli smoke-report --results-root "$BELIEFMOVE_ROOT" \
      --model "$OLLAMA_MODEL" --model-digest "$OLLAMA_MODEL_DIGEST" \
      --require-ready
    beliefmove_for_each_unit pilot_unit ;;
  pilot-report) run_cli pilot-report --results-root "$BELIEFMOVE_ROOT" ;;
  lock) run_cli lock --results-root "$BELIEFMOVE_ROOT" --protocol phase3_beliefmove/protocol.json ;;
  quantitative-support-audit)
    ./phase3_beliefmove/scripts/quantitative_support.sh audit ;;
  quantitative-support)
    ./phase3_beliefmove/scripts/quantitative_support.sh export
    ./phase3_beliefmove/scripts/quantitative_support.sh bind ;;
  teacher-offline)
    echo "NOT_IMPLEMENTED: per-query quantitative support is available, but official teacher collection remains blocked until the confirmatory M1-M5 fit/evaluation contract is integrated." >&2
    exit 3 ;;
  quantitative-teacher) run_cli quantitative-teacher --results-root "$BELIEFMOVE_ROOT" --seeds $BELIEFMOVE_SEEDS ;;
  train) beliefmove_for_each_unit train_unit ;;
  test-bank) beliefmove_for_each_unit test_bank_unit ;;
  evaluate) run_cli evaluate --results-root "$BELIEFMOVE_ROOT" --seeds $BELIEFMOVE_SEEDS ;;
  controls) run_cli controls --results-root "$BELIEFMOVE_ROOT" --seeds $BELIEFMOVE_SEEDS ;;
  status) beliefmove_for_each_unit status_unit ;;
  aggregate)
    run_cli aggregate \
      --results-root "$BELIEFMOVE_ROOT" \
      --protocol phase3_beliefmove/protocol.json \
      --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" ;;
  report) run_cli report --results-root "$BELIEFMOVE_ROOT" ;;
  *)
    echo "Usage: $0 <audit|prepare|smoke|smoke-report|pilot|pilot-report|lock|quantitative-support-audit|quantitative-support|teacher-offline|quantitative-teacher|train|test-bank|evaluate|controls|status|aggregate|report>" >&2
    exit 2 ;;
esac
