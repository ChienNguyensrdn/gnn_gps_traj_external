#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"; PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"
SEED="${SEED:-42}"; TEACHER="${TEACHER:-gru}"; BASE="data/hybrid/TIST2015/$CITY"
ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq10"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
audit() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 2; }
  require_file "$BASE/getnext/train.csv"; require_file "$BASE/getnext/val.csv"
  require_file "$BASE/getnext/test.csv"; require_file "$BASE/candidate_ids.json"
  echo "city=$CITY seed=$SEED teacher=$TEACHER"; echo "output=$ROOT"
  echo "matched_protocol=same splits,candidates,seeds,student architecture"
}
teacher_checkpoint() { echo "$ROOT/teachers/$1/seed-$SEED/best.pt"; }
student_checkpoint() { echo "$ROOT/students/$1/seed-$SEED/best.pt"; }
train_teacher() {
  audit
  local output; output="$(teacher_checkpoint "$TEACHER")"
  [[ -f "$output" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $output"; return; }
  case "$TEACHER" in
    gru) module=hybrid.neural_cgm; command=(train) ;;
    transformer) module=hybrid.transformer_teacher; command=() ;;
    pmt|unitraj) echo "$TEACHER adapter is not protocol-verified; excluded from RQ10." >&2; exit 2 ;;
    *) echo "TEACHER must be gru, transformer, pmt or unitraj" >&2; exit 2 ;;
  esac
  "$PYTHON_BIN" -m "$module" "${command[@]}" --train-csv "$BASE/getnext/train.csv" \
    --validation-csv "$BASE/getnext/val.csv" --candidate-ids "$BASE/candidate_ids.json" \
    --output "$output" --epochs "${TEACHER_EPOCHS:-10}" --batch-size "${BATCH_SIZE:-128}" \
    --learning-rate "${LEARNING_RATE:-0.001}" --seed "$SEED" --device "${DEVICE:-auto}"
}
train_student() {
  audit
  local source="$TEACHER" weights=(1 1 1 1)
  if [[ "$TEACHER" == none ]]; then source=gru; weights=(0 0 0 0); fi
  [[ "$source" == gru || "$source" == transformer ]] || { echo "TEACHER must be none, gru or transformer" >&2; exit 2; }
  local teacher output; teacher="$(teacher_checkpoint "$source")"; output="$(student_checkpoint "$TEACHER")"
  require_file "$teacher"
  [[ -f "$output" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $output"; return; }
  "$PYTHON_BIN" -m hybrid.dual_evolution --teacher-checkpoint "$teacher" \
    --train-csv "$BASE/getnext/train.csv" --validation-csv "$BASE/getnext/val.csv" --output "$output" \
    --seed "$SEED" --epochs "${STUDENT_EPOCHS:-10}" --batch-size "${BATCH_SIZE:-128}" \
    --device "${DEVICE:-auto}" --learning-rate "${LEARNING_RATE:-0.001}" \
    --lambda-kd "${weights[0]}" --lambda-trajectory "${weights[1]}" \
    --lambda-velocity "${weights[2]}" --lambda-temporal "${weights[3]}"
}
evaluate_one() {
  local group="$1" name="$2" checkpoint="$3" output
  output="$ROOT/$group/$name/seed-$SEED"
  require_file "$checkpoint"
  if [[ -f "$output/test.metrics.json" && -f "$output/test.predictions.npz" && "${FORCE:-0}" != 1 ]]; then
    echo "skip existing evaluation $output"
    return
  fi
  "$PYTHON_BIN" -m hybrid.evaluate_student --checkpoint "$checkpoint" --test-csv "$BASE/getnext/test.csv" \
    --output "$output/test.metrics.json" --predictions-output "$output/test.predictions.npz" \
    --batch-size "${BATCH_SIZE:-256}" --device "${DEVICE:-auto}" --seed "$SEED"
}
evaluate() {
  audit
  case "$TEACHER" in
    none) evaluate_one students none "$(student_checkpoint none)" ;;
    gru|transformer)
      evaluate_one teachers "$TEACHER" "$(teacher_checkpoint "$TEACHER")"
      evaluate_one students "$TEACHER" "$(student_checkpoint "$TEACHER")" ;;
    *) echo "TEACHER must be none, gru or transformer" >&2; exit 2 ;;
  esac
}
status() {
  local seed architecture path missing=0
  echo "RQ10 status city=$CITY root=$ROOT seeds=${RQ10_SEEDS:-42 43 44}"
  for seed in ${RQ10_SEEDS:-42 43 44}; do
    for architecture in gru transformer; do
      for path in \
        "$ROOT/teachers/$architecture/seed-$seed/best.pt" \
        "$ROOT/teachers/$architecture/seed-$seed/test.metrics.json" \
        "$ROOT/teachers/$architecture/seed-$seed/test.predictions.npz"; do
        if [[ -f "$path" ]]; then echo "ready   $path"; else echo "missing $path"; missing=$((missing + 1)); fi
      done
    done
    for architecture in none gru transformer; do
      for path in \
        "$ROOT/students/$architecture/seed-$seed/best.pt" \
        "$ROOT/students/$architecture/seed-$seed/test.metrics.json" \
        "$ROOT/students/$architecture/seed-$seed/test.predictions.npz"; do
        if [[ -f "$path" ]]; then echo "ready   $path"; else echo "missing $path"; missing=$((missing + 1)); fi
      done
    done
  done
  if (( missing > 0 )); then
    echo "RQ10 incomplete: $missing artifact(s) missing." >&2
    echo "Resume safely with: CITY=$CITY DEVICE=${DEVICE:-cuda} BATCH_SIZE=${BATCH_SIZE:-128} $0 run-seeds" >&2
    return 2
  fi
  echo "RQ10 complete: all required checkpoints, metrics and paired predictions are ready."
}
run_seeds() {
  local seed architecture
  for seed in ${RQ10_SEEDS:-42 43 44}; do
    for architecture in gru transformer; do CITY="$CITY" SEED="$seed" TEACHER="$architecture" "$0" train-teacher; done
    for architecture in none gru transformer; do
      CITY="$CITY" SEED="$seed" TEACHER="$architecture" "$0" train-student
      CITY="$CITY" SEED="$seed" TEACHER="$architecture" "$0" evaluate
    done
  done
}
aggregate() {
  status || { echo "Aggregation stopped; incomplete runs must not be reported as complete RQ10." >&2; exit 2; }
  "$PYTHON_BIN" -m hybrid.rq10_aggregate --root "$ROOT" --seeds ${RQ10_SEEDS:-42 43 44} \
    --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" \
    --output results/beliefmove-evo/aggregated/rq10_summary.json --markdown ../../ideas/results_rq10.md
}
case "$ACTION" in
  audit) audit ;; train-teacher) train_teacher ;; train-student) train_student ;; evaluate) evaluate ;;
  status) status ;; run-seeds) run_seeds ;; aggregate) aggregate ;;
  *) echo "Usage: $0 <audit|status|train-teacher|train-student|evaluate|run-seeds|aggregate>" >&2; exit 2 ;;
esac
