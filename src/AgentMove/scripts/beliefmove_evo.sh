#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ACTION="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CITY="${CITY:-Tokyo}"
SEED="${SEED:-42}"
BASE="data/hybrid/TIST2015/$CITY"
OUT="results/beliefmove-evo"

require_python() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing $PYTHON_BIN; run ./scripts/setup_ubuntu.sh" >&2; exit 2; }
}

audit() {
  ./scripts/tist2015_pipeline.sh audit
  for config in configs/beliefmove_evo/base.json configs/beliefmove_evo/evolution_ablation.json configs/beliefmove_evo/routing.json; do
    [[ -f "$config" ]] && echo "config=ready $config" || { echo "config=missing $config"; return 2; }
  done
}

environment() {
  require_python; mkdir -p "$OUT"
  "$PYTHON_BIN" - "$OUT/environment_report.json" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
import numpy, pandas, torch
path=Path(sys.argv[1]); payload={
 "python":sys.version, "platform":platform.platform(), "numpy":numpy.__version__,
 "pandas":pandas.__version__, "torch":torch.__version__, "cuda":torch.cuda.is_available(),
 "mps":bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()),
 "git_commit":subprocess.run(["git","rev-parse","HEAD"],text=True,capture_output=True).stdout.strip()
}
path.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))
PY
}

train_student() {
  require_python
  local teacher="$BASE/neural_cgm/best.pt" variant="${VARIANT:-E5-dual}" order="${ORDER_MODE:-correct}"
  [[ -f "$teacher" ]] || { echo "Missing teacher: $teacher; run ./scripts/tist2015_pipeline.sh train" >&2; exit 2; }
  case "$variant" in
    E0-ce) weights=(0 0 0 0) ;; E1-kd) weights=(1 0 0 0) ;; E2-kd-traj) weights=(1 1 0 0) ;;
    E3-kd-vel) weights=(1 0 1 0) ;; E4-layer) weights=(1 1 1 0) ;; E5-dual) weights=(1 1 1 1) ;;
    E6-temporal) weights=(1 0 0 1) ;;
    *) echo "Unknown VARIANT=$variant" >&2; exit 2 ;;
  esac
  local limited=0 run_scope="full"
  local extra_args=()
  if [[ -n "${TRAIN_LIMIT:-}" ]]; then extra_args+=(--train-limit "$TRAIN_LIMIT"); limited=1; fi
  if [[ -n "${VALIDATION_LIMIT:-}" ]]; then extra_args+=(--validation-limit "$VALIDATION_LIMIT"); limited=1; fi
  if [[ "$limited" == "1" ]]; then run_scope="${RUN_TAG:-smoke}"; fi
  local target="$OUT/artifacts/$run_scope/$CITY/$variant/$order/seed-$SEED/best.pt"
  "$PYTHON_BIN" -m hybrid.dual_evolution --teacher-checkpoint "$teacher" \
    --train-csv "$BASE/getnext/train.csv" --validation-csv "$BASE/getnext/val.csv" --output "$target" \
    --seed "$SEED" --order-mode "$order" --epochs "${EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-64}" --device "${DEVICE:-auto}" \
    --learning-rate "${LEARNING_RATE:-0.001}" --lambda-kd "${weights[0]}" \
    --lambda-trajectory "${weights[1]}" --lambda-velocity "${weights[2]}" --lambda-temporal "${weights[3]}" \
    "${extra_args[@]}"
  if [[ "$limited" == "1" ]]; then
    echo "smoke_result=$target (not added to publication raw results)"
    return 0
  fi
  local rq="${TRAIN_RQ:-}" rq_dir
  if [[ -z "$rq" ]]; then
    if [[ "$order" != "correct" ]]; then rq="RQ5"; elif [[ "$variant" == "E6-temporal" ]]; then rq="RQ6"; else rq="RQ4"; fi
  fi
  case "$rq" in RQ4) rq_dir="rq4" ;; RQ5) rq_dir="rq5" ;; RQ6) rq_dir="rq6" ;;
    *) echo "TRAIN_RQ must be RQ4, RQ5 or RQ6" >&2; exit 2 ;; esac
  "$PYTHON_BIN" -m hybrid.record_beliefmove_result --metrics "${target%.pt}.metrics.json" \
    --output "$OUT/raw/$rq_dir/$CITY/$variant-$order-seed-$SEED.json" \
    --rq "$rq" --experiment "$variant-$order" --seed "$SEED" --dataset "TIST2015-$CITY" \
    --config configs/beliefmove_evo/evolution_ablation.json --repository ../.. \
    --dataset-files "$BASE/getnext/train.csv" "$BASE/getnext/val.csv" "$BASE/getnext/test.csv"
  echo "student=$target"
}

order_ablation() {
  local mode permutation_seed
  for mode in correct reverse; do ORDER_MODE="$mode" VARIANT="${VARIANT:-E5-dual}" "$0" train-student; done
  for permutation_seed in ${RANDOM_SEEDS:-42 43 44 45 46 47 48 49 50 51}; do
    ORDER_MODE=random SEED="$permutation_seed" VARIANT="${VARIANT:-E5-dual}" "$0" train-student
  done
}

aggregate_results() {
  require_python
  "$PYTHON_BIN" -m hybrid.beliefmove_results --input "$OUT/raw" --output-dir "$OUT/aggregated" --results-md ../../ideas/results.md
}

evaluate_student() {
  require_python
  local variant="${VARIANT:-E5-dual}" order="${ORDER_MODE:-correct}"
  local rq="${EVALUATION_RQ:-}" rq_dir
  if [[ -z "$rq" ]]; then
    [[ "$order" == "correct" ]] && rq="RQ4" || rq="RQ5"
  fi
  [[ "$rq" == "RQ4" || "$rq" == "RQ5" ]] || { echo "EVALUATION_RQ must be RQ4 or RQ5" >&2; exit 2; }
  [[ "$rq" == "RQ4" ]] && rq_dir="rq4-test" || rq_dir="rq5-test"
  local checkpoint="$OUT/artifacts/full/$CITY/$variant/$order/seed-$SEED/best.pt"
  local metrics="$OUT/artifacts/full/$CITY/$variant/$order/seed-$SEED/test.metrics.json"
  local predictions="$OUT/artifacts/full/$CITY/$variant/$order/seed-$SEED/test.predictions.npz"
  [[ -f "$checkpoint" ]] || { echo "Missing student checkpoint: $checkpoint" >&2; exit 2; }
  "$PYTHON_BIN" -m hybrid.evaluate_student --checkpoint "$checkpoint" --test-csv "$BASE/getnext/test.csv" \
    --output "$metrics" --batch-size "${BATCH_SIZE:-256}" --device "${DEVICE:-auto}" --seed "$SEED" \
    --order-mode "$order" --predictions-output "$predictions"
  "$PYTHON_BIN" -m hybrid.record_beliefmove_result --metrics "$metrics" \
    --output "$OUT/raw/$rq_dir/$CITY/$variant-$order-seed-$SEED.json" \
    --rq "$rq" --experiment "$variant-$order" --seed "$SEED" --dataset "TIST2015-$CITY" \
    --config configs/beliefmove_evo/evolution_ablation.json --repository ../.. --evaluation-split test \
    --dataset-files "$BASE/getnext/train.csv" "$BASE/getnext/val.csv" "$BASE/getnext/test.csv"
  echo "evaluation=$rq order=$order metrics=$metrics"
}

rq5_significance() {
  require_python
  local variant="${VARIANT:-E5-dual}"
  "$PYTHON_BIN" -m hybrid.paired_order_test \
    --artifacts-root "$OUT/artifacts/full/$CITY/$variant" \
    --seeds ${PAIRED_SEEDS:-42 43 44} --comparisons reverse random \
    --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" --seed "${SIGNIFICANCE_SEED:-42}" \
    --output "$OUT/aggregated/rq5_paired_significance.json" \
    --markdown ../../ideas/result_rq5_significance.md
}

evaluate_rq6() {
  require_python
  local variant="${VARIANT:-E5-dual}" order="${ORDER_MODE:-correct}"
  [[ "$order" == "correct" ]] || { echo "RQ6 requires ORDER_MODE=correct" >&2; exit 2; }
  local checkpoint="$OUT/artifacts/full/$CITY/$variant/correct/seed-$SEED/best.pt"
  local teacher="$BASE/neural_cgm/best.pt"
  local metrics="$OUT/artifacts/full/$CITY/$variant/correct/seed-$SEED/rq6.metrics.json"
  local predictions="$OUT/artifacts/full/$CITY/$variant/correct/seed-$SEED/rq6.predictions.npz"
  [[ -f "$checkpoint" ]] || { echo "Missing student checkpoint: $checkpoint" >&2; exit 2; }
  [[ -f "$teacher" ]] || { echo "Missing teacher checkpoint: $teacher" >&2; exit 2; }
  "$PYTHON_BIN" -m hybrid.rq6_evaluate --checkpoint "$checkpoint" --teacher-checkpoint "$teacher" \
    --validation-csv "$BASE/getnext/val.csv" --test-csv "$BASE/getnext/test.csv" \
    --output "$metrics" --predictions-output "$predictions" \
    --batch-size "${BATCH_SIZE:-256}" --device "${DEVICE:-auto}" --seed "$SEED"
  "$PYTHON_BIN" -m hybrid.record_beliefmove_result --metrics "$metrics" \
    --output "$OUT/raw/rq6-test/$CITY/$variant-correct-seed-$SEED.json" \
    --rq RQ6 --experiment "$variant-correct" --seed "$SEED" --dataset "TIST2015-$CITY" \
    --config configs/beliefmove_evo/evolution_ablation.json --repository ../.. --evaluation-split test \
    --dataset-files "$BASE/getnext/train.csv" "$BASE/getnext/val.csv" "$BASE/getnext/test.csv"
  echo "evaluation=RQ6 variant=$variant metrics=$metrics"
}

aggregate_rq6() {
  require_python
  "$PYTHON_BIN" -m hybrid.rq6_aggregate \
    --artifacts-root "$OUT/artifacts/full/$CITY" --seeds ${RQ6_SEEDS:-42 43 44} \
    --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" \
    --output "$OUT/aggregated/rq6_summary.json" --markdown ../../ideas/result_rq6.md
}

case "$ACTION" in
  audit) audit ;; environment) environment ;; prepare) ./scripts/tist2015_pipeline.sh prepare ;;
  train-teacher) ./scripts/tist2015_pipeline.sh train ;; train-student) train_student ;;
  evaluate-student) evaluate_student ;;
  order-ablation) order_ablation ;; rq5-significance) rq5_significance ;;
  evaluate-rq6) evaluate_rq6 ;; aggregate-rq6) aggregate_rq6 ;; aggregate) aggregate_results ;;
  test) require_python; "$PYTHON_BIN" -m unittest discover -s tests -v ;;
  *) echo "Usage: $0 <audit|environment|prepare|train-teacher|train-student|evaluate-student|order-ablation|rq5-significance|evaluate-rq6|aggregate-rq6|aggregate|test>" >&2; exit 2 ;;
esac
