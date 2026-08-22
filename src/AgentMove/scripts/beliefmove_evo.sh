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
  local rq="RQ4" rq_dir="rq4"
  if [[ "$order" != "correct" ]]; then rq="RQ5"; rq_dir="rq5"; fi
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

case "$ACTION" in
  audit) audit ;; environment) environment ;; prepare) ./scripts/tist2015_pipeline.sh prepare ;;
  train-teacher) ./scripts/tist2015_pipeline.sh train ;; train-student) train_student ;;
  order-ablation) order_ablation ;; aggregate) aggregate_results ;;
  test) require_python; "$PYTHON_BIN" -m unittest discover -s tests -v ;;
  *) echo "Usage: $0 <audit|environment|prepare|train-teacher|train-student|order-ablation|aggregate|test>" >&2; exit 2 ;;
esac
