#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh
ACTION="${1:-audit}"
VARIANTS="${RQ4_VARIANTS:-E0-ce E1-kd E5-dual}"
[[ "$ACTION" == audit || "$ACTION" == run ]] || { echo "Usage: $0 <audit|run>" >&2; exit 2; }
READY=0
REQUIRED=0

run_one() {
  local seed variant root
  for seed in $RQ_SEEDS; do for variant in $VARIANTS; do
    root="$P2_RESULTS/artifacts/full/$P2_CITY/$variant/correct/seed-$seed"
    if [[ "$ACTION" == audit ]]; then
      REQUIRED=$((REQUIRED + 2))
      [[ -f "$root/test.metrics.json" ]] && READY=$((READY + 1))
      [[ -f "$root/test.predictions.npz" ]] && READY=$((READY + 1))
      continue
    fi
    if [[ ! -f "$root/best.pt" || "${FORCE:-0}" == 1 ]]; then
      SEED="$seed" VARIANT="$variant" ORDER_MODE=correct EPOCHS="${STUDENT_EPOCHS:-10}" \
        BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh train-student
    fi
    if [[ ! -f "$root/test.metrics.json" || ! -f "$root/test.predictions.npz" || "${FORCE:-0}" == 1 ]]; then
      SEED="$seed" VARIANT="$variant" ORDER_MODE=correct EVALUATION_RQ=RQ4 \
        BATCH_SIZE="$EVAL_BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh evaluate-student
    fi
  done; done
}

for dataset in $PHASE2_DATASETS; do
  for unit in $(phase2_units "$dataset"); do
    phase2_configure "$dataset" "$unit"; phase2_audit_base
    run_one
done
done

if [[ "$ACTION" == audit ]]; then
  echo "RQ4 phase2 artifacts=$READY/$REQUIRED missing=$((REQUIRED - READY))"
  (( READY == REQUIRED )) || exit 1
fi
echo "RQ4 phase2 action=$ACTION complete"
