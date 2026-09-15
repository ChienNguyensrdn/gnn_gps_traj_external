#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh
ACTION="${1:-audit}"
[[ "$ACTION" == audit || "$ACTION" == run ]] || { echo "Usage: $0 <audit|run>" >&2; exit 2; }
READY=0
REQUIRED=0

run_one() {
  local seed checkpoint output
  for seed in $RQ_SEEDS; do
    checkpoint="$P2_RESULTS/artifacts/full/$P2_CITY/E5-dual/correct/seed-$seed/best.pt"
    output="$P2_RESULTS/artifacts/full/$P2_CITY/E5-dual/correct/seed-$seed/rq7"
    if [[ "$ACTION" == audit ]]; then
      REQUIRED=$((REQUIRED + 2))
      [[ -f "$checkpoint" ]] && READY=$((READY + 1))
      [[ -f "$output/rq7.metrics.json" ]] && READY=$((READY + 1))
      continue
    fi
    phase2_require "$checkpoint"
    [[ -f "$output/rq7.metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip $output"; continue; }
    .venv/bin/python -m hybrid.rq7_belief_memory --checkpoint "$checkpoint" \
      --train-csv "$P2_DATA_BASE/getnext/train.csv" --validation-csv "$P2_DATA_BASE/getnext/val.csv" \
      --test-csv "$P2_DATA_BASE/getnext/test.csv" --output-dir "$output" \
      --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$seed"
  done
}

for dataset in $PHASE2_DATASETS; do for unit in $(phase2_units "$dataset"); do
  phase2_configure "$dataset" "$unit"; phase2_audit_base; run_one
done; done
if [[ "$ACTION" == audit ]]; then
  echo "RQ7 phase2 dependencies-and-artifacts=$READY/$REQUIRED missing=$((REQUIRED - READY))"
  (( READY == REQUIRED )) || exit 1
fi
echo "RQ7 phase2 action=$ACTION complete"
