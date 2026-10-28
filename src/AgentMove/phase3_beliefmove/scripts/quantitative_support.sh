#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase3_beliefmove/scripts/common.sh

ACTION="${1:-audit}"
# Before protocol lock, quantitative support may only be created for splits
# used to fit/select the offline teacher.  Test support is an explicit later
# action after lock, never an accidental default.
QUANTITATIVE_SPLITS="${QUANTITATIVE_SPLITS:-train validation}"
QUANTITATIVE_TOP_K="${QUANTITATIVE_TOP_K:-20}"
QUANTITATIVE_BATCH_SIZE="${QUANTITATIVE_BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"

quantitative_unit() {
  local checkpoint="$BM_DATA_BASE/neural_cgm/best.pt"
  beliefmove_audit_base
  case "$ACTION" in
    audit)
      "$PYTHON_BIN" -m hybrid.beliefmove.quantitative_support audit \
        --data-base "$BM_DATA_BASE" \
        --prepared-root "$BM_OUTPUT" \
        --checkpoint "$checkpoint" \
        --results-root "$BELIEFMOVE_ROOT" \
        --protocol phase3_beliefmove/protocol.json \
        --splits $QUANTITATIVE_SPLITS
      ;;
    export)
      "$PYTHON_BIN" -m hybrid.beliefmove.quantitative_support export \
        --data-base "$BM_DATA_BASE" \
        --prepared-root "$BM_OUTPUT" \
        --checkpoint "$checkpoint" \
        --results-root "$BELIEFMOVE_ROOT" \
        --protocol phase3_beliefmove/protocol.json \
        --splits $QUANTITATIVE_SPLITS \
        --top-k "$QUANTITATIVE_TOP_K" \
        --batch-size "$QUANTITATIVE_BATCH_SIZE" \
        --device "$DEVICE"
      ;;
    bind)
      "$PYTHON_BIN" -m hybrid.beliefmove.quantitative_support bind \
        --data-base "$BM_DATA_BASE" \
        --prepared-root "$BM_OUTPUT" \
        --checkpoint "$checkpoint" \
        --splits $QUANTITATIVE_SPLITS
      ;;
    *)
      echo "Usage: $0 <audit|export|bind>" >&2
      return 2
      ;;
  esac
}

beliefmove_for_each_unit quantitative_unit
