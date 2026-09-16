#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh

RQ="${1:-all}"
[[ "$RQ" == rq4 || "$RQ" == rq7 || "$RQ" == all ]] || { echo "Usage: $0 <rq4|rq7|all>" >&2; exit 2; }
read -r -a seeds <<< "$RQ_SEEDS"
read -r -a cities <<< "$TIST_CITIES"
args=("$RQ" --root "$PHASE2_ROOT" --seeds "${seeds[@]}" --cities "${cities[@]}"
      --iterations "${SIGNIFICANCE_ITERATIONS:-10000}")
[[ "${ALLOW_INCOMPLETE:-0}" == 1 ]] && args+=(--allow-incomplete)
.venv/bin/python -m phase2.aggregate_reports "${args[@]}"
