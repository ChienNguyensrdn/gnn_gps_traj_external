#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh
RQ="${1:-all}"
[[ "$RQ" == rq3 || "$RQ" == rq8 || "$RQ" == all ]] || { echo "Usage: $0 <rq3|rq8|all>" >&2; exit 2; }
read -r -a cities <<< "$TIST_CITIES"
seed_values="${RQ8_RANDOM_SEEDS:-$(seq -s ' ' 42 91)}"
read -r -a seeds <<< "$seed_values"
args=("$RQ" --root "$PHASE2_ROOT" --model-slug "$MODEL_SLUG" --limit "$LLM_LIMIT"
      --cities "${cities[@]}" --seeds "${seeds[@]}" --iterations "${SIGNIFICANCE_ITERATIONS:-10000}")
[[ "${ALLOW_INCOMPLETE:-0}" == 1 ]] && args+=(--allow-incomplete)
.venv/bin/python -m phase2.aggregate_llm_reports "${args[@]}"
