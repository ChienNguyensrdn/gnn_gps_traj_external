#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh
read -r -a cities <<< "$TIST_CITIES"
.venv/bin/python -m phase2.llm_progress --root . --phase2-root "$PHASE2_ROOT" --model-slug "$MODEL_SLUG" \
  --limit "$LLM_LIMIT" --top-k "${TOP_K:-10}" --cities "${cities[@]}"
