#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh
ACTION="${1:-audit}"

for dataset in $PHASE2_DATASETS; do for unit in $(phase2_units "$dataset"); do
  phase2_configure "$dataset" "$unit"; phase2_audit_base
  export RQ8_LIMIT="$LLM_LIMIT" OLLAMA_MODEL
  export RQ8_OUTPUT_ROOT="$P2_RESULTS/artifacts/full/$P2_CITY/rq8/$MODEL_SLUG/limit-$LLM_LIMIT"
  case "$ACTION" in
    audit) SEED=42 ./scripts/rq8_routing.sh audit ;;
    collect) SEED=42 ./scripts/rq8_routing.sh collect ;;
    evaluate) SEED=42 ./scripts/rq8_routing.sh evaluate-random ;;
    *) echo "Usage: $0 <audit|collect|evaluate>" >&2; exit 2 ;;
  esac
done; done
echo "RQ8 phase2 action=$ACTION complete"
