#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh
ACTION="${1:-audit}"
RQ3_SEED="${RQ3_SEED:-42}"

for dataset in $PHASE2_DATASETS; do for unit in $(phase2_units "$dataset"); do
  phase2_configure "$dataset" "$unit"; phase2_audit_base
  export RQ3_LIMIT="$LLM_LIMIT" OLLAMA_MODEL SEED="$RQ3_SEED"
  export RQ3_OUTPUT_ROOT="$P2_RESULTS/artifacts/full/$P2_CITY/rq3/$MODEL_SLUG/limit-$LLM_LIMIT/seed-$RQ3_SEED"
  case "$ACTION" in
    audit) ./scripts/rq3_llm_distillation.sh audit ;;
    collect) phase2_collect_evidence ;;
    run) ./scripts/rq3_llm_distillation.sh evaluate ;;
    status) ./scripts/rq3_llm_distillation.sh status ;;
    *) echo "Usage: $0 <audit|collect|run|status>" >&2; exit 2 ;;
  esac
done; done
echo "RQ3 phase2 action=$ACTION complete"
