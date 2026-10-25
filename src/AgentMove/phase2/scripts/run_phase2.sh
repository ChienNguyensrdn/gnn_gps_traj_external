#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ACTION="${1:-audit}"

case "$ACTION" in
  audit)
    failed=0
    for rq in rq4 rq7 rq3 rq8; do
      echo "=== audit $rq ==="
      ./phase2/scripts/$rq.sh audit || failed=$((failed + 1))
    done
    echo "Phase2 audit completed: incomplete_rqs=$failed/4"
    (( failed == 0 )) ;;
  neural)
    ./phase2/scripts/rq4.sh run
    ./phase2/scripts/rq7.sh run ;;
  llm)
    ./phase2/scripts/rq3.sh collect
    ./phase2/scripts/rq3.sh run
    ./phase2/scripts/rq8.sh collect
    ./phase2/scripts/rq8.sh evaluate ;;
  llm-status)
    ./phase2/scripts/llm_status.sh ;;
  llm-report)
    ./phase2/scripts/aggregate_llm.sh all ;;
  direct-llm-prepare)
    ./phase2/scripts/direct_query_llm.sh prepare ;;
  direct-llm)
    ./phase2/scripts/direct_query_llm.sh collect
    ./phase2/scripts/direct_query_llm.sh route ;;
  direct-llm-status)
    ./phase2/scripts/direct_query_llm.sh status ;;
  direct-llm-audit)
    ./phase2/scripts/direct_query_llm.sh audit ;;
  direct-llm-report)
    ./phase2/scripts/direct_query_llm.sh aggregate ;;
  all)
    "$0" neural
    "$0" llm ;;
  report)
    ./phase2/scripts/aggregate.sh all ;;
  *) echo "Usage: $0 <audit|neural|llm|llm-status|llm-report|direct-llm-prepare|direct-llm|direct-llm-status|direct-llm-audit|direct-llm-report|all|report>" >&2; exit 2 ;;
esac
