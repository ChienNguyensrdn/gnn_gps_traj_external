#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ACTION="${1:-audit}"

case "$ACTION" in
  audit)
    for rq in rq4 rq7 rq3 rq8; do ./phase2/scripts/$rq.sh audit; done ;;
  neural)
    ./phase2/scripts/rq4.sh run
    ./phase2/scripts/rq7.sh run ;;
  llm)
    ./phase2/scripts/rq3.sh collect
    ./phase2/scripts/rq3.sh run
    ./phase2/scripts/rq8.sh collect
    ./phase2/scripts/rq8.sh evaluate ;;
  all)
    "$0" neural
    "$0" llm ;;
  *) echo "Usage: $0 <audit|neural|llm|all>" >&2; exit 2 ;;
esac
