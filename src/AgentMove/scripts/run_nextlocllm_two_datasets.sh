#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-all}" in
  shanghai) exec "$DIR/run_nextlocllm_shanghai_200.sh" ;;
  tist2015) exec "$DIR/run_nextlocllm_tist2015_200.sh" pending ;;
  audit) exec "$DIR/run_nextlocllm_tist2015_200.sh" audit ;;
  aggregate) exec "$DIR/run_nextlocllm_tist2015_200.sh" aggregate ;;
  all) "$DIR/run_nextlocllm_shanghai_200.sh"; "$DIR/run_nextlocllm_tist2015_200.sh" pending; "$DIR/run_nextlocllm_tist2015_200.sh" aggregate ;;
  *) echo "Usage: $0 {shanghai|tist2015|audit|aggregate|all}" >&2; exit 2 ;;
esac
