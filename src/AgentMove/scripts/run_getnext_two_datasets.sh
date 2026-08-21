#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-all}" in
  shanghai) exec "$DIR/run_getnext_shanghai_200.sh" ;;
  tist2015) exec "$DIR/run_getnext_tist2015_200.sh" pending ;;
  aggregate) exec "$DIR/run_getnext_tist2015_200.sh" aggregate ;;
  audit) exec "$DIR/run_getnext_tist2015_200.sh" audit ;;
  all) "$DIR/run_getnext_shanghai_200.sh"; "$DIR/run_getnext_tist2015_200.sh" pending; "$DIR/run_getnext_tist2015_200.sh" aggregate ;;
  *) echo "Usage: $0 {shanghai|tist2015|aggregate|audit|all}" >&2; exit 2 ;;
esac
