#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
RESULT_ROOT="${RESULT_ROOT:-results/getnext/TIST2015/limit-$QUERY_LIMIT}"
OUTPUT="${OUTPUT:-$RESULT_ROOT/macro_average_checked.json}"
CITIES=(Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow)

case "${1:-aggregate}" in
  audit)
    printf '%-14s %s\n' city status
    for city in "${CITIES[@]}"; do
      file="$RESULT_ROOT/$city/metrics.json"
      if [[ -f "$file" ]]; then
        count="$($PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("count",p.get("queries",0)))' "$file")"
        printf '%-14s done count=%s\n' "$city" "$count"
      else
        printf '%-14s missing\n' "$city"
      fi
    done
    ;;
  aggregate)
    set +e
    "$PYTHON_BIN" -m hybrid.getnext_results_aggregate \
      --root "$RESULT_ROOT" --output "$OUTPUT" --query-limit "$QUERY_LIMIT"
    status=$?
    set -e
    if [[ "$status" -eq 3 ]]; then
      echo "Interim summary was written, but it is not a complete compatible 12-city result." >&2
    elif [[ "$status" -ne 0 ]]; then
      exit "$status"
    fi
    ;;
  *)
    echo "Usage: $0 {audit|aggregate}" >&2
    exit 2
    ;;
esac
