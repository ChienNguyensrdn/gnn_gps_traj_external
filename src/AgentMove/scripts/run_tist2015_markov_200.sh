#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/tist2015_common.sh"

TARGET="${1:-audit}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="$(tist2015_agentmove_root)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/tist2015-markov/limit-$QUERY_LIMIT}"
cd "$ROOT"

tist2015_require_positive_integer QUERY_LIMIT "$QUERY_LIMIT"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python environment: $PYTHON_BIN" >&2; exit 2; }

run_city() {
  local city="$1" input="data/hybrid/TIST2015/$1/test.jsonl"
  [[ -f "$input" ]] || { echo "Missing Markov test bundle: $input" >&2; return 2; }
  echo "Markov/Bi-gram city=$city limit=$QUERY_LIMIT (CPU only; no Ollama)"
  "$PYTHON_BIN" -m hybrid.tist2015_markov city \
    --input "$input" --output-dir "$OUTPUT_ROOT/$city" --limit "$QUERY_LIMIT"
}

audit() {
  local city path
  echo "TIST2015 Markov/Bi-gram audit limit=$QUERY_LIMIT"
  for city in "${TIST2015_CITIES[@]}"; do
    path="$OUTPUT_ROOT/$city/metrics.json"
    if [[ -f "$path" ]]; then
      echo "complete $city queries=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["queries"])' "$path")"
    else
      echo "pending  $city"
    fi
  done
}

aggregate() {
  "$PYTHON_BIN" -m hybrid.tist2015_markov aggregate \
    --input-root "$OUTPUT_ROOT" --limit "$QUERY_LIMIT"
  echo "Summary: $OUTPUT_ROOT/tist2015_markov_summary.json"
  echo "Table II cells: $OUTPUT_ROOT/tist2015_markov_table2_cells.tex"
}

case "$TARGET" in
  audit) audit ;;
  aggregate) aggregate ;;
  all)
    for city in "${TIST2015_CITIES[@]}"; do run_city "$city"; done
    aggregate
    ;;
  pending)
    for city in "${TIST2015_CITIES[@]}"; do
      [[ -f "$OUTPUT_ROOT/$city/metrics.json" ]] || run_city "$city"
    done
    aggregate
    ;;
  *)
    tist2015_is_city "$TARGET" || { echo "Usage: $0 <audit|aggregate|pending|all|city>" >&2; exit 2; }
    run_city "$TARGET"
    ;;
esac
