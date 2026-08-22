#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing $PYTHON_BIN" >&2; exit 2; }
exec "$PYTHON_BIN" -m hybrid.beliefmove_results \
  --input "${INPUT:-results/beliefmove-evo/raw}" \
  --output-dir "${OUTPUT_DIR:-results/beliefmove-evo/aggregated}" \
  --results-md "${RESULTS_MD:-../../ideas/results.md}"
