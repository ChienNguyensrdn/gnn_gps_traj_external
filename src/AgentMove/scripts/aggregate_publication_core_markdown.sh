#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT="${PUBLICATION_REPORT_MD:-../../ideas/results_publication_core_3datasets.md}"
CITIES="${CITIES:-Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow}"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }

extra=()
if [[ "${ALLOW_INCOMPLETE:-0}" == 1 ]]; then
  extra+=(--allow-incomplete)
fi

"$PY" hybrid/publication_core_report.py \
  --source-dir ../../ideas \
  --output "$OUTPUT" \
  --cities $CITIES \
  "${extra[@]}"

echo "Combined Markdown: $OUTPUT"
