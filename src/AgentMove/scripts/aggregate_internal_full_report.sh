#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${MODEL//[:\/]/-}"
LIMIT="${LLM_LIMIT:-200}"
SEEDS="${RQ_SEEDS:-42 43 44}"
CITIES="Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow"
RANDOM_SEEDS="${RQ8_RANDOM_SEEDS:-$(seq 42 91)}"
OUTPUT="${INTERNAL_SUMMARY_JSON:-results/beliefmove-evo/aggregated/12city/summary_internal.json}"
REPORT="${INTERNAL_REPORT_MD:-../../ideas/report_summary_internal.md}"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }

"$PY" -m hybrid.all_cities_summary \
  --results-root results/beliefmove-evo/artifacts/full \
  --cities $CITIES \
  --seeds $SEEDS \
  --model-slug "$MODEL_SLUG" \
  --limit "$LIMIT" \
  --random-seeds $RANDOM_SEEDS \
  --scope all \
  --allow-gpu-contention \
  --output "$OUTPUT" \
  --markdown "$REPORT"

echo "Internal JSON: $OUTPUT"
echo "Internal report: $REPORT"
echo "WARNING: RQ12 contention is accepted only for internal analysis, not publication."
