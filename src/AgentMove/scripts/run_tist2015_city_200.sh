#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tist2015_common.sh"

# Run one TIST2015 city at a time with a bounded, deterministic query count.
# Usage:
#   ./scripts/run_tist2015_city_200.sh Tokyo
#   QUERY_LIMIT=200 OLLAMA_MODEL=qwen2:7b ./scripts/run_tist2015_city_200.sh Paris

CITY="${1:-${CITY:-}}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
TOP_K="${TOP_K:-10}"
TOP_M="${TOP_M:-5}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-10}"
LLM_RETRIES="${LLM_RETRIES:-2}"
LLM_MISSING_POLICY="${LLM_MISSING_POLICY:-neutral}"

ROOT="$(tist2015_agentmove_root)"
cd "$ROOT"

if [[ -z "$CITY" ]]; then
  echo "Usage: $0 <city>" >&2
  echo "Cities: ${TIST2015_CITIES[*]}" >&2
  exit 2
fi
if ! tist2015_is_city "$CITY"; then
  echo "Invalid city: $CITY" >&2
  echo "Cities: ${TIST2015_CITIES[*]}" >&2
  exit 2
fi
tist2015_require_positive_integer QUERY_LIMIT "$QUERY_LIMIT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
tist2015_require_python "$PYTHON_BIN"
BASE="data/hybrid/TIST2015/$CITY/neural_cgm"
METADATA="data/hybrid/TIST2015/$CITY/candidate_metadata.json"
for required in "$BASE/best.pt" "$BASE/validation.jsonl" "$BASE/test.jsonl" "$METADATA"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done

read -r OSM_COVERAGE OSM_LABEL < <("$PYTHON_BIN" - "$METADATA" <<'PY'
import json, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = list(payload.values()) if isinstance(payload, dict) else payload
covered = sum(bool(row.get("address") or row.get("osm_address") or row.get("display_name")) for row in rows)
coverage = covered / len(rows) if rows else 0.0
print(f"{coverage:.6f}", "full-osm" if coverage >= 0.90 else "no-osm")
PY
)

MODEL_SLUG="$(tist2015_model_slug "$OLLAMA_MODEL")"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/tist2015-hybrid/$MODEL_SLUG/limit-$QUERY_LIMIT/$OSM_LABEL}"
OUTPUT_DIR="$OUTPUT_ROOT/$CITY"

export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"

echo "TIST2015 city run"
echo "city=$CITY validation_limit=$QUERY_LIMIT test_limit=$QUERY_LIMIT"
echo "model=$OLLAMA_MODEL ollama=http://127.0.0.1:11434/v1"
echo "top_k=$TOP_K top_m=$TOP_M batch=$LLM_BATCH_SIZE"
echo "osm_coverage=$OSM_COVERAGE result_label=$OSM_LABEL"
echo "output=$OUTPUT_DIR"
if [[ "$OSM_LABEL" == "no-osm" ]]; then
  echo "WARNING: OSM coverage is below 90%; this run is an ablation and must not populate Ours (full)." >&2
fi

./scripts/start_ollama.sh
OLLAMA_BASE_URL="http://127.0.0.1:11434/v1" ./scripts/test_ollama.sh "$OLLAMA_MODEL"

"$PYTHON_BIN" -m hybrid.cli \
  --validation "$BASE/validation.jsonl" \
  --test "$BASE/test.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --validation-limit "$QUERY_LIMIT" \
  --test-limit "$QUERY_LIMIT" \
  --top-k "$TOP_K" \
  --top-m "$TOP_M" \
  --extractor llm \
  --platform Ollama \
  --model-name "$OLLAMA_MODEL" \
  --llm-batch-size "$LLM_BATCH_SIZE" \
  --llm-retries "$LLM_RETRIES" \
  --llm-missing-policy "$LLM_MISSING_POLICY" \
  --compact-evidence \
  --variants full stage1_only stage1_uncalibrated

echo "Completed: $CITY"
echo "Metrics: $OUTPUT_DIR/full/metrics.json"
