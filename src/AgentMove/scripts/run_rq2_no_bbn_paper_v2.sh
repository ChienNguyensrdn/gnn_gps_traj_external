#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${MODEL//[:\/]/-}"
BASE="${BASE_RESULTS:-results/hybrid/paper-v2-agentmove-200/${MODEL_SLUG}}"
TEST="${TEST_JSONL:-data/hybrid/Shanghai/neural_cgm/paper-v2-agentmove-200/test.jsonl}"
OUT="${OUTPUT_DIR:-results/rq2/paper-v2-agentmove-200/${MODEL_SLUG}/no-bbn-free-text}"
RESTARTS="${OLLAMA_RESTART_ATTEMPTS:-5}"

export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_HOST_URL="http://127.0.0.1:11434"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"

for required in "$TEST" "$BASE/evidence_cache.jsonl" "$BASE/calibration.json"; do
  [[ -f "$required" ]] || { echo "Missing prerequisite: $required" >&2; exit 2; }
done

cmd=("$PYTHON_BIN" -m hybrid.free_text_rerank
  --test "$TEST" --evidence-cache "$BASE/evidence_cache.jsonl"
  --calibration "$BASE/calibration.json" --output-dir "$OUT"
  --model-name "$MODEL" --platform Ollama --top-k "${TOP_K:-10}"
  --top-m "${TOP_M:-5}" --retries "${LLM_RETRIES:-2}")
[[ -n "${TEST_LIMIT:-}" ]] && cmd+=(--limit "$TEST_LIMIT")

if [[ "${HYBRID_DRY_RUN:-0}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
./scripts/start_ollama.sh
attempt=0
while :; do
  set +e; "${cmd[@]}"; exit_code=$?; set -e
  [[ "$exit_code" -eq 0 ]] && break
  [[ "$exit_code" -eq 75 ]] || exit "$exit_code"
  attempt=$((attempt + 1)); [[ "$attempt" -le "$RESTARTS" ]] || exit 75
  ./scripts/start_ollama.sh
done
echo "RQ2 exact w/o BBN completed: $OUT/metrics.json"
