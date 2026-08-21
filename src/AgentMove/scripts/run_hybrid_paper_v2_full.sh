#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Ollama is always local on port 11434.
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_HOST_URL="http://127.0.0.1:11434"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"

# Nominatim is an OSM web service, not an Ollama endpoint. The public HTTPS
# endpoint is rate-limited deliberately and all responses are cached locally.
export NOMINATIM_URL="${NOMINATIM_URL:-https://nominatim.openstreetmap.org}"
export NOMINATIM_DELAY="${NOMINATIM_DELAY:-1.1}"

echo "[1/5] Checking Ollama at http://127.0.0.1:11434"
./scripts/start_ollama.sh
./scripts/test_ollama.sh "$OLLAMA_MODEL"

echo "[2/5] Retrieving/caching OSM metadata through $NOMINATIM_URL"
echo "      First run can take about 65 minutes for 3,536 POIs at 1.1 s/request."
./scripts/run_hybrid_paper_v2.sh osm

echo "[3/5] Preparing the validation-tuned Stage-1 candidate generator"
./scripts/run_hybrid_paper_v2.sh prepare

echo "[4/5] Enforcing candidate-recall and OSM-coverage gates"
./scripts/run_hybrid_paper_v2.sh audit

echo "[5/5] Running compact one-call-per-query Hybrid v2"
./scripts/run_hybrid_paper_v2.sh run

RESULT="results/hybrid/paper-v2-agentmove-200/${OLLAMA_MODEL//[:\/]/-}/full/metrics.json"
echo "Hybrid v2 completed: $RESULT"
if [[ -f "$RESULT" ]]; then
  command -v jq >/dev/null 2>&1 && jq . "$RESULT" || sed -n '1,240p' "$RESULT"
fi
