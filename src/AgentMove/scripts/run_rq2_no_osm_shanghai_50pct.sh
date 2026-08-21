#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
BASE="${BASE_RESULTS:-results/hybrid/shanghai-neural-cgm-50-seed42/qwen2-7b}"
DATA="${DATA_DIR:-data/hybrid/Shanghai/neural_cgm/sample-50-seed42}"
OUT="${OUTPUT_DIR:-results/rq2/shanghai-neural-cgm-50-seed42/qwen2-7b/no-osm}"

./scripts/start_ollama.sh
echo "RQ2 no-OSM: internal LLM knowledge only; output=$OUT"
.venv/bin/python -m hybrid.no_osm_ablation \
  --validation "$DATA/validation.jsonl" --test "$DATA/test.jsonl" \
  --calibration "$BASE/calibration.json" --output-dir "$OUT" \
  --model-name "$MODEL" --platform Ollama --top-k "${TOP_K:-10}" \
  --top-m "${TOP_M:-5}" --batch-size "${LLM_BATCH_SIZE:-3}" \
  --retries "${LLM_RETRIES:-2}" --missing-policy "${LLM_MISSING_POLICY:-neutral}" \
  --min-osm-coverage "${MIN_OSM_COVERAGE:-0.9}"
