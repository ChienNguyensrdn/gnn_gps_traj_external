#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
BASE="${BASE_RESULTS:-results/hybrid/shanghai-neural-cgm-50-seed42/qwen2-7b}"
TEST="${TEST_JSONL:-data/hybrid/Shanghai/neural_cgm/sample-50-seed42/test.jsonl}"
OUT="${OUTPUT_DIR:-results/rq2/shanghai-neural-cgm-50-seed42/qwen2-7b/free-text-rerank}"

./scripts/start_ollama.sh
echo "RQ2 free-text reranking: test=$TEST output=$OUT"
args=(--test "$TEST" --evidence-cache "$BASE/evidence_cache.jsonl"
  --calibration "$BASE/calibration.json" --output-dir "$OUT"
  --model-name "$MODEL" --platform Ollama --top-k "${TOP_K:-10}"
  --top-m "${TOP_M:-5}" --retries "${LLM_RETRIES:-2}")
if [[ -n "${TEST_LIMIT:-}" ]]; then args+=(--limit "$TEST_LIMIT"); fi
.venv/bin/python -m hybrid.free_text_rerank "${args[@]}"
