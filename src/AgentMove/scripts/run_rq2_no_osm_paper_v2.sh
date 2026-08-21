#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"; MODEL_SLUG="${MODEL//[:\/]/-}"
DATA="${V2_DATA_DIR:-data/hybrid/Shanghai/neural_cgm/paper-v2-agentmove-200}"
BASE="${BASE_RESULTS:-results/hybrid/paper-v2-agentmove-200/${MODEL_SLUG}}"
OUT="${OUTPUT_DIR:-results/rq2/paper-v2-agentmove-200/${MODEL_SLUG}/no-osm}"
RESTARTS="${OLLAMA_RESTART_ATTEMPTS:-5}"

export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_HOST_URL="http://127.0.0.1:11434"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"
for required in "$DATA/validation.jsonl" "$DATA/test.jsonl" "$DATA/protocol_audit.json" "$BASE/full/metrics.json"; do
  [[ -f "$required" ]] || { echo "Missing v2 prerequisite: $required" >&2; exit 2; }
done
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("pass") and p["osm"]["coverage"] >= .9 else 2)' "$DATA/protocol_audit.json" || {
  echo "Baseline OSM coverage did not pass; refusing an uninterpretable no-OSM ablation." >&2; exit 2;
}

cmd=("$PYTHON_BIN" -m hybrid.cli --validation "$DATA/validation.jsonl" --test "$DATA/test.jsonl"
  --output-dir "$OUT" --top-k "${TOP_K:-10}" --top-m "${TOP_M:-5}"
  --extractor llm --platform Ollama --model-name "$MODEL"
  --llm-batch-size "${TOP_K:-10}" --llm-retries "${LLM_RETRIES:-2}"
  --llm-missing-policy "${LLM_MISSING_POLICY:-neutral}" --compact-evidence
  --llm-world-mode internal_only --variants full)
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
echo "RQ2 exact no-OSM completed: $OUT/full/metrics.json"
