#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATASET="${DATASET:-isp}"
CITY="${CITY:-Shanghai}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
NOMINATIM_URL="${NOMINATIM_URL:-http://127.0.0.1:8080}"
USE_OSM="${USE_OSM:-0}"

TIST_CITIES=(Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow)

usage() {
  cat <<'EOF'
Usage: ./scripts/hybrid_pipeline.sh COMMAND

Commands:
  extract      Extract/normalize ISP-Shanghai or TIST2015 city data
  osm          Retrieve OSM address fields (optional; requires Nominatim)
  prepare      Create temporal splits, Markov logits, metadata and GETNext CSVs
  ollama-test  Verify the selected Ollama model and structured JSON endpoint
  run          Run LLM evidence extraction, explicit Bayesian Network and RQ reports
  all          Run extract, optional osm, prepare, ollama-test, and run

Environment:
  DATASET=isp|tist2015         Default: isp
  CITY=Shanghai|Tokyo|...      Default: Shanghai
  OLLAMA_MODEL=qwen2:7b        Must be installed in Ollama
  USE_OSM=0|1                  Default: 0
  NOMINATIM_URL=http://...     Local Nominatim default: http://127.0.0.1:8080
  TOP_K=10 TOP_M=5 LLM_BATCH_SIZE=3 LLM_RETRIES=2
  VALIDATION_LIMIT=N TEST_LIMIT=N  Optional smoke-test limits
  OLLAMA_RESTART_ATTEMPTS=3       Auto-restarts/resumes after connection loss
EOF
}

normalized_path() {
  if [[ "$USE_OSM" == "1" && -f "data/input_trajectories_clean/${CITY}_filtered.csv" ]]; then
    echo "data/input_trajectories_clean/${CITY}_filtered.csv"
  else
    echo "data/input_trajectories/${CITY}_filtered.csv"
  fi
}

extract_data() {
  if [[ "$DATASET" == "isp" ]]; then
    [[ "$CITY" == "Shanghai" ]] || { echo "ISP dataset requires CITY=Shanghai" >&2; exit 2; }
    local raw="data/dataset_www2019"
    mkdir -p "$raw"
    if [[ ! -f "$raw/isp" || ! -f "$raw/poi.txt" ]]; then
      unzip -o data/www2019_isp_data.zip -d "$raw"
      unzip -o "$raw/isp.zip" -d "$raw"
      unzip -o "$raw/poi.txt.zip" -d "$raw"
    fi
    [[ -f data/input_trajectories/Shanghai_filtered.csv ]] || "$PYTHON_BIN" -m processing.process_isp_shanghai
  elif [[ "$DATASET" == "tist2015" ]]; then
    local raw="data/dataset_tist2015"
    for file in dataset_TIST2015_Checkins.txt dataset_TIST2015_POIs.txt dataset_TIST2015_Cities.txt; do
      [[ -f "$raw/$file" ]] || { echo "Missing $raw/$file" >&2; exit 2; }
    done
    "$PYTHON_BIN" -m processing.extract_tist2015_cities \
      --input-dir "$raw" --output-dir data/input_trajectories --cities "$CITY"
  else
    echo "Unsupported DATASET=$DATASET" >&2; exit 2
  fi
  echo "Normalized data: data/input_trajectories/${CITY}_filtered.csv"
}

enrich_osm() {
  local input="data/input_trajectories/${CITY}_filtered.csv"
  [[ -f "$input" ]] || { echo "Run extract first: missing $input" >&2; exit 2; }
  curl -fsS --max-time 5 "$NOMINATIM_URL/status.php" >/dev/null 2>&1 || \
    curl -fsS --max-time 5 "$NOMINATIM_URL/search?q=Hanoi&format=jsonv2&limit=1" >/dev/null || {
      echo "Nominatim is unavailable at $NOMINATIM_URL" >&2; exit 2;
    }
  "$PYTHON_BIN" -m hybrid.enrich_osm \
    --input "$input" \
    --output "data/input_trajectories_clean/${CITY}_filtered.csv" \
    --cache "data/nominatim/${CITY}_hybrid.jsonl" \
    --base-url "$NOMINATIM_URL" \
    --delay-seconds "${NOMINATIM_DELAY:-0}"
}

prepare_data() {
  local input
  input="$(normalized_path)"
  [[ -f "$input" ]] || { echo "Run extract first: missing $input" >&2; exit 2; }
  local output
  if [[ "$DATASET" == "isp" ]]; then output="data/hybrid/Shanghai"; else output="data/hybrid/TIST2015/$CITY"; fi
  "$PYTHON_BIN" -m hybrid.prepare_dataset \
    --dataset "$DATASET" --input "$input" --city "$CITY" --output-dir "$output" \
    --history-limit "${HISTORY_LIMIT:-40}" --context-limit "${CONTEXT_LIMIT:-6}"
}

ollama_test() {
  ./scripts/start_ollama.sh
  ./scripts/test_ollama.sh "$OLLAMA_MODEL"
}

run_hybrid() {
  local data_dir
  if [[ "$DATASET" == "isp" ]]; then data_dir="data/hybrid/Shanghai"; else data_dir="data/hybrid/TIST2015/$CITY"; fi
  local validation_file="${VALIDATION_FILE:-$data_dir/validation.jsonl}"
  local test_file="${TEST_FILE:-$data_dir/test.jsonl}"
  local output_dir="${OUTPUT_DIR:-results/hybrid/${DATASET}-${CITY}-ollama-${OLLAMA_MODEL//[:\/]/-}}"
  [[ -f "$validation_file" && -f "$test_file" ]] || {
    echo "Run prepare first; hybrid JSONL files are missing in $data_dir" >&2; exit 2;
  }
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
  export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"
  # Keep the array non-empty for macOS Bash 3.2 + `set -u`. Expanding a
  # separate empty array raises "unbound variable" on that shell version.
  local cmd=(
    "$PYTHON_BIN" -m hybrid.cli
    --validation "$validation_file"
    --test "$test_file"
    --output-dir "$output_dir"
    --top-k "${TOP_K:-10}" --top-m "${TOP_M:-5}"
    --extractor llm --platform Ollama --model-name "$OLLAMA_MODEL"
    --llm-batch-size "${LLM_BATCH_SIZE:-3}" --llm-retries "${LLM_RETRIES:-2}"
    --llm-missing-policy "${LLM_MISSING_POLICY:-neutral}"
  )
  [[ "${COMPACT_EVIDENCE:-0}" == "1" ]] && cmd+=(--compact-evidence)
  if [[ -n "${HYBRID_VARIANTS:-}" ]]; then
    read -r -a requested_variants <<< "$HYBRID_VARIANTS"
    cmd+=(--variants "${requested_variants[@]}")
  fi
  [[ -n "${VALIDATION_LIMIT:-}" ]] && cmd+=(--validation-limit "$VALIDATION_LIMIT")
  [[ -n "${TEST_LIMIT:-}" ]] && cmd+=(--test-limit "$TEST_LIMIT")
  if [[ "${HYBRID_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  # Validate/start the local endpoint before a potentially multi-hour run.
  ./scripts/start_ollama.sh
  local ollama_host="${OLLAMA_BASE_URL%/v1}"
  local tags
  tags="$(curl -fsS --max-time 5 "$ollama_host/api/tags")" || {
    echo "Ollama is not reachable at $ollama_host" >&2
    return 75
  }
  if ! printf '%s' "$tags" | "$PYTHON_BIN" -c \
    'import json,sys; wanted=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if any(m.get("name")==wanted or m.get("model")==wanted for m in data.get("models", [])) else 1)' \
    "$OLLAMA_MODEL"; then
    echo "Ollama model '$OLLAMA_MODEL' is not installed. Run: ollama pull $OLLAMA_MODEL" >&2
    return 2
  fi

  local restart_attempts="${OLLAMA_RESTART_ATTEMPTS:-3}"
  local attempt=0
  local status=0
  while :; do
    set +e
    "${cmd[@]}"
    status=$?
    set -e
    [[ "$status" -eq 0 ]] && return 0
    [[ "$status" -eq 75 ]] || return "$status"
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$restart_attempts" ]]; then
      echo "Ollama remained unavailable after $restart_attempts restart attempts." >&2
      return 75
    fi
    echo "Ollama connection was lost; restart $attempt/$restart_attempts, then resume cache ..." >&2
    ./scripts/start_ollama.sh
  done
}

command="${1:-help}"
case "$command" in
  extract) extract_data ;;
  osm) enrich_osm ;;
  prepare) prepare_data ;;
  ollama-test) ollama_test ;;
  run) run_hybrid ;;
  all) extract_data; [[ "$USE_OSM" == "1" ]] && enrich_osm; prepare_data; ollama_test; run_hybrid ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
