#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tist2015_common.sh"

TARGET="${1:-audit}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EXP_NAME="${EXP_NAME:-tist2015-agentmove-original-no-osm/limit-$QUERY_LIMIT}"
ROOT="$(tist2015_agentmove_root)"
cd "$ROOT"

tist2015_require_positive_integer QUERY_LIMIT "$QUERY_LIMIT"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python environment: $PYTHON_BIN" >&2; exit 2; }
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"
OUTPUT_ROOT="results/$EXP_NAME"

prepare() {
  "$PYTHON_BIN" -m processing.prepare_tist2015_agentmove_no_osm
}

metrics_path() {
  printf '%s/%s/agentmove/%s/agent_move_v6/metrics.json\n' "$OUTPUT_ROOT" "$1" "$OLLAMA_MODEL"
}

run_city() {
  local city="$1" result_dir
  [[ -f "data/input_trajectories_clean/${city}_filtered.csv" ]] || prepare
  echo "AgentMove TIST2015 city=$city limit=$QUERY_LIMIT model=$OLLAMA_MODEL"
  echo "protocol=no-OSM matched; Ollama=http://127.0.0.1:11434/v1"
  "$PYTHON_BIN" -m agent \
    --sample_one_traj_of_user --social_info_type category \
    --traj_min_len 3 --traj_max_len 50 --city_name "$city" \
    --prompt_num "$QUERY_LIMIT" --workers 1 --exp_name "$EXP_NAME" \
    --prompt_type agent_move_v6 --model_name "$OLLAMA_MODEL" \
    --platform Ollama --use_int_venue --skip_existing_prediction
  result_dir="$OUTPUT_ROOT/$city/agentmove/$OLLAMA_MODEL/agent_move_v6"
  "$PYTHON_BIN" -m hybrid.original_baseline_metrics \
    --input-dir "$result_dir" --output "$result_dir/metrics.json"
}

audit() {
  local city path count
  echo "AgentMove TIST2015 audit model=$OLLAMA_MODEL limit=$QUERY_LIMIT port=11434"
  for city in "${TIST2015_CITIES[@]}"; do
    path="$(metrics_path "$city")"
    if [[ -f "$path" ]]; then
      count="$(find "${path%/metrics.json}" -maxdepth 1 -name '*.json' ! -name metrics.json | wc -l | tr -d ' ')"
      echo "complete $city predictions=$count"
    elif [[ -d "${path%/metrics.json}" ]]; then
      count="$(find "${path%/metrics.json}" -maxdepth 1 -name '*.json' ! -name metrics.json | wc -l | tr -d ' ')"
      echo "partial  $city predictions=$count/$QUERY_LIMIT"
    else
      echo "pending  $city"
    fi
  done
}

aggregate() {
  "$PYTHON_BIN" -m hybrid.tist2015_original_agentmove_aggregate \
    --input-root "$OUTPUT_ROOT" --model "$OLLAMA_MODEL" --query-limit "$QUERY_LIMIT"
}

case "$TARGET" in
  prepare) prepare ;;
  audit) audit ;;
  aggregate) aggregate ;;
  all)
    prepare
    ./scripts/start_ollama.sh
    for city in "${TIST2015_CITIES[@]}"; do run_city "$city"; done
    aggregate
    ;;
  pending)
    prepare
    ./scripts/start_ollama.sh
    for city in "${TIST2015_CITIES[@]}"; do
      [[ -f "$(metrics_path "$city")" ]] || run_city "$city"
    done
    aggregate
    ;;
  *)
    tist2015_is_city "$TARGET" || { echo "Usage: $0 <prepare|audit|aggregate|pending|all|city>" >&2; exit 2; }
    prepare
    ./scripts/start_ollama.sh
    run_city "$TARGET"
    ;;
esac
