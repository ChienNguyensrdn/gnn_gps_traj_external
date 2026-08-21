#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-all}"
MODEL="${OLLAMA_MODEL:-qwen2:7b}"
EXP="${BASELINE_EXP_NAME:-paper-original-baselines-shanghai-200}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
export nominatim_deploy_server_address="${nominatim_deploy_server_address:-127.0.0.1:18081}"

run_one() {
  local prompt_type="$1"
  "$PYTHON_BIN" -m agent --sample_one_traj_of_user --social_info_type address \
    --traj_min_len 3 --traj_max_len 50 --city_name Shanghai --prompt_num 200 \
    --workers 1 --exp_name "$EXP" --prompt_type "$prompt_type" \
    --model_name "$MODEL" --platform Ollama --use_int_venue --skip_existing_prediction
  local result_dir="results/$EXP/Shanghai/agentmove/$MODEL/$prompt_type"
  "$PYTHON_BIN" -m hybrid.original_baseline_metrics --input-dir "$result_dir" \
    --output "$result_dir/metrics.json"
}

./scripts/start_ollama.sh
case "$TARGET" in
  llmmob) run_one llmmob ;;
  agentmove) run_one agent_move_v6 ;;
  all) run_one llmmob; run_one agent_move_v6 ;;
  *) echo "Usage: $0 [llmmob|agentmove|all]" >&2; exit 2 ;;
esac
