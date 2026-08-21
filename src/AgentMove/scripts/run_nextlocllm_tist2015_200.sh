#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; LIMIT="${QUERY_LIMIT:-200}"; MODEL="${OLLAMA_MODEL:-qwen2:7b}"; EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"
CITIES=(Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow)
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
run_city() { local city="$1" base="data/hybrid/TIST2015/$1" out="results/nextlocllm/TIST2015/$MODEL/ranking-enhanced/limit-$LIMIT/$1" file;
  [[ -f "$out/metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip completed $city"; return; }
  for file in "$base/getnext/train.csv" "$base/getnext/val.csv" "$base/getnext/test.csv" "$base/candidate_ids.json"; do [[ -f "$file" ]] || { echo "Missing $file" >&2; return 2; }; done
  ./scripts/start_ollama.sh
  "$PYTHON_BIN" -m hybrid.nextlocllm_baseline run --dataset TIST2015 --city "$city" --train-csv "$base/getnext/train.csv" --validation-csv "$base/getnext/val.csv" --test-csv "$base/getnext/test.csv" --candidate-ids "$base/candidate_ids.json" \
    --embedding-cache "$base/nextlocllm/${MODEL//[:\/]/-}-category-embeddings.json" --embedding-model "$MODEL" --ollama-base-url "$OLLAMA_BASE_URL" --output "$out" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --test-limit "$LIMIT" --ranking-loss-weight "${RANKING_LOSS_WEIGHT:-1.0}" --coordinate-loss-weight "${COORDINATE_LOSS_WEIGHT:-0.25}" --distance-weight "${DISTANCE_WEIGHT:-1.0}" --train-limit "${TRAIN_LIMIT:-0}" --validation-limit "${VALIDATION_LIMIT:-0}" --device "${DEVICE:-auto}"; }
case "${1:-pending}" in
  pending) for city in "${CITIES[@]}"; do run_city "$city"; done ;;
  audit) for city in "${CITIES[@]}"; do [[ -f "results/nextlocllm/TIST2015/$MODEL/ranking-enhanced/limit-$LIMIT/$city/metrics.json" ]] && echo "done $city" || echo "pending $city"; done ;;
  aggregate) "$PYTHON_BIN" -m hybrid.nextlocllm_baseline aggregate --root "results/nextlocllm/TIST2015/$MODEL/ranking-enhanced/limit-$LIMIT" --cities "${CITIES[@]}" --output "results/nextlocllm/TIST2015/$MODEL/ranking-enhanced/limit-$LIMIT/macro_average.json" ;;
  *) run_city "$1" ;;
esac
