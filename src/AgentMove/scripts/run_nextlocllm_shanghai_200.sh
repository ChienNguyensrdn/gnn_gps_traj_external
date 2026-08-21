#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"; DATA_DIR="${NEXTLOCLLM_DATA_DIR:-data/hybrid/Shanghai}"
LIMIT="${QUERY_LIMIT:-200}"; MODEL="${OLLAMA_MODEL:-qwen2:7b}"; EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"
for file in "$DATA_DIR/getnext/train.csv" "$DATA_DIR/getnext/val.csv" "$DATA_DIR/getnext/test.csv" "$DATA_DIR/candidate_ids.json"; do
  [[ -f "$file" ]] || { echo "Missing $file. Prepare/copy data/hybrid/Shanghai first." >&2; exit 2; }
done
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
./scripts/start_ollama.sh
exec "$PYTHON_BIN" -m hybrid.nextlocllm_baseline run --dataset ISP-Shanghai --city Shanghai \
  --train-csv "$DATA_DIR/getnext/train.csv" --validation-csv "$DATA_DIR/getnext/val.csv" --test-csv "$DATA_DIR/getnext/test.csv" --candidate-ids "$DATA_DIR/candidate_ids.json" \
  --embedding-cache "$DATA_DIR/nextlocllm/${MODEL//[:\/]/-}-category-embeddings.json" --embedding-model "$MODEL" --ollama-base-url "$OLLAMA_BASE_URL" \
  --output "results/nextlocllm/ISP-Shanghai/$MODEL/ranking-enhanced/limit-$LIMIT" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --test-limit "$LIMIT" \
  --ranking-loss-weight "${RANKING_LOSS_WEIGHT:-1.0}" --coordinate-loss-weight "${COORDINATE_LOSS_WEIGHT:-0.25}" --distance-weight "${DISTANCE_WEIGHT:-1.0}" \
  --train-limit "${TRAIN_LIMIT:-0}" --validation-limit "${VALIDATION_LIMIT:-0}" --device "${DEVICE:-auto}"
