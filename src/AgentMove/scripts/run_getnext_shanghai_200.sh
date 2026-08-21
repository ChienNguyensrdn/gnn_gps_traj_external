#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LIMIT="${QUERY_LIMIT:-200}"; EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"
DATA_DIR="${GETNEXT_DATA_DIR:-data/hybrid/Shanghai}"

prepare_if_missing() {
  local required=("$DATA_DIR/getnext/train.csv" "$DATA_DIR/getnext/val.csv" "$DATA_DIR/getnext/test.csv" "$DATA_DIR/candidate_ids.json")
  local missing=0 file input=""
  for file in "${required[@]}"; do [[ -f "$file" ]] || missing=1; done
  [[ "$missing" -eq 0 ]] && return
  for file in data/input_trajectories_clean/Shanghai_filtered.csv data/input_trajectories/Shanghai_filtered.csv; do
    [[ -f "$file" ]] && { input="$file"; break; }
  done
  if [[ -z "$input" ]]; then
    cat >&2 <<EOF
GETNext Shanghai data is missing under: $DATA_DIR
The normalized source is also missing. The repository ignores data/*, so a Git clone does not contain it.
Copy one of these directories from the preprocessing machine:
  data/hybrid/Shanghai/                         (fastest; already prepared)
or
  data/input_trajectories_clean/Shanghai_filtered.csv
  data/input_trajectories/Shanghai_filtered.csv
Then rerun this command.
EOF
    exit 2
  fi
  echo "Preparing GETNext Shanghai CSVs from $input"
  "$PYTHON_BIN" -m hybrid.prepare_dataset --dataset isp --input "$input" --city Shanghai --output-dir "$DATA_DIR"
}

prepare_if_missing
exec "$PYTHON_BIN" -m hybrid.getnext_baseline run \
  --dataset ISP-Shanghai --city Shanghai \
  --train-csv "$DATA_DIR/getnext/train.csv" \
  --validation-csv "$DATA_DIR/getnext/val.csv" \
  --test-csv "$DATA_DIR/getnext/test.csv" \
  --candidate-ids "$DATA_DIR/candidate_ids.json" \
  --output "results/getnext/ISP-Shanghai/limit-$LIMIT" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --test-limit "$LIMIT" \
  --train-limit "${TRAIN_LIMIT:-0}" --validation-limit "${VALIDATION_LIMIT:-0}" --device "${DEVICE:-auto}"
