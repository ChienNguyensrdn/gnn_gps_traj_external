#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/tist2015_common.sh"

cd "$(tist2015_agentmove_root)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_DIR="data/dataset_tist2015"
NORMALIZED_DIR="data/input_trajectories"
OUTPUT_ROOT="data/hybrid/TIST2015"

if [[ ! -f "$RAW_DIR/dataset_TIST2015_Checkins.txt" ]]; then
  echo "Missing raw TIST2015 files under $RAW_DIR" >&2
  echo "Download dataset_tist2015.zip, extract it there, then rerun this script." >&2
  exit 1
fi

"$PYTHON_BIN" -m processing.extract_tist2015_cities \
  --input-dir "$RAW_DIR" \
  --output-dir "$NORMALIZED_DIR" \
  --cities "${TIST2015_CITIES[@]}"

for city in "${TIST2015_CITIES[@]}"; do
  "$PYTHON_BIN" -m hybrid.prepare_dataset \
    --dataset tist2015 \
    --input "$NORMALIZED_DIR/${city}_filtered.csv" \
    --city "$city" \
    --output-dir "$OUTPUT_ROOT/$city"
done

echo "Hybrid TIST2015 files written below $OUTPUT_ROOT"
