#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_DIR="data/dataset_www2019"
NORMALIZED="data/input_trajectories/Shanghai_filtered.csv"
OUTPUT_DIR="data/hybrid/Shanghai"

if [[ ! -f "$RAW_DIR/isp" || ! -f "$RAW_DIR/poi.txt" ]]; then
  mkdir -p "$RAW_DIR"
  unzip -o data/www2019_isp_data.zip -d "$RAW_DIR"
  unzip -o "$RAW_DIR/isp.zip" -d "$RAW_DIR"
  unzip -o "$RAW_DIR/poi.txt.zip" -d "$RAW_DIR"
fi

if [[ ! -f "$NORMALIZED" ]]; then
  "$PYTHON_BIN" -m processing.process_isp_shanghai
fi

"$PYTHON_BIN" -m hybrid.prepare_dataset \
  --dataset isp \
  --input "$NORMALIZED" \
  --city Shanghai \
  --output-dir "$OUTPUT_DIR"

echo "Hybrid ISP files written to $OUTPUT_DIR"
