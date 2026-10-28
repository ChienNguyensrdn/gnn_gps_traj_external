#!/usr/bin/env bash

BELIEFMOVE_ROOT="${BELIEFMOVE_ROOT:-results/beliefmove-faithful/v2}"
BELIEFMOVE_DATASETS="${BELIEFMOVE_DATASETS:-tist2015 www2019 yjmob100k}"
TIST_CITIES="${TIST_CITIES:-Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow}"
BELIEFMOVE_SEEDS="${BELIEFMOVE_SEEDS:-42 43 44}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
OLLAMA_MODEL_DIGEST="${OLLAMA_MODEL_DIGEST:-unresolved}"
TEACHER_TRAIN_LIMIT="${TEACHER_TRAIN_LIMIT:-1000}"
TEACHER_VAL_LIMIT="${TEACHER_VAL_LIMIT:-250}"
TEACHER_TEST_PANEL_LIMIT="${TEACHER_TEST_PANEL_LIMIT:-500}"
PILOT_REPEAT_VAL_LIMIT="${PILOT_REPEAT_VAL_LIMIT:-50}"
PILOT_REPEAT_COUNT="${PILOT_REPEAT_COUNT:-3}"
REGION_CELL_METERS="${REGION_CELL_METERS:-500}"
MODEL_SLUG="${OLLAMA_MODEL//[:\/]/-}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

beliefmove_units() {
  case "$1" in
    tist2015) echo "$TIST_CITIES" ;;
    www2019) echo Shanghai ;;
    yjmob100k) echo YJMob ;;
    *) echo "Unknown dataset: $1" >&2; return 2 ;;
  esac
}

beliefmove_configure() {
  local dataset="$1" unit="$2"
  BM_DATASET="$dataset"
  BM_UNIT="$unit"
  case "$dataset" in
    tist2015)
      BM_LABEL="TIST2015-$unit"
      BM_DATA_BASE="data/hybrid/TIST2015/$unit" ;;
    www2019)
      BM_LABEL="WWW2019-Shanghai-ISP"
      BM_DATA_BASE="data/hybrid/WWW2019/Shanghai" ;;
    yjmob100k)
      BM_LABEL="YJMob100K-Dataset1"
      BM_DATA_BASE="data/hybrid/YJMob100K/YJMob" ;;
  esac
  BM_OUTPUT="$BELIEFMOVE_ROOT/$dataset/$unit"
  export BM_DATASET BM_UNIT BM_LABEL BM_DATA_BASE BM_OUTPUT
}

beliefmove_require_file() {
  [[ -f "$1" ]] || { echo "MISSING: $1" >&2; return 2; }
}

beliefmove_audit_base() {
  beliefmove_require_file "$BM_DATA_BASE/candidate_ids.json"
  beliefmove_require_file "$BM_DATA_BASE/getnext/train.csv"
  beliefmove_require_file "$BM_DATA_BASE/getnext/val.csv"
  beliefmove_require_file "$BM_DATA_BASE/getnext/test.csv"
  beliefmove_require_file "$BM_DATA_BASE/neural_cgm/best.pt"
  echo "ready dataset=$BM_LABEL data=$BM_DATA_BASE output=$BM_OUTPUT"
}

beliefmove_for_each_unit() {
  local callback="$1" dataset unit
  for dataset in $BELIEFMOVE_DATASETS; do
    for unit in $(beliefmove_units "$dataset"); do
      beliefmove_configure "$dataset" "$unit"
      "$callback"
    done
  done
}
