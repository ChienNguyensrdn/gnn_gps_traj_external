#!/usr/bin/env bash

PHASE2_ROOT="${PHASE2_ROOT:-results/phase2}"
PHASE2_DATASETS="${PHASE2_DATASETS:-tist2015 www2019 yjmob100k}"
TIST_CITIES="${TIST_CITIES:-Tokyo Nairobi NewYork Sydney CapeTown Paris Beijing Mumbai SanFrancisco London SaoPaulo Moscow}"
RQ_SEEDS="${RQ_SEEDS:-42 43 44}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LLM_LIMIT="${LLM_LIMIT:-1000}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
MODEL_SLUG="${OLLAMA_MODEL//[:\/]/-}"

phase2_units() {
  case "$1" in
    tist2015) echo "$TIST_CITIES" ;;
    www2019) echo Shanghai ;;
    yjmob100k) echo YJMob ;;
    *) echo "Unknown dataset: $1" >&2; return 2 ;;
  esac
}

phase2_configure() {
  local dataset="$1" unit="$2"
  case "$dataset" in
    tist2015)
      P2_CITY="$unit"; P2_LABEL="TIST2015-$unit"
      P2_SOURCE_DATASET=tist2015
      P2_DATA_BASE="data/hybrid/TIST2015/$unit"
      P2_CACHE="results/tist2015-hybrid/$MODEL_SLUG/limit-$LLM_LIMIT/no-osm/$unit" ;;
    www2019)
      P2_CITY=Shanghai; P2_LABEL="WWW2019-Shanghai-ISP"
      P2_SOURCE_DATASET=isp
      P2_DATA_BASE="data/hybrid/WWW2019/Shanghai"
      P2_CACHE="results/www2019-hybrid/$MODEL_SLUG/limit-$LLM_LIMIT/no-osm/Shanghai" ;;
    yjmob100k)
      P2_CITY=YJMob; P2_LABEL="YJMob100K-Dataset1"
      P2_SOURCE_DATASET=yjmob100k
      P2_DATA_BASE="data/hybrid/YJMob100K/YJMob"
      P2_CACHE="results/yjmob100k-hybrid/$MODEL_SLUG/limit-$LLM_LIMIT/no-osm/YJMob" ;;
  esac
  P2_RESULTS="$PHASE2_ROOT/$dataset"
  export CITY="$P2_CITY" DATA_BASE="$P2_DATA_BASE" DATASET_LABEL="$P2_LABEL"
  export BELIEFMOVE_OUT="$P2_RESULTS" HYBRID_RUN_DIR="$P2_CACHE"
}

phase2_collect_evidence() {
  DATASET="$P2_SOURCE_DATASET" CITY="$P2_CITY" OLLAMA_MODEL="$OLLAMA_MODEL" \
    VALIDATION_LIMIT="$LLM_LIMIT" TEST_LIMIT="$LLM_LIMIT" \
    VALIDATION_FILE="$P2_DATA_BASE/neural_cgm/validation.jsonl" \
    TEST_FILE="$P2_DATA_BASE/neural_cgm/test.jsonl" \
    OUTPUT_DIR="$P2_CACHE" COMPACT_EVIDENCE=1 ./scripts/hybrid_pipeline.sh run
}

phase2_require() {
  [[ -f "$1" ]] || { echo "MISSING: $1" >&2; return 2; }
}

phase2_audit_base() {
  phase2_require "$P2_DATA_BASE/candidate_ids.json"
  phase2_require "$P2_DATA_BASE/getnext/train.csv"
  phase2_require "$P2_DATA_BASE/getnext/val.csv"
  phase2_require "$P2_DATA_BASE/getnext/test.csv"
  phase2_require "$P2_DATA_BASE/neural_cgm/best.pt"
  echo "ready dataset=$P2_LABEL data=$P2_DATA_BASE output=$P2_RESULTS"
}
