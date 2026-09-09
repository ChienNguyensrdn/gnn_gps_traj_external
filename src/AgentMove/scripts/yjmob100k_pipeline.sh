#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-audit}"; PY="${PYTHON_BIN:-.venv/bin/python}"; CITY=YJMob
RAW="${YJMOB_RAW_DIR:-data/dataset_yjmob100k}"
DATASET_NUMBER="${YJMOB_DATASET:-1}"; SOURCE="$RAW/yjmob100k-dataset${DATASET_NUMBER}.csv.gz"
NORMALIZED="${YJMOB_NORMALIZED:-data/input_trajectories/YJMob_filtered.csv}"
BASE="${YJMOB_DATA_BASE:-data/hybrid/YJMob100K/YJMob}"
RESULTS="${YJMOB_RESULTS_ROOT:-results/beliefmove-evo-yjmob100k}"
SEEDS="${RQ_SEEDS:-42 43 44}"; DEVICE="${DEVICE:-auto}"; BATCH_SIZE="${BATCH_SIZE:-128}"
LIMIT="${LLM_LIMIT:-200}"; MODEL="${OLLAMA_MODEL:-qwen2:7b}"; SLUG="${MODEL//[:\/]/-}"
ITERATIONS="${SIGNIFICANCE_ITERATIONS:-10000}"
HYBRID="results/yjmob100k-hybrid/$SLUG/limit-$LIMIT/no-osm/YJMob"
LABEL="YJMob100K-Dataset${DATASET_NUMBER}"
export CITY DATA_BASE="$BASE" BELIEFMOVE_OUT="$RESULTS" DATASET_LABEL="$LABEL"

required=(candidate_ids.json getnext/train.csv getnext/val.csv getnext/test.csv validation.jsonl test.jsonl)
need_python(){ [[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }; }
need_prepared(){ local f; for f in "${required[@]}"; do [[ -f "$BASE/$f" ]] || { echo "Missing prepared file: $BASE/$f" >&2; return 2; }; done; }

audit(){
  local missing=0 f
  [[ -x "$PY" ]] && echo "python=ready ($PY)" || { echo "python=missing ($PY)"; missing=$((missing+1)); }
  for f in "$SOURCE" "$RAW/cell_POIcat.csv.gz" "$RAW/POI_datacategories.csv"; do
    [[ -f "$f" ]] && echo "raw=ready $f" || { echo "raw=missing $f"; missing=$((missing+1)); }
  done
  [[ -f "$NORMALIZED" ]] && echo "normalized=ready $NORMALIZED" || echo "normalized=pending $NORMALIZED"
  for f in "${required[@]}"; do [[ -f "$BASE/$f" ]] && echo "prepared=ready $BASE/$f" || echo "prepared=pending $BASE/$f"; done
  [[ -f "$BASE/neural_cgm/best.pt" ]] && echo "teacher=ready" || echo "teacher=pending"
  echo "dataset=$LABEL users=${YJMOB_MAX_USERS:-1000} seeds=$SEEDS results=$RESULTS"
  (( missing == 0 )) || return 2
}

download(){
  need_python; local files=(dataset1 poi categories)
  [[ "$DATASET_NUMBER" == 2 ]] && files=(dataset2 poi categories)
  "$PY" -m processing.download_yjmob100k --output-dir "$RAW" --files "${files[@]}"
}

prepare(){
  need_python; [[ -f "$SOURCE" ]] || { echo "Missing $SOURCE; run $0 download" >&2; exit 2; }
  if [[ ! -f "$NORMALIZED" || "${FORCE_PREPARE:-0}" == 1 ]]; then
    if [[ "${YJMOB_KEEP_STAYS:-0}" == 1 ]]; then
      "$PY" -m processing.prepare_yjmob100k --input "$SOURCE" --poi "$RAW/cell_POIcat.csv.gz" \
        --categories "$RAW/POI_datacategories.csv" --output "$NORMALIZED" --dataset-name "$LABEL" \
        --max-users "${YJMOB_MAX_USERS:-1000}" --min-observations "${YJMOB_MIN_OBSERVATIONS:-100}" \
        --chunk-size "${YJMOB_CHUNK_SIZE:-1000000}" --seed "${PREPARE_SEED:-42}" --keep-consecutive-stays
    else
      "$PY" -m processing.prepare_yjmob100k --input "$SOURCE" --poi "$RAW/cell_POIcat.csv.gz" \
        --categories "$RAW/POI_datacategories.csv" --output "$NORMALIZED" --dataset-name "$LABEL" \
        --max-users "${YJMOB_MAX_USERS:-1000}" --min-observations "${YJMOB_MIN_OBSERVATIONS:-100}" \
        --chunk-size "${YJMOB_CHUNK_SIZE:-1000000}" --seed "${PREPARE_SEED:-42}"
    fi
  else echo "skip existing normalized data: $NORMALIZED"; fi
  if need_prepared >/dev/null 2>&1 && [[ "${FORCE_PREPARE:-0}" != 1 ]]; then echo "skip prepared data: $BASE"; return 0; fi
  "$PY" -m hybrid.prepare_dataset --dataset yjmob100k --input "$NORMALIZED" --city "$CITY" --output-dir "$BASE" \
    --history-limit "${HISTORY_LIMIT:-40}" --context-limit "${CONTEXT_LIMIT:-6}" --min-points "${MIN_POINTS:-4}"
}

train_teacher(){
  need_python; need_prepared; mkdir -p "$BASE/neural_cgm" "results/logs/yjmob100k"
  local checkpoint="$BASE/neural_cgm/best.pt" split
  if [[ ! -f "$checkpoint" || "${FORCE_TRAIN:-0}" == 1 ]]; then
    "$PY" -m hybrid.neural_cgm train --train-csv "$BASE/getnext/train.csv" --validation-csv "$BASE/getnext/val.csv" \
      --candidate-ids "$BASE/candidate_ids.json" --output "$checkpoint" --epochs "${EPOCHS:-10}" \
      --batch-size "$BATCH_SIZE" --learning-rate "${LEARNING_RATE:-0.001}" --seed "${TEACHER_SEED:-42}" --device "$DEVICE" \
      2>&1 | tee results/logs/yjmob100k/train-teacher.log
  else echo "skip existing $checkpoint"; fi
  for split in validation test; do
    [[ -f "$BASE/neural_cgm/$split.jsonl" && "${FORCE_EXPORT:-0}" != 1 ]] && { echo "skip export $split"; continue; }
    "$PY" -m hybrid.neural_cgm export --checkpoint "$checkpoint" --input "$BASE/$split.jsonl" \
      --getnext-csv "$BASE/getnext/train.csv" "$BASE/getnext/val.csv" "$BASE/getnext/test.csv" \
      --output "$BASE/neural_cgm/${split}_logits.npy"
    "$PY" -m hybrid.cgm_adapter --logits "$BASE/neural_cgm/${split}_logits.npy" \
      --metadata "$BASE/${split}_metadata.jsonl" --candidate-ids "$BASE/candidate_ids.json" \
      --candidate-metadata "$BASE/candidate_metadata.json" --output "$BASE/neural_cgm/$split.jsonl"
  done
}

neural(){
  need_prepared; [[ -f "$BASE/neural_cgm/best.pt" ]] || train_teacher
  local seed variant order
  for seed in $SEEDS; do
    for variant in E0-ce E1-kd E5-dual; do
      SEED="$seed" VARIANT="$variant" ORDER_MODE=correct EPOCHS="${STUDENT_EPOCHS:-10}" BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh train-student
      SEED="$seed" VARIANT="$variant" ORDER_MODE=correct EVALUATION_RQ=RQ4 BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh evaluate-student
    done
    for order in reverse random; do
      SEED="$seed" VARIANT=E5-dual ORDER_MODE="$order" EPOCHS="${STUDENT_EPOCHS:-10}" BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh train-student
      SEED="$seed" VARIANT=E5-dual ORDER_MODE="$order" EVALUATION_RQ=RQ5 BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh evaluate-student
    done
  done
}

frozen_order(){
  need_python; need_prepared; local seed input_mode checkpoint output_dir
  for seed in $SEEDS; do
    checkpoint="$RESULTS/artifacts/full/$CITY/E5-dual/correct/seed-$seed/best.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing frozen E5 checkpoint: $checkpoint" >&2; exit 2; }
    for input_mode in correct reverse random; do
      output_dir="$RESULTS/artifacts/full/$CITY/E5-dual/frozen-$input_mode/seed-$seed"
      [[ -f "$output_dir/test.metrics.json" && -f "$output_dir/test.predictions.npz" && "${FORCE:-0}" != 1 ]] && { echo "skip frozen-$input_mode seed=$seed"; continue; }
      mkdir -p "$output_dir"
      "$PY" -m hybrid.evaluate_student --checkpoint "$checkpoint" --test-csv "$BASE/getnext/test.csv" \
        --output "$output_dir/test.metrics.json" --predictions-output "$output_dir/test.predictions.npz" \
        --batch-size "${EVAL_BATCH_SIZE:-256}" --device "$DEVICE" --seed "$seed" \
        --order-mode correct --input-order-mode "$input_mode"
    done
  done
}

bayesian(){
  need_python; need_prepared; local seed checkpoint output
  for seed in $SEEDS; do
    checkpoint="$RESULTS/artifacts/full/$CITY/E5-dual/correct/seed-$seed/best.pt"
    output="$RESULTS/artifacts/full/$CITY/E5-dual/correct/seed-$seed/rq7"
    [[ -f "$checkpoint" ]] || { echo "Missing E5 checkpoint: $checkpoint" >&2; exit 2; }
    [[ -f "$output/rq7.metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $output/rq7.metrics.json"; continue; }
    "$PY" -m hybrid.rq7_belief_memory --checkpoint "$checkpoint" --train-csv "$BASE/getnext/train.csv" \
      --validation-csv "$BASE/getnext/val.csv" --test-csv "$BASE/getnext/test.csv" --output-dir "$output" \
      --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$seed"
  done
}

llm_bounded(){
  need_prepared
  DATASET=yjmob100k CITY="$CITY" OLLAMA_MODEL="$MODEL" VALIDATION_LIMIT="$LIMIT" TEST_LIMIT="$LIMIT" \
    VALIDATION_FILE="$BASE/neural_cgm/validation.jsonl" TEST_FILE="$BASE/neural_cgm/test.jsonl" \
    OUTPUT_DIR="$HYBRID" COMPACT_EVIDENCE=1 ./scripts/hybrid_pipeline.sh run
}

summarize(){
  need_python
  if [[ "$1" == status ]]; then
    "$PY" -m hybrid.www2019_summary --root "$RESULTS/artifacts/full/$CITY" \
      --hybrid-metrics "$HYBRID/full/metrics.json" --seeds $SEEDS --iterations "$ITERATIONS" \
      --dataset-label "$LABEL" --report-title "YJMob100K — Cross-dataset validation" \
      --output "$RESULTS/aggregated/yjmob100k_summary.json" --markdown ../../ideas/results_yjmob100k_generated.md --allow-incomplete
  else
    "$PY" -m hybrid.www2019_summary --root "$RESULTS/artifacts/full/$CITY" \
      --hybrid-metrics "$HYBRID/full/metrics.json" --seeds $SEEDS --iterations "$ITERATIONS" \
      --dataset-label "$LABEL" --report-title "YJMob100K — Cross-dataset validation" \
      --output "$RESULTS/aggregated/yjmob100k_summary.json" --markdown ../../ideas/results_yjmob100k_generated.md
  fi
}

case "$ACTION" in
  audit) audit;; download) download;; prepare) prepare;; train-teacher) train_teacher;; neural) neural;;
  frozen-order) frozen_order;; bayesian) bayesian;; llm-bounded) llm_bounded;;
  status) summarize status;; aggregate) summarize aggregate;;
  smoke) need_prepared; [[ -f "$BASE/neural_cgm/best.pt" ]] || train_teacher; SEED=42 VARIANT=E5-dual ORDER_MODE=correct EPOCHS=1 TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-1000}" VALIDATION_LIMIT="${SMOKE_VALIDATION_LIMIT:-200}" RUN_TAG=yjmob100k-smoke BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh train-student;;
  all) prepare; train_teacher; neural; frozen_order; bayesian; llm_bounded; summarize aggregate;;
  *) echo "Usage: $0 <audit|download|prepare|train-teacher|smoke|neural|frozen-order|bayesian|llm-bounded|status|aggregate|all>" >&2; exit 2;;
esac
