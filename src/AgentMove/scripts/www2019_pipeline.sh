#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-audit}"; PY="${PYTHON_BIN:-.venv/bin/python}"; CITY=Shanghai
BASE="${WWW2019_DATA_BASE:-data/hybrid/WWW2019/Shanghai}"
RESULTS="${WWW2019_RESULTS_ROOT:-results/beliefmove-evo-www2019}"
SEEDS="${RQ_SEEDS:-42 43 44}"; DEVICE="${DEVICE:-auto}"; BATCH_SIZE="${BATCH_SIZE:-128}"
LIMIT="${LLM_LIMIT:-200}"; MODEL="${OLLAMA_MODEL:-qwen2:7b}"; SLUG="${MODEL//[:\/]/-}"
ITERATIONS="${SIGNIFICANCE_ITERATIONS:-10000}"
HYBRID="results/www2019-hybrid/$SLUG/limit-$LIMIT/no-osm/Shanghai"
export CITY DATA_BASE="$BASE" BELIEFMOVE_OUT="$RESULTS" DATASET_LABEL="WWW2019-Shanghai-ISP"

required=(candidate_ids.json getnext/train.csv getnext/val.csv getnext/test.csv validation.jsonl test.jsonl)
need_python(){ [[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }; }
need_prepared(){ local f; for f in "${required[@]}"; do [[ -f "$BASE/$f" ]] || { echo "Missing prepared file: $BASE/$f" >&2; return 2; }; done; }

audit(){
  local f missing=0
  [[ -x "$PY" ]] && echo "python=ready ($PY)" || { echo "python=missing ($PY)"; missing=$((missing+1)); }
  for f in data/dataset_www2019/isp data/dataset_www2019/weibo data/dataset_www2019/poi.txt; do
    [[ -f "$f" ]] && echo "raw=ready $f" || { echo "raw=missing $f"; missing=$((missing+1)); }
  done
  for f in "${required[@]}"; do
    [[ -f "$BASE/$f" ]] && echo "prepared=ready $BASE/$f" || { echo "prepared=missing $BASE/$f"; missing=$((missing+1)); }
  done
  [[ -f "$BASE/neural_cgm/best.pt" ]] && echo "teacher=ready" || echo "teacher=pending"
  echo "dataset=WWW2019-Shanghai-ISP seeds=$SEEDS results=$RESULTS"
  (( missing == 0 )) || return 2
}

download(){ need_python; "$PY" -m processing.download --download_mode data --data_name www2019; }

prepare(){
  need_python
  if need_prepared >/dev/null 2>&1 && [[ "${FORCE_PREPARE:-0}" != 1 ]]; then
    echo "skip prepared WWW2019 data: $BASE"
    return 0
  fi
  DATASET=isp CITY=Shanghai PYTHON_BIN="$PY" ./scripts/hybrid_pipeline.sh extract
  local input="data/input_trajectories/Shanghai_filtered.csv"
  [[ -f "$input" ]] || { echo "Missing normalized input: $input" >&2; exit 2; }
  "$PY" -m hybrid.prepare_dataset --dataset isp --input "$input" --city Shanghai --output-dir "$BASE" \
    --history-limit "${HISTORY_LIMIT:-40}" --context-limit "${CONTEXT_LIMIT:-6}"
}

train_teacher(){
  need_python; need_prepared; mkdir -p "$BASE/neural_cgm" "results/logs/www2019"
  local checkpoint="$BASE/neural_cgm/best.pt" split
  if [[ ! -f "$checkpoint" || "${FORCE_TRAIN:-0}" == 1 ]]; then
    "$PY" -m hybrid.neural_cgm train --train-csv "$BASE/getnext/train.csv" --validation-csv "$BASE/getnext/val.csv" \
      --candidate-ids "$BASE/candidate_ids.json" --output "$checkpoint" --epochs "${EPOCHS:-10}" \
      --batch-size "$BATCH_SIZE" --learning-rate "${LEARNING_RATE:-0.001}" --seed "${TEACHER_SEED:-42}" --device "$DEVICE" \
      2>&1 | tee results/logs/www2019/train-teacher.log
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

bayesian(){
  local seed checkpoint output
  for seed in $SEEDS; do
    checkpoint="$RESULTS/artifacts/full/Shanghai/E5-dual/correct/seed-$seed/best.pt"
    output="$RESULTS/artifacts/full/Shanghai/E5-dual/correct/seed-$seed/rq7"
    [[ -f "$checkpoint" ]] || { echo "Missing E5 checkpoint: $checkpoint" >&2; exit 2; }
    [[ -f "$output/rq7.metrics.json" && "${FORCE:-0}" != 1 ]] && { echo "skip existing $output/rq7.metrics.json"; continue; }
    "$PY" -m hybrid.rq7_belief_memory --checkpoint "$checkpoint" --train-csv "$BASE/getnext/train.csv" \
      --validation-csv "$BASE/getnext/val.csv" --test-csv "$BASE/getnext/test.csv" --output-dir "$output" \
      --batch-size "$BATCH_SIZE" --device "$DEVICE" --seed "$seed"
  done
}

llm_bounded(){
  need_prepared
  DATASET=isp CITY=Shanghai OLLAMA_MODEL="$MODEL" VALIDATION_LIMIT="$LIMIT" TEST_LIMIT="$LIMIT" \
    VALIDATION_FILE="$BASE/neural_cgm/validation.jsonl" TEST_FILE="$BASE/neural_cgm/test.jsonl" \
    OUTPUT_DIR="$HYBRID" COMPACT_EVIDENCE=1 ./scripts/hybrid_pipeline.sh run
}

status(){
  need_python
  "$PY" -m hybrid.www2019_summary --root "$RESULTS/artifacts/full/Shanghai" \
    --hybrid-metrics "$HYBRID/full/metrics.json" --seeds $SEEDS \
    --iterations "$ITERATIONS" --output "$RESULTS/aggregated/www2019_summary.json" \
    --markdown ../../ideas/results_www2019_generated.md --allow-incomplete
}
aggregate(){
  need_python
  "$PY" -m hybrid.www2019_summary --root "$RESULTS/artifacts/full/Shanghai" \
    --hybrid-metrics "$HYBRID/full/metrics.json" --seeds $SEEDS \
    --iterations "$ITERATIONS" --output "$RESULTS/aggregated/www2019_summary.json" \
    --markdown ../../ideas/results_www2019_generated.md
}

case "$ACTION" in
  audit) audit;; download) download;; prepare) prepare;; train-teacher) train_teacher;; neural) neural;;
  bayesian) bayesian;; llm-bounded) llm_bounded;; status) status;; aggregate) aggregate;;
  smoke)
    need_prepared
    [[ -f "$BASE/neural_cgm/best.pt" ]] || train_teacher
    SEED=42 VARIANT=E5-dual ORDER_MODE=correct EPOCHS=1 \
      TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-1000}" VALIDATION_LIMIT="${SMOKE_VALIDATION_LIMIT:-200}" \
      RUN_TAG=www2019-smoke BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" ./scripts/beliefmove_evo.sh train-student
    ;;
  all) prepare; train_teacher; neural; bayesian; llm_bounded; aggregate;;
  *) echo "Usage: $0 <audit|download|prepare|train-teacher|smoke|neural|bayesian|llm-bounded|status|aggregate|all>" >&2; exit 2;;
esac
