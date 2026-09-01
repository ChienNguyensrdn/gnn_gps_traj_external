#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
A="${1:-audit}"; PY="${PYTHON_BIN:-.venv/bin/python}"; CITY="${CITY:-Tokyo}"; SEED="${SEED:-42}"; BASE="data/hybrid/TIST2015/$CITY"; ROOT="results/beliefmove-evo/artifacts/full/$CITY/rq2"; TROOT="results/beliefmove-evo/artifacts/full/$CITY/rq10/teachers/gru"
req(){ [[ -f "$1" ]]||{ echo "Missing required file: $1" >&2; exit 2; }; }
audit(){ [[ -x "$PY" ]]||exit 2; req "$BASE/getnext/train.csv"; req "$BASE/getnext/test.csv"; req "$TROOT/seed-$SEED/best.pt"; echo "RQ2 city=$CITY seed=$SEED fit=train eval=test output=$ROOT"; }
evaluate(){ audit; [[ -f "$ROOT/quantitative-teacher/seed-$SEED/rq2.metrics.json" && "${FORCE:-0}" != 1 ]]&&{ echo "skip seed $SEED"; return; }; "$PY" -m hybrid.rq2_data_only --checkpoint "$TROOT/seed-$SEED/best.pt" --train-csv "$BASE/getnext/train.csv" --test-csv "$BASE/getnext/test.csv" --output-root "$ROOT" --seed "$SEED" --batch-size "${BATCH_SIZE:-256}" --device "${DEVICE:-auto}"; }
run_seeds(){ for s in ${RQ2_SEEDS:-42 43 44}; do CITY="$CITY" SEED="$s" "$0" evaluate; done; }
status(){ local miss=0 p v s; for s in ${RQ2_SEEDS:-42 43 44}; do for v in unigram markov-bigram bn-data-only dbn-data-only quantitative-teacher; do p="$ROOT/$v/seed-$s/rq2.metrics.json"; [[ -f "$p" ]]&&echo "ready $p"||{ echo "missing $p"; miss=$((miss+1)); }; done; done; ((miss==0)); }
aggregate(){ status||exit 2; "$PY" -m hybrid.rq2_aggregate --root "$ROOT" --seeds ${RQ2_SEEDS:-42 43 44} --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" --output results/beliefmove-evo/aggregated/rq2_summary.json --markdown ../../ideas/results_rq2.md; }
case "$A" in audit) audit;; evaluate) evaluate;; run-seeds) run_seeds;; status) status;; aggregate) aggregate;; *) echo "Usage: $0 <audit|evaluate|run-seeds|status|aggregate>"; exit 2;; esac
