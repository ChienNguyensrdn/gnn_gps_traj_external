#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh

ACTION="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DIRECT_TOP_K="${DIRECT_TOP_K:-10}"
DIRECT_TOP_M="${DIRECT_TOP_M:-5}"
DIRECT_WORLD_MODE="${DIRECT_WORLD_MODE:-internal_only}"
DIRECT_MAX_TOKENS="${DIRECT_MAX_TOKENS:-512}"
DIRECT_MAX_INPUT_TOKENS="${DIRECT_MAX_INPUT_TOKENS:-2000}"
DIRECT_RANDOM_SEEDS="${DIRECT_RANDOM_SEEDS:-$(seq 42 91)}"
DIRECT_EXPECTED_RANDOM_RUNS="${DIRECT_EXPECTED_RANDOM_RUNS:-50}"

require_file() { [[ -f "$1" ]] || { echo "MISSING: $1" >&2; return 2; }; }

audit_response_bank() {
  require_file "$TEMPERATURE_FILE"
  for cache in "$VALIDATION_CACHE" "$TEST_CACHE"; do
    require_file "$cache/protocol.lock.json"
    require_file "$cache/predictions.jsonl"
    require_file "$cache/predictions.npz"
    require_file "$cache/requests.jsonl"
    require_file "$cache/metrics.json"
  done
}

audit_routing_replay() {
  local seed_count random_seed policy
  seed_count="$(printf '%s\n' $DIRECT_RANDOM_SEEDS | sed '/^$/d' | sort -nu | wc -l | tr -d ' ')"
  if [[ "$seed_count" -ne "$DIRECT_EXPECTED_RANDOM_RUNS" ]]; then
    echo "Expected $DIRECT_EXPECTED_RANDOM_RUNS unique routing seeds, found $seed_count in DIRECT_RANDOM_SEEDS." >&2
    return 2
  fi
  for random_seed in $DIRECT_RANDOM_SEEDS; do
    require_file "$ROUTING_ROOT/seed-$random_seed/rq8.metrics.json"
    require_file "$ROUTING_ROOT/seed-$random_seed/routing_diagnostics.test.npz"
    for policy in never always entropy margin random-budget-matched; do
      require_file "$ROUTING_ROOT/seed-$random_seed/$policy.test.predictions.npz"
    done
  done
  echo "counterfactual_routing_replay=ready random_runs=$seed_count"
}

if [[ "$ACTION" == "aggregate" ]]; then
  "$PYTHON_BIN" -m phase2.aggregate_direct_query_llm \
    --root "$PHASE2_ROOT" --model-slug "$MODEL_SLUG" --limit "$LLM_LIMIT" \
    --cities $TIST_CITIES --seeds $DIRECT_RANDOM_SEEDS \
    --iterations "${SIGNIFICANCE_ITERATIONS:-10000}" "${@:2}"
  exit $?
fi

MODEL_DIGEST="${OLLAMA_MODEL_DIGEST:-}"
if [[ "$ACTION" == "collect" ]]; then
  ./scripts/start_ollama.sh
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}" \
    ./scripts/test_ollama.sh "$OLLAMA_MODEL"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
  export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
  if [[ -z "$MODEL_DIGEST" ]]; then
    OLLAMA_NATIVE_URL="${OLLAMA_BASE_URL%/}"
    OLLAMA_NATIVE_URL="${OLLAMA_NATIVE_URL%/v1}"
    MODEL_DIGEST="$(curl -fsS "$OLLAMA_NATIVE_URL/api/tags" | "$PYTHON_BIN" -c '
import json, sys
model = sys.argv[1]
rows = json.load(sys.stdin).get("models", [])
match = next((row for row in rows if row.get("name") == model or row.get("model") == model), None)
if not match:
    raise SystemExit(f"model not found in Ollama tags: {model}")
print(match.get("digest") or "unknown")
' "$OLLAMA_MODEL")"
  fi
fi

for dataset in $PHASE2_DATASETS; do
  for unit in $(phase2_units "$dataset"); do
    phase2_configure "$dataset" "$unit"
    phase2_audit_base
    DIRECT_ROOT="$P2_RESULTS/artifacts/full/$P2_CITY/direct-query-llm/$MODEL_SLUG/limit-$LLM_LIMIT"
    TEMPERATURE_FILE="$DIRECT_ROOT/stage1_temperature.json"
    VALIDATION_CACHE="$DIRECT_ROOT/always-cache/validation"
    TEST_CACHE="$DIRECT_ROOT/always-cache/test"
    ROUTING_ROOT="$DIRECT_ROOT/routing"

    case "$ACTION" in
      audit-cache)
        echo "dataset=$P2_LABEL direct_root=$DIRECT_ROOT"
        echo "audit_scope=always-response-bank-only routing_outputs=not-checked"
        audit_response_bank
        echo "collection_cost=one-physical-request-per-uncached-validation/test-query"
        ;;
      audit)
        echo "dataset=$P2_LABEL direct_root=$DIRECT_ROOT"
        echo "audit_scope=full-response-bank-and-counterfactual-routing-replay"
        audit_response_bank
        audit_routing_replay
        echo "collection_cost=one-physical-request-per-uncached-validation/test-query"
        echo "reported_policy_calls=simulated-deployment-budget-not-collection-calls"
        if [[ "$DIRECT_WORLD_MODE" == "internal_only" ]]; then
          echo "scope=bounded-transductive internal_only_metadata=category,latitude,longitude"
        fi
        ;;
      prepare)
        "$PYTHON_BIN" -m hybrid.direct_query_llm fit-temperature \
          --validation "$P2_DATA_BASE/neural_cgm/validation.jsonl" \
          --output "$TEMPERATURE_FILE" --limit "$LLM_LIMIT"
        ;;
      collect)
        "$PYTHON_BIN" -m hybrid.direct_query_llm fit-temperature \
          --validation "$P2_DATA_BASE/neural_cgm/validation.jsonl" \
          --output "$TEMPERATURE_FILE" --limit "$LLM_LIMIT"
        for split in validation test; do
          "$PYTHON_BIN" -m hybrid.direct_query_llm collect \
            --input "$P2_DATA_BASE/neural_cgm/$split.jsonl" \
            --temperature-file "$TEMPERATURE_FILE" \
            --output-dir "$DIRECT_ROOT/always-cache/$split" --split "$split" \
            --model-name "$OLLAMA_MODEL" --model-digest "$MODEL_DIGEST" --platform Ollama \
            --limit "$LLM_LIMIT" --top-k "$DIRECT_TOP_K" --top-m "$DIRECT_TOP_M" \
            --world-mode "$DIRECT_WORLD_MODE" --max-tokens "$DIRECT_MAX_TOKENS" \
            --max-input-tokens "$DIRECT_MAX_INPUT_TOKENS"
        done
        ;;
      route)
        audit_response_bank
        echo "routing_mode=counterfactual-deployment-replay-from-always-response-bank"
        echo "physical_requests_during_routing=0"
        echo "collection_cost=already-paid-one-request-per-uncached-validation/test-query"
        echo "reported_policy_calls=simulated-deployment-budget-not-collection-calls"
        if [[ "$DIRECT_WORLD_MODE" == "internal_only" ]]; then
          echo "scope=bounded-transductive internal_only_metadata=category,latitude,longitude"
        fi
        "$PYTHON_BIN" -m hybrid.rq8_routing \
          --validation "$P2_DATA_BASE/neural_cgm/validation.jsonl" \
          --test "$P2_DATA_BASE/neural_cgm/test.jsonl" \
          --validation-always "$VALIDATION_CACHE/predictions.jsonl" \
          --test-always "$TEST_CACHE/predictions.jsonl" \
          --calibration "$TEMPERATURE_FILE" \
          --output-root "$ROUTING_ROOT" --seeds $DIRECT_RANDOM_SEEDS \
          --limit "$LLM_LIMIT"
        ;;
      status)
        "$PYTHON_BIN" -m hybrid.direct_query_llm status \
          --input "$P2_DATA_BASE/neural_cgm/validation.jsonl" \
          --output-dir "$VALIDATION_CACHE" --split validation --limit "$LLM_LIMIT"
        "$PYTHON_BIN" -m hybrid.direct_query_llm status \
          --input "$P2_DATA_BASE/neural_cgm/test.jsonl" \
          --output-dir "$TEST_CACHE" --split test --limit "$LLM_LIMIT"
        ;;
      *)
        echo "Usage: $0 <audit-cache|audit|prepare|collect|route|status|aggregate>" >&2
        exit 2
        ;;
    esac
  done
done

echo "Direct query-time LLM action=$ACTION complete"
