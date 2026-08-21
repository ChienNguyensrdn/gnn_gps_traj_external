#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$SCRIPT_DIR/lib/tist2015_common.sh"
if [[ -f "$COMMON" ]]; then
  # shellcheck source=lib/tist2015_common.sh
  source "$COMMON"
else
  # Keep this runner portable when only the experiment scripts are copied to
  # another machine/repository.
  readonly TIST2015_CITIES=(
    Tokyo Nairobi NewYork Sydney CapeTown Paris
    Beijing Mumbai SanFrancisco London SaoPaulo Moscow
  )
  tist2015_agentmove_root() { cd "$SCRIPT_DIR/.." && pwd; }
  tist2015_is_city() {
    local requested="$1" city
    for city in "${TIST2015_CITIES[@]}"; do
      [[ "$city" == "$requested" ]] && return 0
    done
    return 1
  }
  tist2015_model_slug() {
    local slug="${1//\//-}"
    printf '%s\n' "${slug//:/-}"
  }
fi

ROOT="$(tist2015_agentmove_root)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2:7b}"
NOMINATIM_URL="${NOMINATIM_URL:-http://127.0.0.1:8080}"
NOMINATIM_DELAY="${NOMINATIM_DELAY:-0}"
OSM_MIN_COVERAGE="${OSM_MIN_COVERAGE:-0.90}"
NORMALIZED_ROOT="${NORMALIZED_ROOT:-data/input_trajectories}"
ENRICHED_ROOT="${ENRICHED_ROOT:-data/input_trajectories_clean}"
OSM_CACHE_ROOT="${OSM_CACHE_ROOT:-data/osm_cache/tist2015}"
DATA_ROOT="${DATA_ROOT:-data/hybrid/TIST2015}"
MODEL_SLUG="$(tist2015_model_slug "$OLLAMA_MODEL")"
RESULT_ROOT="${RESULT_ROOT:-results/tist2015-hybrid/$MODEL_SLUG/limit-$QUERY_LIMIT/full-osm}"

usage() {
  cat <<EOF
Usage:
  $0 audit
  $0 enrich <city|pending>
  $0 run <city|pending>
  $0 aggregate
  $0 city <city>        # enrich, coverage gate, then run one city

Environment:
  NOMINATIM_URL=http://127.0.0.1:8080   local Nominatim (required)
  QUERY_LIMIT=200 OLLAMA_MODEL=qwen2:7b

Cities: ${TIST2015_CITIES[*]}
Ollama is always http://127.0.0.1:11434/v1.
EOF
}

require_city() {
  tist2015_is_city "$1" || { echo "Invalid city: $1" >&2; exit 2; }
}

coverage() {
  "$PYTHON_BIN" - "$DATA_ROOT/$1/candidate_metadata.json" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
n = len(p)
c = sum(bool((v.get("address") or "").strip()) for v in p.values())
print(f"{c / n if n else 0:.6f}")
PY
}

coverage_ok() {
  "$PYTHON_BIN" - "$(coverage "$1")" "$OSM_MIN_COVERAGE" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
}

enrich_city() {
  local city="$1" input output cache metadata
  require_city "$city"
  input="$NORMALIZED_ROOT/${city}_filtered.csv"
  output="$ENRICHED_ROOT/${city}_filtered.csv"
  cache="$OSM_CACHE_ROOT/${city}.jsonl"
  metadata="$DATA_ROOT/$city/candidate_metadata.json"
  for file in "$PYTHON_BIN" "$input" "$metadata"; do
    [[ -f "$file" ]] || { echo "Missing required file: $file" >&2; exit 2; }
  done
  if [[ "$NOMINATIM_URL" == https://nominatim.openstreetmap.org* ]]; then
    echo "Refusing the public Nominatim endpoint for the ~232k-POI TIST run." >&2
    echo "Start a local Nominatim service and set NOMINATIM_URL (default port 8080)." >&2
    exit 2
  fi
  echo "[$city] OSM enrichment via $NOMINATIM_URL (resumable cache: $cache)"
  "$PYTHON_BIN" -m hybrid.enrich_osm \
    --input "$input" --output "$output" --cache "$cache" \
    --base-url "$NOMINATIM_URL" --delay-seconds "$NOMINATIM_DELAY"
  "$PYTHON_BIN" -m hybrid.refresh_osm_metadata --csv "$output" --metadata "$metadata"
  echo "[$city] coverage=$(coverage "$city") threshold=$OSM_MIN_COVERAGE"
}

run_city() {
  local city="$1"
  require_city "$city"
  if ! coverage_ok "$city"; then
    echo "[$city] OSM coverage $(coverage "$city") is below $OSM_MIN_COVERAGE; not running Ours (full)." >&2
    exit 3
  fi
  QUERY_LIMIT="$QUERY_LIMIT" OLLAMA_MODEL="$OLLAMA_MODEL" \
    OUTPUT_ROOT="$RESULT_ROOT" ./scripts/run_tist2015_city_200.sh "$city"
}

pending_enrich() {
  local city
  for city in "${TIST2015_CITIES[@]}"; do
    coverage_ok "$city" || enrich_city "$city"
  done
}

pending_run() {
  local city metrics
  for city in "${TIST2015_CITIES[@]}"; do
    metrics="$RESULT_ROOT/$city/full/metrics.json"
    if [[ -f "$metrics" ]]; then
      echo "[$city] done: $metrics"
    else
      run_city "$city"
    fi
  done
}

audit() {
  local city status metrics
  printf '%-14s %-10s %s\n' city coverage status
  for city in "${TIST2015_CITIES[@]}"; do
    if [[ -f "$DATA_ROOT/$city/candidate_metadata.json" ]]; then
      metrics="$RESULT_ROOT/$city/full/metrics.json"
      status=pending
      [[ -f "$metrics" ]] && status=done
      printf '%-14s %-10s %s\n' "$city" "$(coverage "$city")" "$status"
    else
      printf '%-14s %-10s %s\n' "$city" missing missing-metadata
    fi
  done
}

command="${1:-audit}"
target="${2:-}"
case "$command" in
  audit) audit ;;
  enrich)
    [[ -n "$target" ]] || { usage; exit 2; }
    [[ "$target" == pending ]] && pending_enrich || enrich_city "$target"
    ;;
  run)
    [[ -n "$target" ]] || { usage; exit 2; }
    [[ "$target" == pending ]] && pending_run || run_city "$target"
    ;;
  city)
    [[ -n "$target" ]] || { usage; exit 2; }
    enrich_city "$target"
    run_city "$target"
    ;;
  aggregate)
    "$PYTHON_BIN" -m hybrid.tist2015_full_osm_aggregate \
      --root "$RESULT_ROOT" \
      --output "results/tist2015-hybrid/$MODEL_SLUG/limit-$QUERY_LIMIT/full-osm-summary.json" \
      --query-limit "$QUERY_LIMIT"
    ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
