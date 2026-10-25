#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source phase2/scripts/common.sh

ACTION="${1:-audit}"
PAPER_DIR="${PAPER_DIR:-/Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/IEEE-Transactions_Latex}"
TABLE_FILE="$PAPER_DIR/sections/generated_phase2_tables.tex"

case "$ACTION" in
  aggregate)
    ./phase2/scripts/aggregate.sh all
    LLM_LIMIT="${LLM_LIMIT:-200}" ./phase2/scripts/aggregate_llm.sh all
    ;;
  audit)
    .venv/bin/python -m phase2.publication_paper \
      --root "$PHASE2_ROOT/aggregated" --limit "${LLM_LIMIT:-200}" --audit-only
    ;;
  export)
    .venv/bin/python -m phase2.publication_paper \
      --root "$PHASE2_ROOT/aggregated" --limit "${LLM_LIMIT:-200}" --output "$TABLE_FILE"
    ;;
  build)
    [[ -f "$PAPER_DIR/Hybrid_gps_traj.tex" ]] || { echo "Missing paper: $PAPER_DIR/Hybrid_gps_traj.tex" >&2; exit 2; }
    (cd "$PAPER_DIR" && latexmk -pdf -interaction=nonstopmode -halt-on-error Hybrid_gps_traj.tex)
    ;;
  all)
    "$0" aggregate
    "$0" audit
    "$0" export
    "$0" build
    ;;
  *)
    echo "Usage: $0 <aggregate|audit|export|build|all>" >&2
    exit 2
    ;;
esac
