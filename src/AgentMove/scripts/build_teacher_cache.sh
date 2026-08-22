#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TYPE="${TYPE:-llm}"
INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-results/beliefmove-evo/teacher-cache/$TYPE.jsonl}"
VERSION="${VERSION:-v1}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing $PYTHON_BIN" >&2; exit 2; }
[[ -n "$INPUT" && -f "$INPUT" ]] || { echo "Set INPUT to an existing JSONL file" >&2; exit 2; }

case "$TYPE" in
  llm) fields=(habit_score semantic_score valid raw) ;;
  quantitative) fields=(logits hidden_states temporal_states label) ;;
  *) echo "TYPE must be llm or quantitative" >&2; exit 2 ;;
esac

"$PYTHON_BIN" -m hybrid.teacher_cache --input "$INPUT" --output "$OUTPUT" \
  --namespace "$TYPE" --version "$VERSION" --key-field "${KEY_FIELD:-query_id}" --value-fields "${fields[@]}"
