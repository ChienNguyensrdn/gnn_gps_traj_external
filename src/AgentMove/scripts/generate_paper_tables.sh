#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python -m hybrid.paper_tables \
  --agentmove-root . \
  --paper-dir ../../paper \
  --hybrid-results results/hybrid/shanghai-neural-cgm-50-seed42/qwen2-7b \
  --llm-results results/llm-only/shanghai-50-seed42/qwen2-7b

echo "Generated LaTeX rows: ../../paper/generated/"
