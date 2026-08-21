#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./scripts/run_llm_only_shanghai_50pct.sh
