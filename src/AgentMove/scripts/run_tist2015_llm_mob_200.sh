#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LLM_BASELINE=llm-mob
exec ./scripts/run_tist2015_llm_only_200.sh "${1:-audit}"
