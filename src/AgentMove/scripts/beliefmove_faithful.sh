#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./phase3_beliefmove/scripts/run.sh "$@"
