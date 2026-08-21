#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
command -v "$PYTHON_BOOTSTRAP" >/dev/null || {
  echo "Python is missing. Install it with:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" >&2
  exit 2
}

"$PYTHON_BOOTSTRAP" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")
print("bootstrap_python=" + sys.version.split()[0])
PY

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BOOTSTRAP" -m venv .venv || {
    echo "Unable to create .venv. On Ubuntu install python3-venv:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y python3-venv" >&2
    exit 2
  }
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python - <<'PY'
import json
import numpy
import pandas
import sklearn
import torch
print(json.dumps({
    "python": __import__("sys").version.split()[0],
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scikit_learn": sklearn.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
}, indent=2))
PY

echo "Environment ready: $(pwd)/.venv/bin/python"
echo "Next commands:"
echo "  ./scripts/tist2015_pipeline.sh download"
echo "  ./scripts/tist2015_pipeline.sh prepare"
echo "  ./scripts/tist2015_pipeline.sh train"
