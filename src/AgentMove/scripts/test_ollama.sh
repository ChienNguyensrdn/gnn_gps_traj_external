#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL="${1:-qwen2:7b}"
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"

curl -fsS --max-time 3 "${BASE_URL%/v1}/api/version" >/dev/null || {
  echo "Ollama is not running. Run ./scripts/start_ollama.sh first." >&2
  exit 1
}

echo "Installed models:"
curl -fsS "${BASE_URL%/v1}/api/tags" | .venv/bin/python -m json.tool

if ! curl -fsS "${BASE_URL%/v1}/api/tags" | .venv/bin/python -c '
import json, sys
requested = sys.argv[1]
payload = json.load(sys.stdin)
names = {row.get("name") for row in payload.get("models", [])}
raise SystemExit(0 if requested in names else 1)
' "$MODEL"; then
  echo "Model '$MODEL' is not installed in the Ollama server at ${BASE_URL%/v1}." >&2
  echo "Install it into this exact server with:" >&2
  echo "  OLLAMA_HOST=127.0.0.1:11434 ollama pull '$MODEL'" >&2
  exit 2
fi

OLLAMA_BASE_URL="$BASE_URL" OLLAMA_API_KEY=ollama .venv/bin/python - "$MODEL" <<'PY'
import os
import sys
from openai import OpenAI

model = sys.argv[1]
client = OpenAI(base_url=os.environ["OLLAMA_BASE_URL"], api_key=os.environ["OLLAMA_API_KEY"])
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    temperature=0,
    max_tokens=10,
)
print("response:", response.choices[0].message.content)
PY
