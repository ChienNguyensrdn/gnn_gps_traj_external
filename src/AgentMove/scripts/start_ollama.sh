#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
LOG_FILE="${OLLAMA_LOG_FILE:-/tmp/hybrid-ollama.log}"

if curl -fsS --max-time 2 "$OLLAMA_HOST_URL/api/version" >/dev/null; then
  echo "Ollama is already running at $OLLAMA_HOST_URL"
  exit 0
fi

if [[ "$(uname -s)" == "Darwin" && -d /Applications/Ollama.app ]]; then
  echo "Opening /Applications/Ollama.app ..."
  open -a Ollama
elif command -v systemctl >/dev/null && systemctl cat ollama.service >/dev/null 2>&1; then
  echo "Starting the system Ollama service..."
  sudo systemctl start ollama
else
  command -v ollama >/dev/null || { echo "ollama command not found" >&2; exit 1; }
  echo "No systemd Ollama unit found; starting a user Ollama server; log: $LOG_FILE"
  nohup ollama serve >"$LOG_FILE" 2>&1 &
fi

for _ in {1..30}; do
  if curl -fsS --max-time 2 "$OLLAMA_HOST_URL/api/version" >/dev/null; then
    echo "Ollama is ready at $OLLAMA_HOST_URL"
    echo "Local API key: none required (use OLLAMA_API_KEY=ollama only for OpenAI SDK compatibility)"
    exit 0
  fi
  sleep 1
done

echo "Ollama did not become ready." >&2
if command -v systemctl >/dev/null && systemctl cat ollama.service >/dev/null 2>&1; then
  echo "Check: sudo systemctl status ollama --no-pager" >&2
  echo "Check: sudo journalctl -u ollama -n 100 --no-pager" >&2
else
  echo "Check $LOG_FILE or open Ollama.app manually." >&2
fi
exit 1
