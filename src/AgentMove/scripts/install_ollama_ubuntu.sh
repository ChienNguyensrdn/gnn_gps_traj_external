#!/usr/bin/env bash
set -euo pipefail

PRIMARY_MODEL="${PRIMARY_MODEL:-qwen2:7b}"
# Llama 3.1 is an open-weight second backbone for robustness checks; it is not an OpenAI model.
SECOND_MODEL="${SECOND_MODEL:-llama3.1:8b}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer targets Ubuntu/Linux." >&2; exit 2
fi
command -v curl >/dev/null || { echo "curl is required: sudo apt-get install -y curl" >&2; exit 2; }

if ! command -v ollama >/dev/null; then
  echo "Installing Ollama from the official installer..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if command -v systemctl >/dev/null; then
  sudo systemctl enable --now ollama
else
  nohup ollama serve >"${OLLAMA_LOG_FILE:-/tmp/ollama.log}" 2>&1 &
fi

for _ in {1..30}; do
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null && break
  sleep 1
done
curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null || { echo "Ollama failed to start" >&2; exit 1; }

ollama pull "$PRIMARY_MODEL"
if [[ "${INSTALL_SECOND_MODEL:-1}" == "1" ]]; then
  ollama pull "$SECOND_MODEL"
fi
echo "Ready: $PRIMARY_MODEL and $SECOND_MODEL at http://127.0.0.1:11434"
echo "For a proprietary OpenAI API comparison, use an OpenAI model through its API; it cannot be pulled by Ollama."
