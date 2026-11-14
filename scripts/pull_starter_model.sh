#!/usr/bin/env bash
# Pull one small model from the Ollama library (default: gemma3:270m, ~292 MB).
# Usage: bash scripts/pull_starter_model.sh [tag]
set -euo pipefail
TAG="${1:-gemma3:270m}"
if ! [[ "${TAG}" =~ ^[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?$ ]]; then
  echo "Invalid tag: ${TAG}"; exit 1
fi
command -v ollama >/dev/null 2>&1 || export PATH="${HOME}/.local/bin:${PATH}"
command -v ollama >/dev/null 2>&1 || { echo "Run scripts/install_ollama.sh first."; exit 1; }
OLLAMA_HOST="127.0.0.1:11434" ollama pull "${TAG}"
echo "Done. '${TAG}' now appears under 'Local (Ollama)' in the app's model list."
