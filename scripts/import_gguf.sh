#!/usr/bin/env bash
# Register a local .gguf file (e.g. your own fine-tune) as an Ollama model.
#
# Usage: bash scripts/import_gguf.sh path/to/model.gguf my-model [--template auto|chatml|llama3|gemma|mistral] [--adapter path/to/lora.gguf]
#
# 'auto' (default) lets Ollama read the chat template from the GGUF metadata.
# If the model answers with garbage, try an explicit template.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python
command -v ollama >/dev/null 2>&1 || export PATH="${HOME}/.local/bin:${PATH}"
exec "${PY}" -m scripts.import_gguf "$@"
