"""CLI: import a local GGUF file into Ollama (wrapper around src.local_models.import_gguf)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.local_models import CHAT_TEMPLATES, OllamaError, import_gguf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path, help="path to the .gguf file")
    ap.add_argument("name", help="model name to create in Ollama, e.g. my-finetune")
    ap.add_argument("--template", default="auto", choices=CHAT_TEMPLATES)
    ap.add_argument("--adapter", type=Path, default=None, help="optional LoRA adapter (.gguf)")
    args = ap.parse_args(argv)
    try:
        name = import_gguf(args.gguf, args.name, args.template, args.adapter)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Created Ollama model '{name}'. It now appears under 'Local (Ollama)' in the app.")
    print("Tip: run  python -m scripts.check_endpoint --ollama " + name + "  to probe it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
