"""CLI capability probe for any OpenAI-compatible endpoint or local Ollama model.

Examples:
  python -m scripts.check_endpoint --base-url http://127.0.0.1:8080/v1 --model my-finetune
  python -m scripts.check_endpoint --ollama gemma3:270m
  python -m scripts.check_endpoint --registry-id lmstudio-qwen      # entry in models.toml
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.model_registry import CustomModelSpec, load_custom_registry, probe
from src.security import UrlPolicyError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    g.add_argument("--ollama", metavar="TAG", help="installed Ollama model tag")
    g.add_argument("--registry-id", help="id of an entry in models/models.toml")
    ap.add_argument("--model", help="model name expected by the server (with --base-url)")
    ap.add_argument("--api-key-env", default="", help="NAME of env var holding an API key")
    ap.add_argument("--allow-public", action="store_true", help="allow non-local hosts")
    args = ap.parse_args(argv)

    try:
        if args.registry_id:
            specs, errors = load_custom_registry()
            for e in errors:
                print(f"registry warning: {e}", file=sys.stderr)
            match = [s for s in specs if s.id == args.registry_id]
            if not match:
                print(f"No valid entry with id '{args.registry_id}'.", file=sys.stderr)
                return 1
            custom = match[0]
        elif args.ollama:
            custom = CustomModelSpec(id="cli-probe", provider="ollama", model=args.ollama,
                                     api_base="http://127.0.0.1:11434")
        else:
            if not args.model:
                ap.error("--model is required with --base-url")
            custom = CustomModelSpec(id="cli-probe", model=args.model, api_base=args.base_url,
                                     api_key_env=args.api_key_env,
                                     allow_public_endpoint=args.allow_public)
        spec = custom.to_model_spec()
    except (UrlPolicyError, ValueError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    print(f"Probing {spec.model_route} at {spec.api_base} ...")
    rep = asyncio.run(probe(spec))
    yn = {True: "yes", False: "no", None: "unknown"}
    print(f"  reachable          : {yn[rep.reachable]}")
    print(f"  completion works   : {yn[rep.completion_ok]}  (latency {rep.latency_s} s)")
    print(f"  system role        : {yn[rep.system_role]}")
    print(f"  seed deterministic : {yn[rep.seed_deterministic]}")
    print(f"  JSON mode          : {yn[rep.json_mode]}")
    print(f"  logprobs           : {yn[rep.logprobs]}")
    for m in rep.messages:
        print(f"  note: {m}")
    return 0 if rep.completion_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
