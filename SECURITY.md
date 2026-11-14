# Security Policy

## Intended scope

LLM Noise & Bias Audit Studio is a single-user research prototype. It is designed
to run inside an isolated GitHub Codespace or a local VS Code Dev Container.

It is not designed or tested for public multi-user hosting. Do not deploy it on
a public server. If you do anyway, you are responsible for authentication, network
isolation and rate limiting.

> **Important:** Everything below describes **design goals and best-effort measures**,
> not guaranteed security properties. The code is largely AI-generated and has not had an
> independent security audit. No warranty, see [DISCLAIMER.md](DISCLAIMER.md) and
> [LICENSE](LICENSE).

## Threat model (summary)

| Asset | Threat | Intended mitigation (best effort) |
|---|---|---|
| API keys | leakage via disk, logs, exports, error messages, child processes | Keys intended to live only in `st.session_state` or env (Codespaces Secrets) and not to be persisted; regex masking of error messages; sandbox child started with an allowlisted environment (`sandbox_environ`) |
| API keys | a custom endpoint entry that sends another service's key to a foreign server | `api_key_env` may not name a provider key (`OPENAI_API_KEY`, ...) or a `GITHUB_`/`AWS_`/`AZURE_`/`GOOGLE_` variable; a key goes to a public host over `https://` only; no HTTP redirects are followed |
| Host / container | malicious or buggy analysis code in the editor | AST validation + separate process + audit hook + resource limits + hard timeout (see below) |
| App UI | other machines on the network using the app (no login) | Streamlit listens on `127.0.0.1` only (`.streamlit/config.toml`, `scripts/start_app.sh`) |
| Spreadsheet users | CSV/Excel formula injection from LLM output | Cells and column headers starting with `= + - @ \t \r` (after leading blanks) are prefixed with `'` on export |
| Local model servers | unauthenticated Ollama / OpenAI-compatible APIs | Bound to `127.0.0.1`; never forwarded; custom endpoint URLs restricted to loopback / private networks unless explicitly overridden; link-local / metadata addresses always refused, also as IPv4-mapped IPv6, hex / decimal notation or a host name that resolves to them; proxy variables ignored for local servers |
| App availability | oversized or pathological model replies | Replies over 5 MB are discarded; parsing is linear in the reply length and capped at 200 000 characters |
| Supply chain | compromised PyPI packages or GitHub Actions | Hash-pinned `requirements.txt` and `requirements-dev.txt` (development tools: pytest, ruff, pip-audit, uv), both installed with `--require-hashes`; `pip-audit` + blocklist in CI; GitHub Actions pinned to commit hashes; Dependabot. **Open:** the Dev Container image is referenced by tag, not by digest |
| Shell | command injection through model names | Strict regex validation of tags; subprocess calls use argument lists (`shell=False`) |

## Analysis-code sandbox (`src/evaluator.py`)

The sandbox is defense in depth, not a security boundary. The real boundary is
the container you run the app in.

1. AST validation rejects: all imports, `while` loops, `global`/`nonlocal`,
   class definitions, async code, any name or attribute starting with `_`,
   dangerous builtins (`open`, `exec`, `eval`, `compile`, `getattr`, `__import__`, ...),
   also when reached as an attribute (`module.getattr`, `module.import_module`),
   and dangerous attributes (`format`, `read_*`, `to_*`, `load`, `eval`, `query`, ...).
   Methods that take a function name as a string (`agg`, `apply`, `transform`, ...) must
   be called directly; they cannot be stored in a variable or passed on.
2. Restricted globals: only an allowlist of builtins plus pre-injected `np`, `pd`,
   `stats`, `px`, `go`, `math`.
3. Process isolation: execution in a fresh `spawn`ed process that receives only an
   allowlist of environment variables (`PATH`, locale, thread counts, ...).
4. Audit hook (`sys.addaudithook`, cannot be removed) installed before the user code
   runs: refuses sockets, new processes, `ctypes`, file writes, and file or directory
   reads outside the Python installation. This is what limits the damage if the
   validator is bypassed.
5. Resource limits (Linux): address space, CPU time, no file creation.
6. Hard wall-clock timeout (default 5 s) followed by `terminate()`/`kill()`.
7. Validated output: only JSON-serialisable metrics, tables and Plotly figure JSON
   are returned to the UI.

Known limits: the validator is a blocklist over large libraries and may have further
gaps; the audit hook does not see what native library code does on its own; on Windows
and macOS there are no memory limits. Do not paste analysis code from sources you do
not trust.

## Port policy

- Port 8501 (Streamlit) must stay **private** in Codespaces. A public port would let
  anyone with the URL use your API keys and run analysis code.
- The app has no login. It listens on `127.0.0.1` only (`[server] address` in
  `.streamlit/config.toml`). Do not change this to `0.0.0.0` and do not publish the port
  from a container (`docker run -p ...`) on a shared network.
- Port 11434 (Ollama) is bound to loopback and is never forwarded.
- Custom endpoint host names are checked when they are configured; the name is resolved
  again when the request is sent, so a hostile DNS server could answer differently then
  (DNS rebinding). Only allow public endpoints you trust.

## Supply-chain notice: LiteLLM (removed)

In 2026 several LiteLLM releases on PyPI were compromised (`1.82.7`, `1.82.8`,
`1.107.0`). Early development versions of this project pinned `litellm==1.106.2`; its
dependency tree also kept `filelock < 4` with open advisories. LiteLLM was removed on
2026-11-08. All LLM calls now go through `src/providers.py`, a small client built on
`httpx` (already a dependency) that speaks the OpenAI Chat Completions and Anthropic
Messages protocols. This reduced the locked dependencies from 81 to 47 packages.
CI fails if `litellm` reappears in `requirements.txt`.

If you ever installed one of the compromised versions in any environment, **rotate all
API keys and cloud credentials** that were present on that machine.

Upgrade policy (intention, not a commitment): dependency bumps via pull requests, not to
releases younger than 7 days.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
"Report a vulnerability" form (Security tab, then Private vulnerability reporting) on this
repository. Include steps to reproduce, impact and affected version/commit.

This is a volunteer hobby project. Reports are handled when time allows; there is no
guaranteed response time, no obligation to fix, and no bug bounty.

## Supported versions

There are no supported versions in the sense of a maintenance commitment. If security
fixes are made, they go to the latest commit on `main` only.
