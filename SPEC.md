# LLM Noise & Bias Audit Studio: Specification (v2)

> **AI disclosure:** This specification was drafted with the help of Google Gemini and
> Anthropic Claude (incl. Claude-based coding agents). Code in this repository is largely
> AI-generated and has not necessarily been reviewed line by line. This document describes
> design goals and intended behaviour, not guaranteed properties. See README section
> "AI-Assisted Development" and [DISCLAIMER.md](DISCLAIMER.md).

> **Positioning:** This is a **research prototype / educational tool** for single-user use
> in GitHub Codespaces or a local dev container. It is NOT designed for public
> multi-user hosting.

---

## 0. Changelog vs. v1

- Added AI disclosure, honest positioning (no "production-grade" claim).
- Security: allowlist-based sandbox, `spawn` child process with scrubbed env, resource
  limits, supply-chain hardening (hash-pinned deps, `pip-audit`), Excel formula-injection
  escaping, private-port policy.
- Methodology: new **case dimension** (`case_id`) enabling a proper Kahneman noise audit
  (Level / Stable Pattern / Occasion noise), `ddof=0` for MSE decomposition, bootstrap CIs,
  balanced (Latin-square) order permutations, randomized group assignment, parse-method
  tracking, split typed result columns.
- New: **Local open-weight models via Ollama** (section F) and **Bring Your Own Model**
  (section G), any OpenAI-compatible local server, GGUF import, Hugging Face GGUF pulls,
  file-based model registry.
- Cost/ETA guard, CI, tests per phase, modular UI.

---

## 1. Tech Stack & Environment

- **Environment:** GitHub Codespaces **and** local VS Code Dev Containers
  (`.devcontainer/devcontainer.json`, Python 3.11+, non-root `vscode` user,
  port `8501` forwarded with visibility **private**).
- **UI:** Streamlit + Plotly. Code editor: `st.text_area` (monospace); `streamlit-ace`
  dropped (unmaintained since 2021, see §6.5).
- **LLM orchestration:** in-house `httpx` client (`src/providers.py`) + `asyncio`.
  No response caching. (Originally LiteLLM - removed, see §6.6.)
- **Validation:** Pydantic v2; provider-native structured output (`response_format`)
  where supported, then JSON extraction, then regex fallback.
- **Stats/data:** pandas, numpy, scipy, openpyxl. No statsmodels.
- **Config files:** TOML via stdlib `tomllib` (no PyYAML dependency).
- **Quality:** type hints, `ruff`, `pytest`, `pip-audit`, GitHub Actions CI, pre-commit.

---

## 2. Security & Governance

### 2.1 Sandbox (`src/evaluator.py`)
The AST check is a *hygiene* layer, not a security boundary. The real boundary is the
container. Layers:

1. **AST validator (allowlist-oriented):**
   - Block `Import`, `ImportFrom`, `While`, `Global`, `Nonlocal`, `AsyncFunctionDef`,
     `Await`, `ClassDef`, `Try`/`Raise` allowed only with builtin exceptions.
   - Block any attribute or name starting with `_` (covers all dunders).
   - Block names: `__import__ open exec eval compile globals locals vars getattr setattr
     delattr input breakpoint help memoryview type object super`.
   - Block attribute names: `format format_map read_* to_* load loads save savez
     eval query system popen fromfile tofile` (prevents `pd.read_pickle`, `df.to_csv`,
     `np.load(allow_pickle=True)`, `"{0.__class__}".format(x)`, `df.eval/query`).
   - `__builtins__` replaced by an explicit allowlist (`abs all any bool dict enumerate
     float int len list max min range round set sorted str sum tuple zip isinstance
     print ValueError ...`).
   - Pre-injected: `np`, `pd`, `stats` (scipy.stats), `px`, `go`, `math`, `itertools`.
   - Hardening after the security review of 2026-11-14: the dangerous builtin names are
     also blocked as attributes (`module.getattr`, `module.import_module`, `Popen`,
     `environ`, ...), and string-dispatch methods (`agg`, `apply`, `transform`, `pipe`,
     `rolling`, ...) must be the direct callee of a call (no `f = df.agg`).
2. **Process isolation:** fresh interpreter (`python -I src/sandbox_child.py`), child
   receives only the serialized `df_results` + `ground_truth`; environment built from an
   allowlist (`security.sandbox_environ`: `PATH`, locale, temp dir, thread counts).
2a. **Audit hook in the child** (`sys.addaudithook`, installed right before the user
   code): refuses `socket.*`, `subprocess.*`, `ctypes.*`, `os.system/exec/spawn/fork`,
   file writes, and `open` / `listdir` outside the Python installation and the library
   directories. First-time imports of `ctypes`, `socket`, `subprocess`, `multiprocessing`,
   `asyncio`, `ssl` are refused.
3. **Resource limits in child:** `resource.setrlimit` for address space (default 1 GB),
   CPU time (5 s), file size (0). Hard wall-clock timeout 5 s → `terminate()` / `kill()`.
4. **Output contract:** `evaluate()` must return a dict with `summary_metrics: dict`,
   optional `per_model_table: DataFrame`, optional `figures: list[plotly Figure]`
   (transferred as JSON via `fig.to_json()`), validated by Pydantic in the parent.

### 2.2 Secrets
- Keys live only in `st.session_state` or in Codespaces Secrets (env vars). Never
  written to disk, logs, exports, or run-config JSON.
- Exception messages pass a regex masker (`sk-…`, `sk-ant-…`, `AIza…`, `Bearer …`,
  generic 32+ char tokens) before rendering.
- `.env.example` only documents variable names and recommends Codespaces Secrets.

### 2.3 Supply chain
- `requirements.in` → `requirements.txt` via `pip-compile --generate-hashes`;
  install with `--require-hashes`.
- Dependabot + `pip-audit` in CI. LiteLLM had compromised PyPI releases in 2026:
  pin to a verified version and document key rotation in SECURITY.md.

### 2.4 Exports
- CSV/XLSX: escape cells and column headers starting with `= + - @ \t \r`, also after
  leading blanks or line breaks (prefix `'`).
- Exported run config (`run_config.json`) contains models, params, seeds, prompts,
  schema, never keys.

### 2.5 Network
- Port 8501: private only, and Streamlit binds to `127.0.0.1` (`[server] address` in
  `.streamlit/config.toml`, `--server.address` in `scripts/start_app.sh`); the app has no
  login. Ollama (11434) and other local model servers bind to `127.0.0.1` and are never
  forwarded.
- HTTP clients do not follow redirects; for loopback / private hosts they ignore proxy
  variables (`trust_env=False`). Responses over 5 MB are discarded.
- Custom endpoints (§G) default to a **loopback/private-network allowlist**; public
  URLs require an explicit "I understand" toggle (SSRF / data exfiltration awareness).

### 2.6 Governance files
`LICENSE` (MIT, AS-IS), `SECURITY.md` (scope, threat model, sandbox layers, disclosure),
`CONTRIBUTING.md` (incl. AI-assistance policy), `CODE_OF_CONDUCT.md` (optional),
README "AI-Assisted Development" + "Third-party model licenses" sections.

---

## 3. Directory Structure

```text
.
├── .devcontainer/
│   └── devcontainer.json          # non-root, Python 3.12, port 8501 private
├── .github/
│   ├── workflows/ci.yml           # ruff, pytest, pip-audit
│   └── dependabot.yml
├── models/
│   ├── models.example.toml        # documented template for custom models (committed)
│   ├── models.toml                # user's own registry (git-ignored)
│   └── modelfiles/                # optional Ollama Modelfiles (examples committed)
├── scripts/
│   ├── install_ollama.sh          # opt-in, pinned + checksum-verified
│   ├── pull_starter_model.sh      # optional: pull one small model
│   ├── import_gguf.sh             # wrap a local .gguf into an Ollama model
│   └── check_endpoint.py          # CLI smoke test for any custom endpoint
├── src/
│   ├── __init__.py
│   ├── config.py                  # constants, defaults, curated LOCAL_MODEL_CATALOG
│   ├── schemas.py                 # 7 output schemas + parsers (+ parse_method)
│   ├── cases.py                   # case-set loading (CSV) & validation
│   ├── context_engine.py          # N isolated windows x cases x variation modes
│   ├── llm_runner.py              # async executor, retries, mock simulator
│   ├── model_registry.py          # unified model list: cloud + Ollama + custom endpoints
│   ├── local_models.py            # Ollama manager (status, list, pull, delete, import)
│   ├── presets.py                 # 6+ statistical presets with plain-English metadata
│   ├── evaluator.py               # AST + spawn + rlimit sandbox
│   ├── export.py                  # safe CSV/XLSX + run_config.json
│   └── security.py                # secret masking, URL policy, tag validation
├── ui/
│   ├── sidebar.py
│   ├── tab_models.py              # "Models" tab: local manager + BYOM wizard
│   ├── tab_setup.py
│   ├── tab_editor.py
│   └── tab_results.py
├── tests/
│   ├── test_schemas.py
│   ├── test_context_engine.py
│   ├── test_presets.py
│   ├── test_sandbox_security.py
│   ├── test_model_registry.py     # TOML parsing, URL policy, mocked endpoints
│   ├── test_local_models.py       # mocked Ollama HTTP API
│   └── test_export.py
├── docs/
│   └── BRING_YOUR_OWN_MODEL.md    # step-by-step guide (see §G)
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── SECURITY.md
├── README.md
├── SPEC.md
├── pyproject.toml                 # ruff/pytest config
├── requirements.in
├── requirements.txt               # hash-pinned (generated)
└── app.py                         # thin entry point wiring the ui/ modules
```

---

## 4. Core Components

### A. Output schemas (`src/schemas.py`)
Same 7 schemas as v1 (Boolean, Percentage, Float, Likert 1..K, Forced-Choice Enum,
Ranking, Confidence Tuple). Additions:
- Parse order: native structured output → JSON block extraction → regex fallback.
- Every parse records `parse_method` in {`native`, `json`, `regex`, `failed`}.
- Regex fallback for numeric schemas takes the **last** number in an explicit
  `ANSWER:` line only, never an arbitrary number (avoids extracting the anchor itself).
- Enum matching: case-insensitive exact match after trimming; no fuzzy matching.
- Ranking: must be a permutation of the provided options (no missing/duplicate items).

### B. Cases & context windows (`src/cases.py`, `src/context_engine.py`)
- **Case set:** single prompt OR CSV upload with columns `case_id, prompt[, ground_truth]`.
  A noise audit in Kahneman's sense needs multiple cases x multiple judges.
- **Judges** = model x persona (when persona jitter is active) or model.
- Variation modes (as v1) with fixes:
  - *Occasion noise:* identical prompts; warn if temperature == 0.
  - *Order permutation:* balanced Latin-square / cyclic design over k chunks;
    if k! <= N, enumerate all permutations evenly.
  - *Anchoring / distractor:* **randomized** 50/50 assignment with fixed seed.
  - *Framing:* user provides **two** frame texts (A and B); randomized balanced split.
  - *Persona jitter:* round-robin over personas, balanced per case.
- Live preview + deterministic `design_seed` stored in run config.

### C. Execution engine (`src/llm_runner.py`)
- `asyncio.gather` + `Semaphore` (per-provider concurrency) + exponential backoff with
  jitter; honors `Retry-After`.
- Stateless requests; `seed=i` where supported; LiteLLM caching OFF; no random nonces
  injected into prompts. `drop_params=True`; actually-sent params are logged.
- Note in docs: provider prompt caching does not return cached completions; it does not
  bias variance. Cloud `seed` is best-effort.
- **Pre-run guard:** shows `calls = cases x models x N`, estimated cost (LiteLLM price map;
  local = $0) and ETA (local: measured sec/call from a 1-call warm-up). Confirmation
  required above configurable thresholds.
- **Mock mode:** deterministic given seed; simulates model-specific level bias, pattern
  noise per case, occasion noise, anchoring/framing sensitivity and parse failures for
  all 7 schemas.

**DataFrame contract `df_results`:**
`run_id, timestamp, case_id, model, model_source (cloud|ollama|custom|mock),
context_window_id, context_variant, persona, system_prompt, final_prompt, raw_output,
parsed_value_num (float|NaN), parsed_value_json (str|None), confidence, token_prob,
parse_success, parse_method, seed, temperature, model_version_returned, finish_reason,
prompt_tokens, completion_tokens, cost_usd, latency_ms, error (masked)`.

Parse success rate is reported **per model x variant**; presets warn if failure rates
differ by more than 10 percentage points between compared groups.

### D. Statistical presets (`src/presets.py`)
As v1 (id, name, compatible_schemas, plain_english_summary, kahneman_concept,
how_to_interpret, code), plus `min_n`, `requires_ground_truth`, `requires_cases`.
All numeric metrics get **bootstrap 95% CIs** (default 1000 resamples, seeded).

1. **MSE decomposition**: `MSE = Bias^2 + Noise^2` with population SD (`ddof=0`);
   per case and averaged over cases.
2. **MPD & RMS noise**: intra- and inter-model; normalized by scale range where defined.
3. **Noise audit (Level / Pattern / Occasion)**: two-way decomposition over
   judges x cases with replicates: `System^2 = Level^2 + StablePattern^2 + Occasion^2`.
   Requires >= 2 cases and >= 2 judges; otherwise explains why it cannot run.
4. **Categorical stability**: mode agreement, flip rate defined as
   `1 - mode share`, pairwise disagreement, normalized Shannon entropy.
5. **Anchoring / framing effect**: Mann-Whitney U + Cohen's d (numeric);
   Fisher's exact test + risk difference (Boolean / Enum via chi-square).
6. **Ranking stability**: mean normalized Kendall tau distance + Kendall's W.

### E. UI (`app.py` + `ui/`)
Tabs: **0 Models**, 1 Prompt & Context, 2 Statistics Editor & Guide, 3 Results & Export.

> Note (2026-11-14): the UI has since been reworded for non-technical users. The tabs are
> now *Start here*, *1 · Question*, *2 · Analysis & run*, *3 · Results* and *More models*;
> a context window is called a *repetition* in the UI and mock models are called *practice
> models*. The start page has its own run button and shows a summary of the last result.
> The current wording is described in [docs/USER_GUIDE.md](docs/USER_GUIDE.md).
Sidebar: mock toggle (ON when no model is usable), N slider (1–50), temperature,
concurrency, masked API keys, model multi-select grouped by source
(Cloud / Local Ollama / Custom).


---

### F. Local open-weight models via Ollama (`src/local_models.py`)

- **Runtime:** opt-in `scripts/install_ollama.sh` (pinned version + SHA-256 check, see §6.5).
  No automatic model pull on create (protects the 15 GB free-tier storage quota);
  optional `scripts/pull_starter_model.sh`.
- **Binding:** `127.0.0.1:11434` only; never in `forwardPorts` (Ollama API is unauthenticated).
- **Curated catalog** (`LOCAL_MODEL_CATALOG` in `config.py`): dataclass entries
  `tag, display_name, params_b, approx_disk_gb, min_ram_gb, license, license_url,
  open_source_osi: bool, notes`. ~5 CPU-friendly models (<= 4B params). Tags are
  verified against the Ollama library during Phase 3, not guessed.
- **Hardware check:** RAM from `/proc/meminfo`, free disk via `shutil.disk_usage`;
  traffic light per model (fits / tight / won't fit). Hint to switch to a 4-core/16 GB
  Codespace (consumes free core-hours twice as fast).
- **Wording:** UI says "open-weight" unless the license is OSI-approved; each model shows
  its license link. The project never redistributes weights.

---

### G. Bring Your Own Model (BYOM): make adding *your own* local model trivial

**Goal:** Any user can plug in their own local model in under 5 minutes, via UI *or* a
config file, without touching Python code. Four supported paths, one unified registry.

#### G.1 Supported paths

| Path | Typical user | How it is added | Model route |
|---|---|---|---|
| 1. Ollama library tag | "I want model X from ollama.com" | UI: type tag → Pull | `ollama_chat/<tag>` |
| 2. Hugging Face GGUF | "I found a GGUF on Hugging Face" | UI: paste `hf.co/<user>/<repo>[:<quant>]` → Pull | `ollama_chat/hf.co/...` |
| 3. Local `.gguf` file / fine-tune | "I have my own fine-tuned weights" | UI upload or `scripts/import_gguf.sh` → Modelfile → `ollama create` | `ollama_chat/<name>` |
| 4. Any OpenAI-compatible server | LM Studio, llama.cpp `llama-server`, vLLM, LocalAI, llamafile, Jan, own FastAPI wrapper | UI wizard or `models/models.toml` entry | `openai/<model>` + `api_base` |

Path 4 is the universal escape hatch: **if it speaks the OpenAI Chat Completions API,
it works.** The wizard pre-fills common default base URLs for popular servers, the
actual defaults are verified in Phase 3 and shown as editable suggestions only.

#### G.2 Model registry file (`models/models.toml`)

- Git-ignored user file; committed, fully commented `models/models.example.toml`.
- Parsed with stdlib `tomllib`, validated by a Pydantic model `CustomModelSpec`:

```toml
# models/models.toml: your own models. Use "Reload registry" in the UI after editing.

[[model]]
id           = "my-finetune"                 # unique, shown in the UI
display_name = "My Llama fine-tune (Q4)"
provider     = "openai_compatible"           # "ollama" | "openai_compatible"
model        = "my-finetune-q4"              # name the server expects
api_base     = "http://127.0.0.1:8080/v1"
api_key_env  = ""                            # optional: NAME of an env var, never the key itself
supports_seed            = true
supports_json_mode       = false             # false -> prompt-only formatting + JSON/regex parse
supports_logprobs        = false
max_concurrency          = 1
timeout_s                = 120
system_prompt_supported  = true              # false -> system prompt prepended to user message
license  = "Llama 3.2 Community License"
notes    = "Fine-tuned on internal data, CPU only"

[model.default_params]                       # optional extra params passed through
top_p = 0.95
```

- Keys are **never** stored in the TOML, only the *name* of an env var / Codespaces Secret.
- Invalid entries are reported in the UI with a precise, friendly message (entry, field,
  reason) and skipped; the rest of the registry still loads.

#### G.3 "Add your own model" wizard (Tab 0 → "Bring Your Own Model")

1. **Choose path** (Ollama tag / Hugging Face GGUF / upload GGUF / OpenAI-compatible URL).
2. **Fill in** minimal fields; advanced options collapsed.
3. **Test connection** runs a capability probe (G.4) and shows a green/yellow/red checklist.
4. **Save** → appended to `models/models.toml` (or "session only" to just try it).
5. Model immediately appears in the sidebar multi-select under "Custom".


#### G.4 Capability probe (`model_registry.probe(spec)`)

A handful of tiny requests that report:
- reachable / auth OK / model name accepted (`GET /v1/models` if available),
- basic completion works + latency of one call (feeds the ETA),
- `seed` honored? (two identical seeded calls at temperature > 0 → identical output?),
- JSON mode / `response_format` accepted?
- logprobs returned?
- system role accepted?

Results are stored as `probed_capabilities`, shown as badges, and used by the runner to
adapt automatically (e.g. fall back to prompt-only JSON instructions). They are exported
in `run_config.json` so every audit documents which capabilities were actually present.

#### G.5 Import local GGUF / fine-tunes

- `scripts/import_gguf.sh path/to/model.gguf my-name [--template chatml|llama3|...]`
  generates a Modelfile from `models/modelfiles/` templates and runs `ollama create`.
- UI upload: configurable size limit (default 8 GB), streamed to a temp dir **outside**
  the repo, extension + GGUF magic bytes checked, temp file deleted after import.
- Optional LoRA adapter support via the `ADAPTER` line in the Modelfile template.
- Warning in UI/docs: wrong chat templates are the #1 cause of garbage output, the
  wizard lets users pick a template and runs the probe afterwards.

#### G.6 Where does "local" actually run? (must be documented clearly)

- **In Codespaces:** "local" = inside the Codespace VM. A model running on *your own
  laptop* (e.g. LM Studio) is **not** reachable from a cloud Codespace.
- **Own GPU machine:** open the repo with **VS Code Dev Containers locally** (same
  `devcontainer.json`). Host servers are reachable via `http://host.docker.internal:<port>`,
  which is on the default URL allowlist.
- Tunnelling a laptop server into a Codespace is possible but not documented as a default
  (it exposes an unauthenticated API).
- Later/optional: `docker-compose.gpu.yml` for GPU-enabled Ollama on Linux/NVIDIA hosts.

#### G.7 Safety rules for BYOM

- URL policy (`security.py`): default allowlist `127.0.0.1`, `localhost`, `::1`,
  `host.docker.internal`, RFC1918 ranges. Public hosts need an explicit opt-in and show a
  warning that prompts will leave the machine. Link-local / metadata addresses are always
  refused: as literal, as IPv4-mapped IPv6, in hex / decimal / short notation, and as a
  host name that resolves to one (checked at configuration time, cached for 60 s).
- `api_key_env` may not name a provider key (`PROVIDER_KEY_ENV`) or a variable starting
  with `GITHUB_`, `GH_`, `AWS_`, `AZURE_`, `GOOGLE_`, `GCP_`, `CODESPACE`. With a key, a
  public endpoint must use `https://`.
- Model names / tags validated by regex; subprocess calls use argument lists only
  (`shell=False`), never string interpolation.
- Imported weights and their licenses are the user's responsibility; README says so.

#### G.8 Documentation & tests

- `docs/BRING_YOUR_OWN_MODEL.md`: one copy-paste recipe per path + troubleshooting table
  (connection refused, 404 model not found, template garbage, out of memory, too slow).
- Tests: TOML parsing (valid/invalid), URL policy, probe logic against a mocked
  OpenAI-compatible server (no network), Modelfile generation, tag regex.
- `scripts/check_endpoint.py --base-url ... --model ...` runs the same probe from the CLI.

---

## 5. Phased Execution Plan (tests in every phase)

| Phase | Deliverables | Tests / checks |
|---|---|---|
| 1 | Architecture & security blueprint (§6), resolve open questions | - |
| 2 | devcontainer, CI, governance files, `requirements.in/.txt`, `config.py`, `schemas.py`, `security.py` | `test_schemas.py`, masking/URL-policy tests, ruff |
| 3a | `cases.py`, `context_engine.py`, `llm_runner.py` (+ mock), `evaluator.py` | context + sandbox security tests |
| 3b | `presets.py` (6 presets + explainers), `export.py` | preset tests on deterministic mock data |
| 3c | `model_registry.py`, `local_models.py`, scripts, `models.example.toml` | registry / probe / Ollama-mock tests |
| 4 | `app.py` + `ui/`, README, `docs/BRING_YOUR_OWN_MODEL.md` | full pytest, manual smoke test in Codespace |


---

## 6. Phase 1 Blueprint

### 6.1 Schema → preset compatibility matrix

| Schema \ Preset | 1 MSE | 2 MPD/RMS | 3 Noise audit | 4 Categorical | 5 Effect size | 6 Ranking |
|---|---|---|---|---|---|---|
| Boolean | yes | yes | roughly proportions | yes | yes Fisher | - |
| Percentage | yes | yes | yes | - | yes | - |
| Float | yes | yes | yes | - | yes | - |
| Likert 1..K | yes | yes | yes | roughly as categories | yes | - |
| Forced-Choice Enum | - | - | - | yes | yes chi-square | - |
| Ranking | - | - | - | - | - | yes |
| Confidence Tuple | yes decision | yes | yes | - | yes | - |

yes = recommended · roughly = possible with caveats (shown in UI) · - = hidden by default

### 6.2 File contracts (key signatures)

```python
# schemas.py
def build_format_instructions(schema: SchemaConfig) -> str: ...
def parse_response(raw: str, schema: SchemaConfig) -> ParseResult: ...
#   ParseResult: value_num, value_json, confidence, success, method

# context_engine.py
def build_windows(cases: list[Case], mode: VariationConfig, n: int,
                  design_seed: int) -> list[ContextWindow]: ...

# model_registry.py
def list_models(session_keys: dict[str, str]) -> list[ModelSpec]: ...   # cloud + ollama + custom
def load_custom_registry(path: Path) -> tuple[list[CustomModelSpec], list[RegistryError]]: ...
async def probe(spec: ModelSpec) -> CapabilityReport: ...

# local_models.py
def ollama_status() -> OllamaStatus: ...
def list_installed() -> list[InstalledModel]: ...
def pull(tag: str) -> Iterator[PullProgress]: ...
def delete(tag: str) -> None: ...
def import_gguf(path: Path, name: str, template: str) -> None: ...

# llm_runner.py
async def run_audit(windows: list[ContextWindow], models: list[ModelSpec],
                    params: RunParams, mock: bool) -> pd.DataFrame: ...   # §4C contract

# evaluator.py
def validate_code(code: str) -> list[Violation]: ...
def run_sandboxed(code: str, df: pd.DataFrame, ground_truth: float | None,
                  timeout_s: float = 5.0) -> EvaluationResult: ...
```

### 6.3 Sandbox rules (summary)

AST allowlist/blocklist (§2.1) → child with allowlisted env → audit hook → rlimits (1 GB address
space, 5 s CPU, no file writes) → 5 s wall-clock kill → Pydantic-validated return payload.

Inputs that MUST be rejected or safely fail in tests:
`import os`, `__import__('subprocess')`, `while True: pass`, `().__class__.__bases__`,
`"{0.__class__}".format(1)`, `pd.read_pickle('x')`, `df.to_csv('x')`,
`np.zeros(10**11)`, `open('/etc/passwd')`, `getattr(np, 'load')`.

### 6.4 Open questions to verify before Phase 2

1. ~~Pin a LiteLLM version~~ → resolved in §6.5.
2. ~~`streamlit-ace`~~ → dropped (§6.5).
3. ~~Ollama devcontainer feature~~ → replaced by pinned install script, version pin still open (§6.5).
4. (see §6.5) Verify starter catalog tags and licenses on ollama.com.
5. (see §6.5) Verify whether Ollama via LiteLLM returns logprobs (`token_prob` column).
6. (see §6.5) Verify default base URLs/ports of LM Studio, llama.cpp server, vLLM, LocalAI for the
   wizard presets.
7. (see §6.5) Confirm Ollama's Hugging Face GGUF pull syntax (`hf.co/<user>/<repo>:<quant>`).


### 6.5 Phase 1 decisions (resolved 2026-11-08)

| # | Question | Finding | Decision |
|---|---|---|---|
| 1 | LiteLLM version | Compromised PyPI releases in 2026: `1.82.7`, `1.82.8` (March) and `1.107.0` (Oct 21, quarantined by PyPI ~2 h later). LiteLLM and security vendors (Wiz, Datadog, Chainguard, Cycode) name **`1.106.2`** as the last safe version; Chainguard also lists `1.107.0rc1/rc2` as clean. Later 1.106.x releases exist on PyPI but were not independently assessed. | Pin `litellm==1.106.2` with `--require-hashes`. CI blocklist rejects `1.82.7`, `1.82.8`, `1.107.0`. **Policy:** upgrades only via reviewed PR, never to a release younger than 7 days. SDK only (`litellm.acompletion`) — no proxy server, no Docker images. |
| 2 | `streamlit-ace` | Last release `0.1.1` on 2021-12-07, single maintainer, no provenance. | **Dropped.** Use `st.text_area` (monospace via CSS). No extra dependency. |
| 3 | Ollama install | Latest Ollama release could **not** be verified in this session (GitHub API/releases page unreachable from the Codespace). Third-party devcontainer features could not be verified either (fetch timed out / 404). The official installer script supports `OLLAMA_VERSION` and requires `zstd`. | **No third-party feature.** Opt-in `scripts/install_ollama.sh` with an explicit `OLLAMA_VERSION` pin (chosen in Phase 2 from the GitHub releases page; must be >= 7 days old), checksum verification, install to `~/.local`, `ollama serve` with `OLLAMA_HOST=127.0.0.1:11434`. Keeps the base container lean for cloud-only users. **Open item for Phase 2.** |
| 4 | Starter catalog | Verified tags on ollama.com (2026-11-08): `gemma3:270m` (292 MB), `qwen3:0.6b` (523 MB), `gemma3:1b` (806 MB), `llama3.2:1b` (1.3 GB), `qwen3:1.7b` (1.4 GB), `llama3.2:3b` (2.0 GB), `qwen3:4b` (2.5 GB). | Catalog = these 7. Licenses: Qwen3 = Apache-2.0 (OSI); Llama 3.2 = Llama 3.2 Community License (open-weight); Gemma 3 = Gemma Terms of Use (open-weight). License strings re-checked against model pages in Phase 2. Note: `qwen3` models default to *thinking* mode — runner strips `<think>…</think>` before parsing and offers a "disable thinking" option. |
| 5 | Logprobs via Ollama | Not verified from docs. | Not assumed. The capability probe (§G.4) detects it at runtime; `token_prob` stays `None` otherwise. |
| 6 | Default ports for BYOM presets | Well-known defaults: Ollama `11434`, LM Studio `1234`, llama.cpp `llama-server` `8080`, vLLM `8000`, LocalAI `8080`. | Used only as **editable suggestions** in the wizard; the probe confirms. |
| 7 | Hugging Face GGUF pull | Confirmed: `hf.co/{user}/{repo}` with optional `:{quant}` (case-insensitive, default Q4_K_M if present); `huggingface.co` also accepted. Chat template is auto-selected from GGUF metadata. | Validation regex: `^(hf\.co|huggingface\.co)/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?$`. Private repos (SSH key) documented, not automated. |

Additional Phase 1 findings:
- **Codespace resources (measured):** 2 cores, ~7.8 GB RAM, Python 3.14 on host image.
  The devcontainer pins **Python 3.12** (LiteLLM requires `>=3.10,<3.15`; 3.12 has the
  broadest wheel coverage for numpy/scipy/pandas).
- On 2 cores / 8 GB, models up to ~3B (Q4) are the practical ceiling; `qwen3:4b` is
  marked "tight" by the hardware check.

### 6.6 Decision: LiteLLM removed (2026-11-08)

- **Trigger:** `pip-audit` reported 13 advisories in `filelock 3.32.7`. The fix needs
  `filelock >= 4.2.1`, but `litellm 1.106.2` (last version confirmed clean) requires
  `filelock < 4`; `filelock` was only present because of LiteLLM.
- **Analysis:** LiteLLM was used in one function (`acompletion`) plus its price map;
  31 of 79 locked packages existed only because of it.
- **Decision:** replace it with `src/providers.py` (~250 lines, `httpx`): OpenAI Chat
  Completions protocol (OpenAI, OpenRouter, Gemini's official OpenAI-compatible
  endpoint, Ollama `/v1`, any custom server) and Anthropic Messages protocol.
  Anthropic is pinned to `anthropic-version: 2023-06-01`, because version `2026-10-21`
  removed `temperature`/`top_p`/`top_k`, which noise experiments require.
- **Cost estimate:** static, dated price table in `config.py` (approximate; unknown
  models show "cost unknown").
- **Result:** 47 locked packages (was 81), `pip-audit`: no known vulnerabilities;
  model ids (`provider/model`) and `models.toml` are unchanged.
- **Limits:** only these providers are supported natively (others via OpenRouter or any
  OpenAI-compatible gateway); provider API changes must be followed manually; the real
  cloud endpoints are tested only against recorded request/response formats.
- Rows §6.4 item 1/5 and §6.5 row 1 above are kept as history.

