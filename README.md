# LLM Noise & Bias Audit Studio

[![CI](https://github.com/DarianVoester/llm_noise_bias_audit/actions/workflows/ci.yml/badge.svg)](https://github.com/DarianVoester/llm_noise_bias_audit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/DarianVoester/llm_noise_bias_audit)

If you ask a language model the same question twenty times, do you get the same answer
twenty times? Usually not. This app lets you measure how much the answers scatter (noise)
and whether they lean in one direction (bias). It asks the question in many separate
conversations, so the model never sees its earlier answers, and it can do that for several
models side by side.

The idea comes from *Noise: A Flaw in Human Judgment* by Daniel Kahneman, Olivier Sibony and
Cass R. Sunstein, which describes the same problem for human judges. You do not need to know
statistics or programming to use the app. Every number it shows comes with an explanation in
plain words.

> **Research prototype / educational tool.** Single-user, meant for GitHub Codespaces or
> a local Dev Container. Not designed for public multi-user hosting. See [SECURITY.md](SECURITY.md).
>
> **Free, non-commercial project by a private individual, provided "AS IS" without
> warranty.** You run your own copy and are responsible for your API keys, costs, data,
> provider terms and model licenses. Not for decisions about individual people. See
> [DISCLAIMER.md](DISCLAIMER.md).

## Try it in two minutes

The easiest way is GitHub Codespaces, because nothing has to be installed on your computer.

1. On the GitHub page of this repository, click **Code → Codespaces → Create codespace**
   (or the badge above) and wait until the setup has finished.
2. The app starts by itself and opens in a new browser tab. If no tab opens, go to the
   **PORTS** panel and click the globe icon next to port 8501. Leave the port private.
3. On the page **Start here**, choose a test and press **Run the test**. The result appears
   on the same page.

That first run uses practice models. They are simulations that cost nothing and need no
account, so you can see what results look like before you connect a real model.

The [User Guide](docs/USER_GUIDE.md) walks through the first run, explains which settings
belong together and shows how to read the results.

## What you can test

The start page offers seven ready-made tests. Each one is phrased as a question:

- Does a model give the same answer every time?
- How far off is a model, and is that bias or noise?
- Why do different models disagree?
- Does an irrelevant number sway the answer?
- Does the wording change the answer?
- Does a model flip its yes/no decision?
- Does a model reshuffle its priorities?

You can also build your own test in the tabs **1 · Question** and **2 · Analysis & run**.
There you write the question, say what form the answer must have, and decide what changes
between the repetitions.

The answer can be a yes or no, a percentage, any number, a rating on a scale, one of several
categories, a ranking, or a decision plus a confidence. The app adds strict format
instructions to every question (percentages on the 0 to 100 scale, never as fractions, for
example). An answer that breaks the format, such as `0.42` instead of `42`, is rejected and
the model is reminded once in the same conversation. The column `format_retries` and both
replies in `raw_output` show when that happened. In code you can switch the reminder off
with `RunParams(repair_format=False)`.

Between repetitions you can change nothing at all (which measures pure chance), the order
of the facts, an irrelevant number that is slipped into half of the conversations, the
wording of a fact, or the role the model is told to play.

Six analyses are built in:

| Analysis | What it tells you |
|---|---|
| MSE noise decomposition | How much of the error is bias and how much is noise (MSE = Bias² + Noise²). Needs the correct answer. |
| Mean pairwise distance & RMS noise | How much the answers disagree, without knowing the correct answer. |
| Noise audit (level / pattern / occasion) | Why several judges disagree. |
| Categorical stability | How often a model flips its yes/no answer or its category. |
| Anchoring & framing effect size | Whether an irrelevant number or a different wording shifted the answers. |
| Ranking stability | Whether a model keeps its ranking (Kendall's τ and W). |

Before a run, the app checks whether your settings fit together and says what to change if
they do not. It also shows the number of requests, an estimated cost and the expected
duration. Results can be downloaded as CSV, as an Excel file and as a settings file that
lets you repeat the run.

The app has a light and a dark theme and follows your system setting. To switch by hand,
open the menu at the top right, then **Settings → Theme**.

## Running it on your own machine

```bash
python -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt
pip install --require-hashes -r requirements-dev.txt     # tests, linting
streamlit run app.py
```

The app has no login, so it listens on `127.0.0.1` only (set in `.streamlit/config.toml`):
it can be opened from the same machine, not from other machines on the network. Leave it
that way. Anyone who can reach the app can use your API keys and run analysis code.

## Starting, stopping and restarting

In Codespaces the app starts by itself. After that there are three ways to control it, and
they all do the same thing:

- In the app: open **App: restart or stop** at the bottom of the sidebar.
- In VS Code: press **Ctrl+Shift+B** to restart the app (or to start it if it is not
  running). The other actions are under **Ctrl+Shift+P → Tasks: Run Task**: *App: Start*,
  *App: Stop*, *App: Status* and *App: Show log*.
- In a terminal: `bash scripts/start_app.sh`, followed by `stop`, `restart`, `status` or
  `log` if you want something other than starting. The log file is
  `~/.local/state/llm-noise-audit/streamlit.log`.

Restart the app after you have updated the code. A running app does not always notice
changed files, and the usual sign is an `ImportError` on the page.

To debug, use **Run and Debug → "Streamlit: debug app.py"**, which stops the background app
first.

## Using real models

### Cloud models

OpenAI, Anthropic, Gemini and OpenRouter work with an API key. The best place for keys is
**GitHub Codespaces Secrets** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`OPENROUTER_API_KEY`). You can also paste them into the sidebar. The app is designed to keep
them in the browser session's memory only and not to write them to disk, logs or exports
(best effort, no guarantee). The provider bills you for every request, so set a spending
limit there.

### Free local models with Ollama

```bash
bash scripts/install_ollama.sh                 # pinned release, SHA-256 verified, loopback only
bash scripts/pull_starter_model.sh gemma3:270m # 292 MB, good for a first test
```

The tab **More models** does the same with buttons. It also shows whether a model fits your
hardware, how much disk space the models use, and lets you delete them again. The installer
adds `zstd` on its own and copes with broken third-party apt sources (such as the Yarn
repository in some devcontainer images).

A standard Codespace has 2 CPU cores, 8 GB RAM and no GPU. Models up to about 3B parameters
run acceptably, though each answer takes seconds. The free Codespaces storage quota (15 GB)
is shared by all your codespaces, so delete models you no longer need.

### Your own model

There are four ways to connect a model of your own, and none of them needs Python. They are
described in [docs/BRING_YOUR_OWN_MODEL.md](docs/BRING_YOUR_OWN_MODEL.md).

1. Any tag from the Ollama library. Type it in the tab **More models**.
2. Any GGUF file on Hugging Face, written as `hf.co/<user>/<repo>[:<quant>]`.
3. Your own `.gguf` file or fine-tune. Upload it or run `bash scripts/import_gguf.sh`.
4. Any server that speaks the OpenAI API (LM Studio, llama.cpp, vLLM, LocalAI and others),
   through the form in the app or through `models/models.toml`. A built-in probe checks
   whether the server supports seeds, JSON mode, logprobs and a system role, and the app
   adapts to what it finds.

## How the code is organised

```text
app.py                  thin Streamlit entry point
.streamlit/config.toml  light + dark theme, loopback-only server address
ui/
  tab_guide.py          start page: ready-made tests, run button, summary
  tab_setup.py          question, answer format, change between repetitions
  tab_editor.py         analysis choice, settings check, run panel
  tab_results.py        results, charts, single answers, downloads
  tab_models.py         Ollama manager, form for your own model
  sidebar.py, legal.py  settings, disclaimer
src/
  config.py             defaults, cloud + local model catalogs
  guidance.py           ready-made tests, settings check, metric glossary, chart tips
  schemas.py            7 answer formats: strict instructions, validation, repair prompt
  cases.py              single question / CSV files with several questions
  context_engine.py     N isolated conversations per question, 5 kinds of variation
  llm_runner.py         async executor, retries, format reminder, seeds, cost/ETA guard
  providers.py          tiny httpx client: OpenAI-compatible + Anthropic protocols
  mock_simulator.py     deterministic practice models with known noise & bias
  presets.py            6 analyses with their explanations
  evaluator.py          AST validator + isolated-process sandbox
  sandbox_child.py      child process: restricted globals, audit hook, rlimits, timer
  model_registry.py     registry for your own models (models.toml) + capability probe
  local_models.py       Ollama manager, GGUF import, hardware check
  export.py             safe CSV / XLSX / run config
  security.py           secret masking, URL policy, identifier checks
scripts/                start_app.sh, install_ollama.sh, pull_starter_model.sh, import_gguf.sh
docs/                   USER_GUIDE.md, BRING_YOUR_OWN_MODEL.md
```

In the code, one of those separate conversations is called a *context window*. The app and
the user guide call it a repetition.

The result table `df_results` has one row per model, question and repetition. Besides the
answer it records `parsed_value_num`, `parsed_value_json`, `parse_method`, `format_retries`,
`seed`, `temperature`, `dropped_params` (settings a model refused, so the call ran without
them), `model_version_returned`, `latency_ms`, `cost_usd` and more. This is meant to make
runs traceable and, where the provider honours seeds, repeatable. Exact reproducibility is
not guaranteed.

## Development

```bash
ruff check .
pytest                 # includes sandbox attack cases, format checks and UI smoke tests
pytest -m ollama       # optional: round trip with a real local model
```

## Security

These measures are **best-effort design goals, not guarantees**. User analysis code is
AST-checked (aiming to block imports, `while`, dunders, file/eval access) and runs in a
separate Python process that gets only an allowlist of environment variables, memory/CPU
limits and a 5-second timer. Inside that process an audit hook refuses network access, new
processes, file writes and file reads outside the Python installation, so a gap in the AST
check does not by itself expose your keys or files.

The app has no login and listens on `127.0.0.1` only. An API key for a custom endpoint must
come from a variable of its own (not `OPENAI_API_KEY` and the like) and is sent to public
hosts over `https://` only. Custom endpoint addresses that point to cloud metadata services
are refused in every notation.

Dependencies and development tools are hash-pinned and checked with `pip-audit` in CI, and the GitHub Actions are
pinned to commits. LLM calls use a small in-house `httpx` client instead of a large SDK.
LiteLLM was removed after several compromised 2026 releases, which cut the dependency tree
from 81 to 47 packages. Exports are designed to escape spreadsheet formulas (cells and
column headers) and to leave out keys. Details: [SECURITY.md](SECURITY.md).

## AI-Assisted Development

Honest disclosure: the specification was drafted with **Google Gemini** and **Anthropic
Claude**, and the code was written with an AI coding agent.
AI-generated code can contain errors. The code has **not** necessarily been reviewed line by
line, and the automated tests cover only some cases. Neither guarantees correctness,
completeness or security. Check the results yourself before relying on them. Contribution
rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Licenses

Code: [MIT](LICENSE), provided "AS IS". Model weights downloaded through this tool are
third-party artifacts under their own licenses (e.g. Apache-2.0 for Qwen3, Gemma Terms of
Use, Llama 3.2 Community License). This project does not distribute any model weights.
The book *Noise* is cited for its concepts only; this project is not affiliated with its
authors.

**Disclaimer & liability:** see [DISCLAIMER.md](DISCLAIMER.md). In short: it is non-commercial,
comes with no warranty, and liability is limited as far as the law allows. You are
responsible for your own use, data and costs.
