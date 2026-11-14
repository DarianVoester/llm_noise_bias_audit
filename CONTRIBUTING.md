# Contributing

This is a small research and education project. Contributions are welcome. Please keep pull
requests small and include tests.

## Setup

```bash
pip install --require-hashes -r requirements.txt
pip install -r requirements-dev.txt
ruff check .
pytest
streamlit run app.py
```

The Codespace installs only the Python extension, which is enough to use and debug the
app. If you edit code, the `charliermarsh.ruff` extension shows lint problems while you
type; `ruff check .` does the same from the terminal.

## Rules

- Code, UI text, docstrings and docs are written in English.
- Use type hints, keep the code clean under `ruff`, and add tests for new features.
- Texts in the app are read by people without a technical background. Use everyday words,
  explain a technical term the first time it appears, and say what the user should do next.
  The tab names live in `ui/state.py`; refer to them through those constants.
- Never commit API keys, `.env` files, `models/models.toml` or model weights.
- To change a dependency, edit `requirements.in` and regenerate `requirements.txt` with the
  `uv pip compile` command at the top of `requirements.in`. That command enforces the 7-day
  rule through `--exclude-newer`: never bump to a release younger than 7 days. Explain the
  bump in the pull request.
- Changes to `src/evaluator.py`, `src/sandbox_child.py` or `src/security.py` need new
  tests in `tests/test_sandbox_security.py` or `tests/test_security.py`. For the sandbox,
  add the attack to `STATIC_ATTACKS` (AST validator) and, if it concerns files, network or
  processes, to `HOOK_ATTACKS` (audit hook of the child process).
- Do not change `[server] address` in `.streamlit/config.toml` away from `127.0.0.1`. The
  app has no login.
- GitHub Actions in `.github/workflows/` are pinned to a commit hash with the version as a
  comment. Keep that form; Dependabot updates both.
- `requirements-dev.txt` is not hash-pinned yet. The steps to pin it are written at the
  top of that file.

## AI-assisted contributions

This project itself was specified with Google Gemini and Anthropic Claude and
implemented with an AI coding agent. AI-assisted contributions
are welcome under these conditions:

1. Say so in the PR description (which tool, for what part).
2. As a contributor, read, understand and test the code you submit.
3. Do not submit code you cannot explain or maintain.

Each contributor is responsible for their own contribution. Merging a pull request is
**not** a confirmation by the maintainer that the code was reviewed line by line or is
correct, secure or fit for any purpose, and the maintainer does not take on liability for
it. See [DISCLAIMER.md](DISCLAIMER.md).

## Statistical presets

New analyses go into `src/presets.py`. Each one needs a `plain_english_summary`, a
`kahneman_concept`, a `how_to_interpret` text, an everyday question in `PRESET_QUESTIONS`
(`src/guidance.py`) and a test that runs it on practice data through the sandbox.
