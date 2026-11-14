# Bring Your Own Model

You can test your own local model in the app without writing any Python. Which of the four
ways you use depends on what you already have:

| You have | Use |
|---|---|
| a model name from ollama.com | [1. Name from ollama.com](#1-name-from-ollamacom) |
| a GGUF model on Hugging Face | [2. Model from Hugging Face](#2-model-from-hugging-face) |
| a `.gguf` file on disk, for example your own fine-tune | [3. Model file](#3-model-file-gguf) |
| a server that is already running (LM Studio, llama.cpp, vLLM, LocalAI and others) | [4. Model server](#4-a-model-server-that-is-already-running) |

The first three need Ollama. Install it with `bash scripts/install_ollama.sh`. The script
fetches a pinned version, verifies its checksum and makes Ollama listen on `127.0.0.1` only.

Everything below happens in the tab **More models**, in the section *Connect a model of
your own*.

## What "local" means

In GitHub Codespaces, "local" is the Codespace itself: a virtual machine with 2 CPU cores,
8 GB RAM and, by default, no GPU. A model that runs on your laptop, in LM Studio for
example, cannot be reached from a Codespace in the cloud.

On your own machine, open the repository with **VS Code → Dev Containers: Reopen in
Container**. Servers that run on the host are then reachable at
`http://host.docker.internal:<port>`.

## 1. Name from ollama.com

Open the sub-tab **Name from ollama.com**, type a name such as `qwen3:1.7b`, confirm that
you checked the license and the size, and press **Download**.

From the command line: `bash scripts/pull_starter_model.sh qwen3:1.7b`

## 2. Model from Hugging Face

Open the sub-tab **Model from Hugging Face** and paste `hf.co/<user>/<repo>` or
`hf.co/<user>/<repo>:<quantization>`, for example
`hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M`. If you leave out the quantization,
Ollama takes `Q4_K_M` when the repository has it.

## 3. Model file (.gguf)

Files up to 8 GB can be uploaded in the sub-tab **Model file (.gguf)**. For larger files,
and for anything you want to script, use the command line:

```bash
bash scripts/import_gguf.sh path/to/my-model.gguf my-finetune            # template from GGUF metadata
bash scripts/import_gguf.sh path/to/my-model.gguf my-finetune --template chatml
bash scripts/import_gguf.sh base.gguf my-lora --adapter my-lora-adapter.gguf
```

The chat templates are in `models/modelfiles/` (`chatml`, `llama3`, `gemma`, `mistral`). If
yours is missing, add a `.tmpl` file there. A wrong template is the most common reason for
nonsense output, so try another one if the answers look odd.

## 4. A model server that is already running

Any server that speaks the OpenAI Chat Completions API will work. These are the usual
default addresses. Check your server's documentation to be sure.

| Server | Base URL |
|---|---|
| LM Studio | `http://127.0.0.1:1234/v1` |
| llama.cpp `llama-server` | `http://127.0.0.1:8080/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |
| LocalAI | `http://127.0.0.1:8080/v1` |
| Ollama (OpenAI mode) | `http://127.0.0.1:11434/v1` |

### With the form in the app

Open the sub-tab **A model server that is already running**. Pick the server type, give the
model a short name of your choice, and enter the model name the server expects and its
address. Press **Test the connection**. Then choose **Use until I close the app** or **Save
permanently**, which writes the entry to `models/models.toml`.

### With the config file

```bash
cp models/models.example.toml models/models.toml   # git-ignored
```

```toml
[[model]]
id           = "llamacpp-ft"
display_name = "My fine-tune"
provider     = "openai_compatible"   # or "ollama"
model        = "my-finetune"
api_base     = "http://127.0.0.1:8080/v1"
api_key_env  = ""                    # NAME of an env var of its own (e.g. MY_SERVER_KEY), never the key itself
supports_json_mode = false
max_concurrency    = 1
```

Then click **Reload the file**. If an entry is invalid, the app names the field and the
reason and still loads the other entries.

### From the command line

```bash
python -m scripts.check_endpoint --base-url http://127.0.0.1:8080/v1 --model my-finetune
python -m scripts.check_endpoint --ollama gemma3:270m
python -m scripts.check_endpoint --registry-id llamacpp-ft
```

### What the connection test checks

| Check | Why it matters |
|---|---|
| A completion works, and how long it takes | Basic connectivity. The time feeds the duration estimate. |
| The system role is accepted | If not, the background instruction is merged into the question. |
| The seed is reproducible | Two seeded calls at temperature 1.0 return identical text, so runs can be repeated exactly. |
| JSON mode | Without it the app relies on its format instructions and tolerant parsing. |
| Logprobs | Fills the `token_prob` column. |

The outcome of the test is saved with the settings of each run, so the export documents
what the model actually supported.

## Security rules

These are the intended rules (best effort, no guarantee, see [DISCLAIMER.md](../DISCLAIMER.md)):

- Custom URLs are limited to loopback, private networks and `host.docker.internal`.
  Public hosts need `allow_public_endpoint = true`. Your prompts then leave the machine.
- Cloud metadata addresses (`169.254.x.x`) are designed to be blocked, also when written
  as IPv6 (`::ffff:169.254.169.254`), as a hex or decimal number, or as a host name that
  resolves to them. Write IP addresses as four decimal numbers, e.g. `192.168.1.20`.
- `api_key_env` must name a variable that belongs to this server, for example
  `MY_SERVER_KEY`. The keys of other services (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and variables starting with `GITHUB_`, `GH_`,
  `AWS_`, `AZURE_`, `GOOGLE_`, `GCP_` or `CODESPACE`) are not sent to custom endpoints.
- A key is sent to a public host over `https://` only. Plain `http://` with a key works
  for local and private-network servers.
- Redirects are not followed, and proxy settings of the environment (`HTTP_PROXY`, ...)
  are ignored for local servers.
- Model names are validated; shell commands are designed not to interpolate user input.
- Never expose port 11434 (Ollama has no authentication).
- You are responsible for the licenses of the weights you download or import.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `Connection refused` | The server is not running, the port is wrong, or (in Codespaces) the server runs on your laptop. |
| `404 model not found` | The model name is not the one the server expects. The connection test lists the names it found. |
| Nonsense or endless output | The chat template is wrong. Import again with `--template`. |
| Many answers in the wrong form | Small models tend to ignore formats. Try a larger model, JSON mode, or the rating or yes/no format. |
| Out of memory, or the process is killed | Use a smaller or more strongly quantized model (Q4) and close other models. |
| Very slow | The model runs on the CPU. Use fewer repetitions, a model of at most 3B parameters, or a 4-core Codespace. |
| `qwen3` shows `<think>` text | Nothing to do. The app removes reasoning blocks before it reads the answer. |
