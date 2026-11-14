"""Central configuration: constants, defaults, provider and local model catalogs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Run defaults -----------------------------------------------------------
DEFAULT_N_WINDOWS = 10
MAX_N_WINDOWS = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 32
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_DESIGN_SEED = 42

# Confirmation thresholds for the pre-run guard
CONFIRM_ABOVE_CALLS = 200
CONFIRM_ABOVE_COST_USD = 1.0

# --- Sandbox ----------------------------------------------------------------
SANDBOX_TIMEOUT_S = 5.0
SANDBOX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GiB address space
SANDBOX_CPU_SECONDS = 5

# --- Statistics -------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 12345
PARSE_FAILURE_GAP_WARN = 0.10  # warn if parse failure rates differ by > 10 pp

# --- Local models -----------------------------------------------------------
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
CUSTOM_MODELS_FILE = PROJECT_ROOT / os.environ.get(
    "LLM_NOISE_MODELS_FILE", "models/models.toml"
)
CUSTOM_MODELS_EXAMPLE_FILE = PROJECT_ROOT / "models" / "models.example.toml"
MODELFILE_TEMPLATE_DIR = PROJECT_ROOT / "models" / "modelfiles"
CODESPACES_FREE_STORAGE_GB = 15.0
GGUF_UPLOAD_LIMIT_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class CloudModel:
    """A pre-configured cloud model (called via src/providers.py)."""

    route_id: str
    display_name: str
    provider: str  # key into PROVIDER_KEY_ENV
    supports_seed: bool = True


# Provider name -> environment variable holding its API key.
PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Small, editable starter list. Users can type any other provider/model id in the UI.
CLOUD_MODELS: list[CloudModel] = [
    CloudModel("openai/gpt-4o-mini", "OpenAI GPT-4o mini", "openai"),
    CloudModel("openai/gpt-4o", "OpenAI GPT-4o", "openai"),
    # Claude Haiku 4.5 is chosen on purpose: Claude 4.7 and later reject a non-default
    # `temperature`, which noise experiments need.
    CloudModel("anthropic/claude-haiku-4-5-20251001", "Anthropic Claude Haiku 4.5", "anthropic",
               supports_seed=False),
    CloudModel("gemini/gemini-3.6-flash", "Google Gemini 3.6 Flash", "gemini"),
    CloudModel("openrouter/meta-llama/llama-3.1-8b-instruct", "OpenRouter Llama 3.1 8B",
               "openrouter"),
]


# Static price table: model id -> (USD per 1M input tokens, USD per 1M output tokens).
# APPROXIMATE, NO GUARANTEE - prices change. Used only for the pre-run estimate and the
# cost_usd column. Unknown models show "cost unknown" instead of a guess.
# Last updated: 2026-11-08 (values from the providers' public price lists as known to
# the AI-assisted authoring process, not independently checked; verify before relying on
# them and update via pull request).
MODEL_PRICES_AS_OF = "2026-11-10"
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    # Google lists 0.75 / 3.75 through 2026-12-31 and 1.50 / 7.50 from 2027-01-01.
    "gemini/gemini-3.6-flash": (0.75, 3.75),
    "openrouter/meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
}


@dataclass(frozen=True)
class LocalModelInfo:
    """Curated, CPU-friendly open-weight model available in the Ollama library."""

    tag: str
    display_name: str
    params_b: float
    approx_disk_gb: float
    min_ram_gb: float
    license: str
    license_url: str
    open_source_osi: bool
    notes: str = ""


_GEMMA = ("Gemma Terms of Use", "https://ai.google.dev/gemma/terms", False)
_LLAMA = ("Llama 3.2 Community License", "https://www.llama.com/llama3_2/license/", False)
_QWEN = ("Apache-2.0", "https://ollama.com/library/qwen3", True)

# Tags and sizes verified on ollama.com on 2026-11-08 (see SPEC.md §6.5).
LOCAL_MODEL_CATALOG: list[LocalModelInfo] = [
    LocalModelInfo("gemma3:270m", "Gemma 3 270M", 0.27, 0.29, 1.0, *_GEMMA,
                   "Tiny and fast; good for smoke tests, weak reasoning."),
    LocalModelInfo("qwen3:0.6b", "Qwen3 0.6B", 0.6, 0.52, 1.5, *_QWEN,
                   "Thinking mode by default; <think> blocks are stripped before parsing."),
    LocalModelInfo("gemma3:1b", "Gemma 3 1B", 1.0, 0.81, 2.0, *_GEMMA),
    LocalModelInfo("llama3.2:1b", "Llama 3.2 1B", 1.2, 1.3, 2.5, *_LLAMA),
    LocalModelInfo("qwen3:1.7b", "Qwen3 1.7B", 1.7, 1.4, 3.0, *_QWEN,
                   "Thinking mode by default."),
    LocalModelInfo("llama3.2:3b", "Llama 3.2 3B", 3.2, 2.0, 4.0, *_LLAMA),
    LocalModelInfo("qwen3:4b", "Qwen3 4B", 4.0, 2.5, 5.5, *_QWEN,
                   "Tight on a 2-core / 8 GB Codespace."),
]


@dataclass(frozen=True)
class EndpointPreset:
    """Editable default for OpenAI-compatible local servers (BYOM wizard)."""

    name: str
    api_base: str
    note: str = ""


# Well-known default ports. Editable suggestions only; the capability probe confirms.
ENDPOINT_PRESETS: list[EndpointPreset] = [
    EndpointPreset("Ollama (OpenAI-compatible)", "http://127.0.0.1:11434/v1"),
    EndpointPreset("LM Studio", "http://127.0.0.1:1234/v1"),
    EndpointPreset("llama.cpp llama-server", "http://127.0.0.1:8080/v1"),
    EndpointPreset("vLLM", "http://127.0.0.1:8000/v1"),
    EndpointPreset("LocalAI", "http://127.0.0.1:8080/v1"),
    EndpointPreset("Host machine (local Dev Container only)",
                   "http://host.docker.internal:1234/v1",
                   "Only works when the repo runs in a local Dev Container, not in Codespaces."),
]

DEFAULT_PERSONAS: list[str] = [
    "You are a careful, conservative analyst.",
    "You are an optimistic, growth-minded analyst.",
    "You are a strict, skeptical auditor.",
    "You are a pragmatic generalist.",
]


@dataclass(frozen=True)
class MockModelProfile:
    """Behavioural profile for the mock simulator (all shifts in units of scale range)."""

    name: str
    level_bias: float  # systematic shift (e.g. 0.05 = +5 % of the scale)
    occasion_sd: float  # random scatter per call
    pattern_sd: float  # stable per-case idiosyncrasy
    anchor_sensitivity: float  # 0..1 pull towards the anchor
    framing_shift: float  # shift applied to frame_B
    parse_failure_rate: float
    order_sensitivity: float = 0.03
    persona_sd: float = 0.03
    extra: dict[str, float] = field(default_factory=dict)


MOCK_MODELS: list[MockModelProfile] = [
    MockModelProfile("mock/steady-sage", 0.00, 0.03, 0.02, 0.10, 0.02, 0.01),
    MockModelProfile("mock/anchored-optimist", 0.08, 0.06, 0.04, 0.45, 0.06, 0.03),
    MockModelProfile("mock/noisy-novice", -0.05, 0.14, 0.08, 0.25, 0.10, 0.08),
]
