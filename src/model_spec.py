"""Unified description of a model that the runner can call (cloud, Ollama, custom, mock)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ModelSource(StrEnum):
    CLOUD = "cloud"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    MOCK = "mock"


class ModelSpec(BaseModel):
    """Everything the runner needs to call one model. Never contains an API key."""

    id: str  # unique id shown in the UI and stored in df_results.model
    display_name: str
    source: ModelSource
    model_route: str = ""  # "provider/model", e.g. "openai/gpt-4o-mini", "ollama_chat/gemma3:270m"
    api_base: str | None = None
    provider: str | None = None  # key into config.PROVIDER_KEY_ENV for cloud models
    api_key_env: str | None = None  # custom models: NAME of an env var holding the key
    supports_seed: bool = True
    supports_json_mode: bool = True
    supports_logprobs: bool = False
    system_prompt_supported: bool = True
    max_concurrency: int = 4
    timeout_s: float = 120.0
    default_params: dict[str, Any] = Field(default_factory=dict)
    license: str | None = None
    notes: str | None = None
    probed_capabilities: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.source in (ModelSource.OLLAMA, ModelSource.CUSTOM, ModelSource.MOCK)
