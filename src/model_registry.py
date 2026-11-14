"""Unified model registry: cloud, local Ollama, custom (BYOM) and mock models.

Custom models live in ``models/models.toml`` (git-ignored). The file is validated
entry by entry so one broken entry never breaks the rest.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.config import CLOUD_MODELS, CUSTOM_MODELS_FILE, PROVIDER_KEY_ENV
from src.model_spec import ModelSource, ModelSpec
from src.security import (
    CUSTOM_ID_RE,
    UrlPolicyError,
    check_endpoint_url,
    is_private_host,
    is_valid_model_name,
    mask_secrets,
)

_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
# Variables that hold credentials for other services (GitHub, cloud accounts, Codespaces).
_RESERVED_ENV_PREFIXES = ("GITHUB_", "GH_", "AWS_", "AZURE_", "GOOGLE_", "GCP_", "CODESPACE")


class CustomModelSpec(BaseModel):
    """One ``[[model]]`` entry of models/models.toml."""

    id: str
    display_name: str | None = None
    provider: Literal["ollama", "openai_compatible"] = "openai_compatible"
    model: str
    api_base: str
    api_key_env: str = ""
    allow_public_endpoint: bool = False
    supports_seed: bool = True
    supports_json_mode: bool = False
    supports_logprobs: bool = False
    max_concurrency: int = Field(default=1, ge=1, le=32)
    timeout_s: float = Field(default=120.0, gt=0, le=3600)
    system_prompt_supported: bool = True
    license: str | None = None
    notes: str | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not CUSTOM_ID_RE.fullmatch(v):
            raise ValueError("id must be lowercase letters, digits, '-' or '_' (max 48 chars)")
        return v

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        if not is_valid_model_name(v):
            raise ValueError("model name contains invalid characters")
        return v

    @field_validator("api_base")
    @classmethod
    def _base(cls, v: str) -> str:
        return v.strip()  # the stored value is the one that was checked

    @field_validator("api_key_env")
    @classmethod
    def _env(cls, v: str) -> str:
        if v and not _ENV_NAME_RE.fullmatch(v):
            raise ValueError("api_key_env must be the NAME of an environment variable "
                             "(e.g. MY_SERVER_KEY), never the key itself")
        return v

    @field_validator("default_params")
    @classmethod
    def _params(cls, v: dict[str, Any]) -> dict[str, Any]:
        forbidden = {"api_key", "api_base", "messages", "model", "base_url"}
        bad = forbidden & set(v)
        if bad:
            raise ValueError(f"default_params may not override {sorted(bad)}")
        return v

    @model_validator(mode="after")
    def _key_destination(self) -> CustomModelSpec:
        """An API key may not be sent to a place it was not made for, or in clear text."""
        if not self.api_key_env:
            return self
        name = self.api_key_env
        if name in PROVIDER_KEY_ENV.values() or name.startswith(_RESERVED_ENV_PREFIXES):
            raise ValueError(
                f"api_key_env '{name}' holds a key for another service and is not sent to "
                "custom endpoints. Store this server's key in a variable of its own "
                "(e.g. MY_SERVER_KEY).")
        parsed = urlparse(self.api_base.strip())
        if parsed.scheme != "https" and not is_private_host(parsed.hostname or ""):
            raise ValueError("an API key is only sent to a public endpoint over https://")
        return self

    def checked_api_base(self) -> str:
        return check_endpoint_url(self.api_base, allow_public=self.allow_public_endpoint)

    def to_model_spec(self) -> ModelSpec:
        base = self.checked_api_base()
        if self.provider == "ollama":
            model_route = f"ollama_chat/{self.model}"
            base = re.sub(r"/v1/?$", "", base)
        else:
            model_route = f"openai/{self.model}"
        return ModelSpec(
            id=f"custom/{self.id}",
            display_name=(self.display_name or self.id) + " (custom)",
            source=ModelSource.CUSTOM,
            model_route=model_route,
            api_base=base,
            api_key_env=self.api_key_env or None,
            supports_seed=self.supports_seed,
            supports_json_mode=self.supports_json_mode,
            supports_logprobs=self.supports_logprobs,
            system_prompt_supported=self.system_prompt_supported,
            max_concurrency=self.max_concurrency,
            timeout_s=self.timeout_s,
            default_params=self.default_params,
            license=self.license,
            notes=self.notes,
        )


@dataclass
class RegistryError:
    entry: str
    message: str

    def __str__(self) -> str:
        return f"{self.entry}: {self.message}"


def load_custom_registry(path: Path = CUSTOM_MODELS_FILE) -> tuple[list[CustomModelSpec],
                                                                     list[RegistryError]]:
    """Parse models.toml. Returns (valid entries, friendly error messages)."""
    if not path.is_file():
        return [], []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [], [RegistryError(path.name, f"TOML syntax error: {exc}")]
    entries = data.get("model", [])
    if not isinstance(entries, list):
        return [], [RegistryError(path.name, "use [[model]] tables (array of tables)")]
    specs: list[CustomModelSpec] = []
    errors: list[RegistryError] = []
    seen: set[str] = set()
    for i, raw in enumerate(entries, start=1):
        label = f"[[model]] #{i} ({raw.get('id', 'no id')})" if isinstance(raw, dict) \
            else f"[[model]] #{i}"
        try:
            spec = CustomModelSpec.model_validate(raw)
            spec.checked_api_base()
        except ValidationError as exc:
            msgs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            errors.append(RegistryError(label, msgs))
            continue
        except UrlPolicyError as exc:
            errors.append(RegistryError(label, f"api_base: {exc}"))
            continue
        if spec.id in seen:
            errors.append(RegistryError(label, f"duplicate id '{spec.id}'"))
            continue
        seen.add(spec.id)
        specs.append(spec)
    return specs, errors


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_toml_value(str(k))} = {_toml_value(x)}"
                                for k, x in v.items()) + " }"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    # TOML forbids raw control characters in strings; one of them would make the whole
    # file unreadable.
    s = "".join(f"\\u{ord(ch):04X}" if ord(ch) < 0x20 or ord(ch) == 0x7F else ch for ch in s)
    return f'"{s}"'


def _entry_lines(spec: CustomModelSpec) -> list[str]:
    lines = ["", "[[model]]"]
    for key, value in spec.model_dump().items():
        if value is None or value == {} or (key == "api_key_env" and not value):
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return lines


_REGISTRY_HEADER = ("# Your own models (git-ignored). See models/models.example.toml and\n"
                    "# docs/BRING_YOUR_OWN_MODEL.md. Never put API keys in this file.\n")


def append_custom_model(spec: CustomModelSpec, path: Path = CUSTOM_MODELS_FILE) -> None:
    """Append a validated entry to models.toml (creating the file if needed)."""
    existing, _ = load_custom_registry(path)
    if any(e.id == spec.id for e in existing):
        raise ValueError(f"A custom model with id '{spec.id}' already exists.")
    spec.checked_api_base()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "" if path.exists() else _REGISTRY_HEADER
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(header + "\n".join(_entry_lines(spec)) + "\n")


def remove_custom_model(model_id: str, path: Path = CUSTOM_MODELS_FILE) -> bool:
    """Rewrite models.toml without the given entry. Returns True if something was removed."""
    specs, errors = load_custom_registry(path)
    if errors:
        raise ValueError("Fix the errors in models.toml before removing entries.")
    keep = [s for s in specs if s.id != model_id]
    if len(keep) == len(specs):
        return False
    text = _REGISTRY_HEADER + "".join("\n".join(_entry_lines(s)) + "\n" for s in keep)
    # Write the new file next to the old one and swap it in, so that an interruption
    # never leaves the registry half-written or missing.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


# --- Unified model list ------------------------------------------------------

def cloud_model_specs(api_keys: dict[str, str]) -> list[ModelSpec]:
    """Cloud models whose provider key is available (session or environment)."""
    out = []
    for m in CLOUD_MODELS:
        env = PROVIDER_KEY_ENV.get(m.provider, "")
        if api_keys.get(m.provider) or os.environ.get(env):
            out.append(ModelSpec(id=m.route_id, display_name=m.display_name,
                                 source=ModelSource.CLOUD, model_route=m.route_id,
                                 provider=m.provider, supports_seed=m.supports_seed,
                                 supports_json_mode=m.provider in ("openai", "gemini"),
                                 max_concurrency=16))
    return out


def custom_cloud_spec(route_id: str) -> ModelSpec:
    """Any other cloud model id typed by the user (e.g. 'openrouter/qwen/qwen3-8b')."""
    if not is_valid_model_name(route_id) or "/" not in route_id:
        raise ValueError("Use the format 'provider/model-name'.")
    provider = route_id.split("/", 1)[0]
    if provider not in PROVIDER_KEY_ENV:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {sorted(PROVIDER_KEY_ENV)}")
    return ModelSpec(id=route_id, display_name=route_id, source=ModelSource.CLOUD,
                     model_route=route_id, provider=provider,
                     supports_json_mode=provider in ("openai", "gemini"), max_concurrency=16)


def _http_get_json(url: str, key: str | None) -> dict[str, Any]:
    import httpx

    headers = {"Authorization": f"Bearer {key}"} if key and key != "not-needed" else {}
    # No redirects (the key would follow them) and no proxy for local servers.
    r = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=False,
                  trust_env=not is_private_host(urlparse(url).hostname or ""))
    r.raise_for_status()
    return r.json()



# --- Capability probe --------------------------------------------------------

@dataclass
class CapabilityReport:
    reachable: bool = False
    completion_ok: bool = False
    latency_s: float | None = None
    seed_deterministic: bool | None = None
    json_mode: bool | None = None
    logprobs: bool | None = None
    system_role: bool | None = None
    listed_models: list[str] = field(default_factory=list)
    model_listed: bool | None = None
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ("messages", "listed_models")}


async def probe(spec: ModelSpec, completion_fn: Any = None, http_get: Any = None,
                timeout_s: float = 60.0) -> CapabilityReport:
    """Send a handful of tiny requests and report what the endpoint actually supports."""
    from src.llm_runner import _default_completion, _resolve_api_key

    fn = completion_fn or _default_completion
    rep = CapabilityReport()
    key = _resolve_api_key(spec, {}) or (
        "not-needed" if spec.source == ModelSource.CUSTOM else None)
    base_kwargs: dict[str, Any] = {"model": spec.model_route, "max_tokens": 16,
                                   "timeout": timeout_s, "num_retries": 0}
    if spec.api_base:
        base_kwargs["api_base"] = spec.api_base
    if key:
        base_kwargs["api_key"] = key

    if spec.api_base and spec.model_route.startswith("openai/"):
        try:  # optional: GET /models
            getter = http_get or _http_get_json
            data = await asyncio.to_thread(getter, spec.api_base.rstrip("/") + "/models", key)
            rep.listed_models = [str(m.get("id", "")) for m in data.get("data", [])][:50]
            rep.model_listed = spec.model_route.split("/", 1)[1] in rep.listed_models
            rep.reachable = True
            if rep.listed_models and not rep.model_listed:
                rep.messages.append("Model name not found in /models. Available: "
                                    + ", ".join(rep.listed_models[:10]))
        except Exception as exc:
            rep.messages.append(f"GET /models failed (optional): {mask_secrets(exc)[:150]}")

    async def ask(**extra: Any) -> Any:
        return await fn(**{**base_kwargs, **extra})

    user = [{"role": "user", "content": "Reply with the single word: OK"}]
    try:
        t0 = time.perf_counter()
        await ask(messages=user, temperature=0.0)
        rep.latency_s = round(time.perf_counter() - t0, 2)
        rep.reachable = rep.completion_ok = True
    except Exception as exc:
        rep.messages.append(f"Basic completion failed: {mask_secrets(exc)[:300]}")
        return rep

    try:
        await ask(messages=[{"role": "system", "content": "Be brief."}, *user], temperature=0.0)
        rep.system_role = True
    except Exception:
        rep.system_role = False

    try:
        prompt = [{"role": "user", "content": "Write one random English word."}]
        a = await ask(messages=prompt, temperature=1.0, seed=1234, max_tokens=8)
        b = await ask(messages=prompt, temperature=1.0, seed=1234, max_tokens=8)
        rep.seed_deterministic = a.choices[0].message.content == b.choices[0].message.content
    except Exception:
        rep.seed_deterministic = None

    try:
        r = await ask(messages=[{"role": "user", "content": 'Return {"answer": 1} as JSON.'}],
                      temperature=0.0, response_format={"type": "json_object"}, max_tokens=20)
        rep.json_mode = "{" in (r.choices[0].message.content or "")
    except Exception:
        rep.json_mode = False

    try:
        r = await ask(messages=user, temperature=0.0, logprobs=True)
        lp = getattr(r.choices[0], "logprobs", None)
        rep.logprobs = bool(lp and getattr(lp, "content", None))
    except Exception:
        rep.logprobs = False

    if rep.seed_deterministic is False:
        rep.messages.append("Seed is not honoured: identical seeded calls gave different "
                            "outputs. Occasion noise is still measurable, but runs are not "
                            "exactly reproducible.")
    return rep


def apply_probe(spec: ModelSpec, rep: CapabilityReport) -> ModelSpec:
    """Return a copy of ``spec`` adapted to the probed capabilities."""
    update: dict[str, Any] = {"probed_capabilities": rep.as_dict()}
    if rep.json_mode is not None:
        update["supports_json_mode"] = rep.json_mode
    if rep.logprobs is not None:
        update["supports_logprobs"] = rep.logprobs
    if rep.system_role is False:
        update["system_prompt_supported"] = False
    return spec.model_copy(update=update)

