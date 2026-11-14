"""Minimal, dependency-light LLM client (replaces LiteLLM).

Two wire protocols cover every supported model:

* **OpenAI Chat Completions** (``POST {base}/chat/completions``): OpenAI, OpenRouter,
  Google Gemini (official OpenAI-compatible endpoint), Ollama (``/v1``) and every
  OpenAI-compatible server (LM Studio, llama.cpp, vLLM, LocalAI, ...).
* **Anthropic Messages** (``POST {base}/v1/messages``).

Model ids keep the ``provider/model`` convention, e.g. ``openai/gpt-4o-mini``,
``anthropic/claude-...``, ``gemini/gemini-...``, ``openrouter/vendor/model``,
``ollama_chat/gemma3:270m`` (custom endpoints use ``openai/<name>`` + ``api_base``).

API keys are only ever sent in request headers - never in URLs - and every error
message is masked before it leaves this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from src.security import is_private_host, mask_secrets

OPENAI_BASE = "https://api.openai.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_BASE = "https://api.anthropic.com"
# Pinned on purpose: API version 2026-10-21 removed `temperature`/`top_p`/`top_k`
# (requests containing them are rejected). Noise experiments need temperature control,
# and Anthropic preserves the parameters of an existing version.
ANTHROPIC_VERSION = "2023-06-01"
OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434"
# A chat answer is a few kilobytes. Anything far beyond that is a broken or hostile server.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# Parameters each protocol accepts. Everything else is silently dropped (like the
# former litellm.drop_params) and the actually sent parameters are what we log.
_OPENAI_PARAMS = {"temperature", "max_tokens", "seed", "response_format", "logprobs",
                  "top_p", "top_k", "presence_penalty", "frequency_penalty", "stop",
                  "max_completion_tokens"}
_ANTHROPIC_PARAMS = {"temperature", "max_tokens", "top_p", "top_k", "stop_sequences"}
_GEMINI_DROP = {"seed", "logprobs"}  # not reliably supported by the compat endpoint

# --- Settings a model refuses ---------------------------------------------------------
# Models differ in which optional settings they accept, and providers change this per
# model (e.g. "`temperature` is deprecated for this model" on newer Claude models,
# "Unsupported value: 'temperature'" on OpenAI reasoning models). Instead of a fragile
# per-model list, a rejected request (HTTP 400/422) is repeated without the setting the
# error names, and the model is remembered so later calls leave it out right away.
_ADAPT_STATUS = {400, 422}
_OPTIONAL_KEYS = ("temperature", "top_p", "top_k", "seed", "response_format", "logprobs",
                  "presence_penalty", "frequency_penalty", "stop", "stop_sequences")
# Wording providers use when a setting is not accepted but its name is not given.
_REJECTION_HINTS = ("deprecated", "not supported", "unsupported", "not permitted",
                    "not allowed", "extra inputs", "unknown parameter", "unrecognized",
                    "invalid parameter", "does not support", "is not available")
_MAX_ADAPTATIONS = 4
_UNSUPPORTED: dict[tuple[str, str], set[str]] = {}


def reset_unsupported_cache() -> None:
    """Forget which settings models refused (used by tests)."""
    _UNSUPPORTED.clear()


def unsupported_settings(model_id: str, api_base: str | None = None) -> set[str]:
    """Settings this model refused earlier in this app session."""
    route = resolve_route(model_id, api_base)
    return set(_UNSUPPORTED.get((route.base_url, route.model), set()))


def _rejected_keys(message: str, body: dict[str, Any], protocol: str) -> list[str]:
    """Optional request settings that a provider error message rejects."""
    msg = message.lower()
    named = [k for k in _OPTIONAL_KEYS if k in body and re.search(rf"\b{k}\b", msg)]
    if protocol == "openai" and "max_tokens" in body and "max_completion_tokens" in msg:
        named.append("max_tokens")  # renamed, not removed (see _drop_key)
    if named:
        return named
    if any(h in msg for h in _REJECTION_HINTS):  # unnamed: fall back to the bare minimum
        return [k for k in _OPTIONAL_KEYS if k in body]
    return []


def _drop_key(body: dict[str, Any], key: str) -> dict[str, Any]:
    out = {k: v for k, v in body.items() if k != key}
    if key == "max_tokens":  # keep the length limit under the newer name
        out["max_completion_tokens"] = body["max_tokens"]
    return out


class ProviderError(Exception):
    """HTTP or protocol error with retry metadata (masked message)."""

    def __init__(self, message: str, status_code: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


# --- Normalised response objects (attribute-compatible with the old code paths) ---

@dataclass
class TokenLogprob:
    token: str
    logprob: float | None


@dataclass
class Logprobs:
    content: list[TokenLogprob] = field(default_factory=list)


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message
    finish_reason: str | None = None
    logprobs: Logprobs | None = None


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class ChatResponse:
    choices: list[Choice]
    model: str | None = None
    usage: Usage | None = None
    sent_params: dict[str, Any] = field(default_factory=dict)
    # Settings that were requested but left out because the model refuses them.
    dropped_params: list[str] = field(default_factory=list)


# --- Routing ---------------------------------------------------------------

@dataclass(frozen=True)
class Route:
    protocol: str  # "openai" | "anthropic"
    base_url: str
    model: str
    provider: str


def resolve_route(model_id: str, api_base: str | None = None) -> Route:
    """Map ``provider/model`` (+ optional api_base override) to protocol and URL."""
    if "/" not in model_id:
        raise ProviderError(f"Model id '{model_id}' must look like 'provider/model'.")
    provider, name = model_id.split("/", 1)
    base = api_base.rstrip("/") if api_base else None
    if provider == "openai":
        return Route("openai", base or OPENAI_BASE, name, provider)
    if provider == "openrouter":
        return Route("openai", base or OPENROUTER_BASE, name, provider)
    if provider == "gemini":
        return Route("openai", base or GEMINI_OPENAI_BASE, name, provider)
    if provider in ("ollama", "ollama_chat"):
        root = (base or OLLAMA_DEFAULT_BASE).removesuffix("/v1")
        return Route("openai", root + "/v1", name, "ollama")
    if provider == "anthropic":
        return Route("anthropic", base or ANTHROPIC_BASE, name, provider)
    raise ProviderError(f"Unsupported provider '{provider}'. Use openai, anthropic, gemini, "
                        "openrouter, ollama_chat, or openai/<name> with a custom api_base.")


# --- Request builders ------------------------------------------------------

def _openai_request(route: Route, messages: list[dict[str, str]], api_key: str | None,
                    params: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    sent = {k: v for k, v in params.items() if k in _OPENAI_PARAMS and v is not None}
    if route.provider == "gemini":
        sent = {k: v for k, v in sent.items() if k not in _GEMINI_DROP}
    if sent.pop("logprobs", None):
        sent["logprobs"] = True
    body = {"model": route.model, "messages": messages, "stream": False, **sent}
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    return f"{route.base_url}/chat/completions", headers, body


def _anthropic_request(route: Route, messages: list[dict[str, str]], api_key: str | None,
                       params: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    sent = {k: v for k, v in params.items() if k in _ANTHROPIC_PARAMS and v is not None}
    sent.setdefault("max_tokens", 512)
    if "temperature" in sent:  # Anthropic accepts 0..1 only
        sent["temperature"] = max(0.0, min(1.0, float(sent["temperature"])))
    body: dict[str, Any] = {"model": route.model, "messages": turns, **sent}
    if system:
        body["system"] = system
    headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
    if api_key:
        headers["x-api-key"] = api_key
    return f"{route.base_url}/v1/messages", headers, body


# --- Response parsers ------------------------------------------------------

def _content_text(content: Any) -> str:
    """Message content as plain text.

    Some OpenAI-compatible servers return a list of typed parts instead of a string.
    Only text parts are kept; reasoning / thinking parts are not part of the answer.
    """
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content
                       if isinstance(p, dict) and p.get("type") in (None, "text", "output_text"))
    return str(content or "")


def _parse_openai(data: dict[str, Any]) -> ChatResponse:
    choices = []
    for c in data.get("choices") or []:
        msg = c.get("message") or {}
        lp = None
        raw_lp = (c.get("logprobs") or {}).get("content") if isinstance(c.get("logprobs"),
                                                                     dict) else None
        if raw_lp:
            lp = Logprobs([TokenLogprob(str(t.get("token", "")), t.get("logprob"))
                           for t in raw_lp])
        choices.append(Choice(Message(_content_text(msg.get("content"))),
                              c.get("finish_reason"), lp))
    if not choices:
        raise ProviderError("Response contained no choices.")
    u = data.get("usage") or {}
    return ChatResponse(choices, data.get("model"),
                        Usage(u.get("prompt_tokens"), u.get("completion_tokens")))


def _parse_anthropic(data: dict[str, Any]) -> ChatResponse:
    text = "".join(b.get("text", "") for b in data.get("content") or []
                   if b.get("type") == "text")
    u = data.get("usage") or {}
    return ChatResponse([Choice(Message(text), data.get("stop_reason"))], data.get("model"),
                        Usage(u.get("input_tokens"), u.get("output_tokens")))


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", data)
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except (ValueError, AttributeError):
        msg = resp.text[:300]
    return f"HTTP {resp.status_code}: {msg}"


# --- Public API --------------------------------------------------------------

async def acompletion(
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 120.0,
    client: httpx.AsyncClient | None = None,
    **params: Any,
) -> ChatResponse:
    """Send one stateless chat request and return a normalised response.

    Unknown parameters are dropped per protocol; ``ChatResponse.sent_params`` records
    what was actually sent. Settings the model refuses (HTTP 400/422 naming them) are
    left out and listed in ``ChatResponse.dropped_params``. ``num_retries`` is accepted
    and ignored (retries happen in the runner).
    """
    params.pop("num_retries", None)
    route = resolve_route(model, api_base)
    builder = _anthropic_request if route.protocol == "anthropic" else _openai_request
    url, headers, body = builder(route, messages, api_key, params)
    secrets = [api_key] if api_key else []
    cache_key = (route.base_url, route.model)
    dropped: list[str] = []
    for key in sorted(_UNSUPPORTED.get(cache_key, set())):  # refused earlier: skip it
        if key in body:
            body = _drop_key(body, key)
            dropped.append(key)
    owns = client is None
    # No redirects (the API key would follow them). For local / private servers the proxy
    # variables of the environment are ignored, so prompts never leave through a proxy.
    local = is_private_host(urlparse(route.base_url).hostname or "")
    c = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0),
                                    follow_redirects=False, trust_env=not local)
    try:
        for _ in range(_MAX_ADAPTATIONS + 1):
            try:
                resp = await c.post(url, headers=headers, content=json.dumps(body))
            except httpx.TimeoutException as exc:
                raise ProviderError(f"Timeout after {timeout:.0f} s calling {route.provider}",
                                    status_code=408) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(mask_secrets(f"Connection error ({route.provider}): {exc}",
                                                 secrets)) from exc
            if resp.status_code not in _ADAPT_STATUS:
                break
            keys = [k for k in _rejected_keys(_error_message(resp), body, route.protocol)
                    if k not in dropped]
            if not keys:
                break
            for key in keys:  # the model refuses this setting: send the request without it
                body = _drop_key(body, key)
                dropped.append(key)
                _UNSUPPORTED.setdefault(cache_key, set()).add(key)
    finally:
        if owns:
            await c.aclose()
    if resp.status_code >= 400:
        raise ProviderError(mask_secrets(_error_message(resp), secrets), resp.status_code,
                            _retry_after(resp))
    if len(resp.content) > MAX_RESPONSE_BYTES:
        raise ProviderError(f"Response is larger than {MAX_RESPONSE_BYTES // 1024**2} MB "
                            "and was discarded.", resp.status_code)
    try:
        data = resp.json()
    except ValueError as exc:
        raise ProviderError("Provider returned invalid JSON.", resp.status_code) from exc
    parsed = _parse_anthropic(data) if route.protocol == "anthropic" else _parse_openai(data)
    parsed.sent_params = {k: v for k, v in body.items() if k not in ("messages", "system")}
    parsed.dropped_params = dropped
    return parsed

