"""Asynchronous multi-model executor with retries, seeds, cost/ETA guard and mock mode.

Each context window is sent as a completely stateless request (fresh message list,
no conversation history) through the in-house client in ``src/providers.py`` (no
response caching of any kind) and every window gets its own ``seed`` where supported.
No random nonces are injected into the prompt because that would itself be a context
modification.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MODEL_PRICES_USD_PER_MTOK,
    PROVIDER_KEY_ENV,
)
from src.context_engine import ContextWindow
from src.mock_simulator import MOCK_PROFILES, simulate_response
from src.model_spec import ModelSource, ModelSpec
from src.schemas import (
    EMPTY_RESPONSE_ERROR,
    ParseResult,
    SchemaConfig,
    build_repair_message,
    parse_response,
    response_format_payload,
)
from src.security import mask_secrets

DF_COLUMNS = [
    "run_id", "timestamp", "case_id", "model", "model_source", "context_window_id",
    "context_variant", "persona", "system_prompt", "final_prompt", "raw_output",
    "parsed_value_num", "parsed_value_json", "confidence", "token_prob", "parse_success",
    "parse_method", "format_retries", "ground_truth", "schema_type", "scale_min", "scale_max",
    "seed",
    "temperature", "dropped_params", "model_version_returned",
    "finish_reason", "prompt_tokens", "completion_tokens", "cost_usd", "latency_ms", "error",
]

CompletionFn = Callable[..., Awaitable[Any]]
ProgressFn = Callable[[int, int], None]

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class RunParams(BaseModel):
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    concurrency: int = DEFAULT_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    base_seed: int = 0
    use_json_mode: bool = True
    # One follow-up turn ("your reply did not follow the format ...") when a reply cannot be
    # parsed or violates the scale (e.g. 0.42 instead of 42 for percentages).
    repair_format: bool = True
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class RunEstimate:
    calls: int
    est_cost_usd: float
    est_seconds: float
    unknown_cost_models: list[str]


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_run(
    windows: list[ContextWindow],
    models: list[ModelSpec],
    params: RunParams,
    sec_per_call: dict[str, float] | None = None,
) -> RunEstimate:
    """Rough pre-run estimate of calls, cost (cloud only) and wall-clock time."""
    calls = len(windows) * len(models)
    prompt_tokens = sum(_approx_tokens(w.system_prompt + w.user_prompt) for w in windows)
    cost = 0.0
    unknown: list[str] = []
    seconds = 0.0
    for m in models:
        if m.source == ModelSource.CLOUD:
            price = _price_per_token(m.model_route)
            if price is None:
                unknown.append(m.id)
            else:
                in_p, out_p = price
                cost += prompt_tokens * in_p + len(windows) * params.max_tokens * out_p
        per_call = (sec_per_call or {}).get(m.id)
        if per_call is None:
            per_call = 0.01 if m.source == ModelSource.MOCK else (
                8.0 if m.source in (ModelSource.OLLAMA, ModelSource.CUSTOM) else 3.0)
        parallel = max(1, min(params.concurrency, m.max_concurrency))
        seconds = max(seconds, len(windows) * per_call / parallel)
    return RunEstimate(calls, round(cost, 4), round(seconds, 1), unknown)


def _price_per_token(model_id: str) -> tuple[float, float] | None:
    """(input, output) USD per token from the static price table, if known."""
    price = MODEL_PRICES_USD_PER_MTOK.get(model_id)
    if price is None:
        return None
    return price[0] / 1e6, price[1] / 1e6


def _resolve_api_key(model: ModelSpec, api_keys: dict[str, str]) -> str | None:
    import os

    if model.source == ModelSource.CLOUD and model.provider:
        env_name = PROVIDER_KEY_ENV.get(model.provider, "")
        return api_keys.get(model.provider) or os.environ.get(env_name) or None
    if model.api_key_env:
        return os.environ.get(model.api_key_env) or None
    return None


def _messages(model: ModelSpec, window: ContextWindow) -> list[dict[str, str]]:
    if model.system_prompt_supported:
        return [{"role": "system", "content": window.system_prompt},
                {"role": "user", "content": window.user_prompt}]
    return [{"role": "user", "content": f"{window.system_prompt}\n\n{window.user_prompt}"}]


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and value >= 0:
        return min(float(value), 60.0)
    return None


def _is_retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None:
        return code in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "ratelimit", "unavailable"))


def _actual_cost(model: ModelSpec, prompt_tokens: Any, completion_tokens: Any) -> float | None:
    if model.is_local:
        return 0.0
    price = _price_per_token(model.model_route)
    if price is None or not isinstance(prompt_tokens, int) or \
            not isinstance(completion_tokens, int):
        return None
    return round(prompt_tokens * price[0] + completion_tokens * price[1], 8)


def _token_prob(choice: Any) -> float | None:
    """Geometric-mean token probability of the completion, if logprobs were returned."""
    try:
        content = choice.logprobs.content  # OpenAI format
        vals = [t.logprob for t in content if t.logprob is not None]
    except AttributeError:
        return None
    if not vals:
        return None
    return float(math.exp(sum(vals) / len(vals)))


async def _default_completion(**kwargs: Any) -> Any:
    from src.providers import acompletion

    return await acompletion(**kwargs)


def _base_row(params: RunParams, model: ModelSpec, w: ContextWindow, seed: int,
              schema: SchemaConfig) -> dict[str, Any]:
    lo, hi = schema.scale_range or (float("nan"), float("nan"))
    return {
        "run_id": params.run_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_id": w.case_id,
        "model": model.id,
        "model_source": model.source.value,
        "context_window_id": w.window_id,
        "context_variant": w.variant,
        "persona": w.persona,
        "system_prompt": w.system_prompt,
        "final_prompt": w.user_prompt,
        "raw_output": "",
        "parsed_value_num": float("nan"),
        "parsed_value_json": None,
        "confidence": None,
        "token_prob": None,
        "parse_success": False,
        "parse_method": "failed",
        "format_retries": 0,
        "ground_truth": w.ground_truth,
        "schema_type": schema.type.value,
        "scale_min": lo,
        "scale_max": hi,
        "seed": seed,
        "temperature": params.temperature,
        "dropped_params": None,
        "model_version_returned": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
        "latency_ms": None,
        "error": None,
    }


def _apply_parse(row: dict[str, Any], raw: str, schema: SchemaConfig,
                 json_mode: bool) -> ParseResult:
    row["raw_output"] = raw
    res = parse_response(raw, schema, json_mode=json_mode)
    row["parse_success"] = res.success
    row["parse_method"] = res.method
    row["parsed_value_num"] = res.value_num if res.value_num is not None else float("nan")
    row["parsed_value_json"] = res.value_json
    row["confidence"] = res.confidence
    row["error"] = None if res.success else f"parse: {res.error}"
    return res


# A provider may reject an optional request setting for one model (HTTP 400, e.g.
# "`temperature` is deprecated for this model"). The call is then repeated without that
# setting, and the row records it in `dropped_params`.
_ADAPT_STATUS = {400, 422}
_OPTIONAL_PARAMS = ("temperature", "top_p", "top_k", "seed", "response_format", "logprobs")


def _rejected_param(exc: BaseException, kwargs: dict[str, Any]) -> str | None:
    """Name of a sent setting that the provider's error message rejects, if any."""
    if _status_code(exc) not in _ADAPT_STATUS:
        return None
    msg = str(exc).lower()
    for param in _OPTIONAL_PARAMS:
        if param in kwargs and re.search(rf"\b{param}\b", msg):
            return param
    # Newer OpenAI models: "'max_tokens' is not supported ... use 'max_completion_tokens'".
    if "max_tokens" in kwargs and "max_completion_tokens" in msg:
        return "max_tokens"
    return None


def _without_param(kwargs: dict[str, Any], param: str) -> dict[str, Any]:
    out = {k: v for k, v in kwargs.items() if k != param}
    if param == "max_tokens":  # the limit itself is kept under its new name
        out["max_completion_tokens"] = kwargs["max_tokens"]
    return out


def _note_dropped(row: dict[str, Any], param: str) -> None:
    done = [p for p in (row["dropped_params"] or "").split(", ") if p]
    if param not in done:
        done.append(param)
    row["dropped_params"] = ", ".join(done)
    if param in ("temperature", "seed"):  # not sent, so the row must not claim a value
        row[param] = None


def _empty_reply_error(finish_reason: Any, max_tokens: Any) -> str:
    """Explain a reply without any answer text (there is nothing to parse or to repair)."""
    if finish_reason in ("length", "max_tokens"):
        return (f"{EMPTY_RESPONSE_ERROR} - the model used all {max_tokens} tokens before "
                "writing an answer (typical for reasoning / thinking models). Raise "
                "'Longest allowed answer' in the sidebar (More settings).")
    return f"{EMPTY_RESPONSE_ERROR} (finish_reason: {finish_reason})"


def _keep_first_raw(row: dict[str, Any], first_raw: str) -> None:
    """After a repair turn, keep both replies visible in raw_output for auditing."""
    row["raw_output"] = f"[1st reply - format violated]\n{first_raw}\n\n" \
                        f"[2nd reply - after format reminder]\n{row['raw_output']}"


def _add(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return a if b is None else b
    return a + b


def _build_kwargs(model: ModelSpec, w: ContextWindow, schema: SchemaConfig, params: RunParams,
                  seed: int, api_keys: dict[str, str]) -> tuple[dict[str, Any], bool]:
    kwargs: dict[str, Any] = {
        "model": model.model_route,
        "messages": _messages(model, w),
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "timeout": model.timeout_s,
        "num_retries": 0,  # retries are handled here
        **model.default_params,
    }
    if model.supports_seed:
        kwargs["seed"] = seed
    json_mode = params.use_json_mode and model.supports_json_mode
    if json_mode:
        kwargs["response_format"] = response_format_payload(schema)
    if model.supports_logprobs:
        kwargs["logprobs"] = True
    if model.api_base:
        kwargs["api_base"] = model.api_base
    key = _resolve_api_key(model, api_keys)
    if key:
        kwargs["api_key"] = key
    elif model.source == ModelSource.CUSTOM:
        kwargs["api_key"] = "not-needed"  # OpenAI-compatible clients require a value
    return kwargs, json_mode



async def _call_one(
    model: ModelSpec,
    w: ContextWindow,
    schema: SchemaConfig,
    params: RunParams,
    api_keys: dict[str, str],
    sem: asyncio.Semaphore,
    completion_fn: CompletionFn,
    unsupported: set[str],
) -> dict[str, Any]:
    seed = params.base_seed + w.window_id
    row = _base_row(params, model, w, seed, schema)
    secrets = [v for v in api_keys.values() if v]

    if model.source == ModelSource.MOCK:
        t0 = time.perf_counter()
        raw = simulate_response(MOCK_PROFILES[model.id], w, schema, seed, params.temperature)
        row.update(latency_ms=(time.perf_counter() - t0) * 1000, model_version_returned=model.id,
                   finish_reason="stop", cost_usd=0.0)
        res = _apply_parse(row, raw, schema, json_mode=params.use_json_mode)
        if not res.success and params.repair_format:
            raw2 = simulate_response(MOCK_PROFILES[model.id], w, schema, seed, params.temperature,
                                     repair=True)
            _apply_parse(row, raw2, schema, json_mode=params.use_json_mode)
            row["format_retries"] = 1
            _keep_first_raw(row, raw)
        return row

    kwargs, json_mode = _build_kwargs(model, w, schema, params, seed, api_keys)
    first_raw: str | None = None
    attempt = 0
    async with sem:
        for param in sorted(unsupported):  # already rejected by this model in this run
            if param in kwargs:
                kwargs = _without_param(kwargs, param)
                _note_dropped(row, param)
        while True:
            t0 = time.perf_counter()
            try:
                resp = await completion_fn(**kwargs)
                row["latency_ms"] = _add(row["latency_ms"], (time.perf_counter() - t0) * 1000)
                choice = resp.choices[0]
                raw = choice.message.content or ""
                row["finish_reason"] = getattr(choice, "finish_reason", None)
                row["model_version_returned"] = getattr(resp, "model", None)
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                row["prompt_tokens"] = _add(row["prompt_tokens"], pt)
                row["completion_tokens"] = _add(row["completion_tokens"], ct)
                row["cost_usd"] = _add(row["cost_usd"], _actual_cost(model, pt, ct))
                row["token_prob"] = _token_prob(choice)
                provider_dropped = list(getattr(resp, "dropped_params", None) or [])
                for param in provider_dropped:  # refused by the model, sent without it
                    unsupported.add(param)
                    _note_dropped(row, param)
                res = _apply_parse(row, raw, schema,
                                   json_mode=json_mode and "response_format" in kwargs
                                   and "response_format" not in provider_dropped)
                empty = res.error == EMPTY_RESPONSE_ERROR
                if empty:
                    limit = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
                    row["error"] = "parse: " + _empty_reply_error(row["finish_reason"], limit)
                if first_raw is not None:
                    _keep_first_raw(row, first_raw)
                elif not res.success and params.repair_format and not empty:
                    # Same conversation (still one isolated context window): show the model
                    # its own reply and ask once more for the exact format.
                    first_raw = raw
                    row["format_retries"] = 1
                    kwargs = {**kwargs, "messages": [
                        *kwargs["messages"],
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": build_repair_message(schema, res.error)},
                    ]}
                    continue
                return row
            except Exception as exc:  # provider SDKs raise many error types
                row["latency_ms"] = _add(row["latency_ms"], (time.perf_counter() - t0) * 1000)
                param = _rejected_param(exc, kwargs)
                if param is not None:  # the model does not accept it: go on without it
                    unsupported.add(param)
                    kwargs = _without_param(kwargs, param)
                    _note_dropped(row, param)
                    continue
                if attempt < params.max_retries and _is_retryable(exc):
                    attempt += 1
                    jitter = 0.5 + random.random()  # noqa: S311 - jitter, not crypto
                    await asyncio.sleep(_retry_after(exc) or min(30.0, 2.0**attempt) * jitter)
                    continue
                row["error"] = mask_secrets(f"{type(exc).__name__}: {exc}", secrets)[:500]
                if first_raw is not None:  # repair call failed: keep the original reply
                    row["raw_output"] = first_raw
                return row


async def run_audit(
    windows: list[ContextWindow],
    models: list[ModelSpec],
    schema: SchemaConfig,
    params: RunParams,
    api_keys: dict[str, str] | None = None,
    completion_fn: CompletionFn | None = None,
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """Execute every window against every model concurrently and return ``df_results``."""
    api_keys = api_keys or {}
    fn = completion_fn or _default_completion
    sems = {m.id: asyncio.Semaphore(max(1, min(params.concurrency, m.max_concurrency)))
            for m in models}
    # Settings a model rejected during this run; later calls leave them out right away.
    unsupported: dict[str, set[str]] = {m.id: set() for m in models}
    tasks = [_call_one(m, w, schema, params, api_keys, sems[m.id], fn, unsupported[m.id])
             for m in models for w in windows]
    total = len(tasks)
    rows: list[dict[str, Any]] = []
    for done, fut in enumerate(asyncio.as_completed(tasks), start=1):
        rows.append(await fut)
        if progress:
            progress(done, total)
    df = pd.DataFrame(rows, columns=DF_COLUMNS)
    if not df.empty:
        df = df.sort_values(["model", "case_id", "context_window_id"]).reset_index(drop=True)
    return df


def run_audit_sync(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Synchronous wrapper for Streamlit scripts."""
    return asyncio.run(run_audit(*args, **kwargs))


def parse_success_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Parse success rate per model and variant (flags non-random missingness)."""
    if df.empty:
        return pd.DataFrame(columns=["model", "context_variant", "n", "parse_success_rate"])
    g = df.groupby(["model", "context_variant"])["parse_success"]
    return g.agg(n="size", parse_success_rate="mean").reset_index()


def call_failed_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose call produced no reply at all (HTTP / network error).

    Format problems (``error`` starting with ``parse:``) are not call failures: the model
    answered, only the answer could not be read.
    """
    err = df["error"]
    return err.notna() & ~err.astype(str).str.startswith("parse:")


def call_failure_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct call errors per model with their count (most frequent first)."""
    cols = ["model", "error", "n"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    failed = df[call_failed_mask(df)]
    if failed.empty:
        return pd.DataFrame(columns=cols)
    out = failed.groupby(["model", "error"]).size().reset_index(name="n")
    return out.sort_values(["model", "n"], ascending=[True, False]).reset_index(drop=True)


def dropped_params_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Settings a model rejected and that were therefore left out, with call counts."""
    cols = ["model", "dropped_params", "n"]
    if df.empty or "dropped_params" not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df[df["dropped_params"].notna() & (df["dropped_params"] != "")]
    if d.empty:
        return pd.DataFrame(columns=cols)
    return d.groupby(["model", "dropped_params"]).size().reset_index(name="n")


_HTTP_STATUS_RE = re.compile(r"\bHTTP (\d{3})\b")


def call_failure_hint(error: str) -> str:
    """One plain-English sentence on what to do about a call error ('' if unknown)."""
    m = _HTTP_STATUS_RE.search(error)
    status = int(m.group(1)) if m else None
    if status in (401, 403):
        return "The API key was rejected. Check the key in the sidebar or your Codespaces secret."
    if status == 404:
        return ("The provider does not know this model id: it may have been retired, or your "
                "account has no access. Choose another model or add a current id under "
                "'More settings' in the sidebar.")
    if status == 400:
        return ("The provider rejected the request. The message usually names a setting this "
                "model does not accept (for example temperature or max_tokens).")
    if status == 429:
        return ("Rate limit or quota reached. Lower 'Max parallel requests' or check your plan "
                "and credit with the provider.")
    if status == 408 or "Connection error" in error or "Timeout" in error:
        return ("The server did not answer. For local models, check that the server is running "
                "and the address is correct.")
    if status is not None and status >= 500:
        return "The provider had a temporary problem. Try again in a moment."
    return ""


def mock_model_specs() -> list[ModelSpec]:
    return [ModelSpec(id=name, display_name=name.split("/", 1)[1].replace("-", " ").title()
                      + " (mock)", source=ModelSource.MOCK, max_concurrency=64)
            for name in MOCK_PROFILES]

