"""In-house provider client: routing, request format, response parsing, errors (no network)."""

import asyncio
import json

import httpx
import pytest

from src.providers import (
    ANTHROPIC_VERSION,
    GEMINI_OPENAI_BASE,
    ProviderError,
    acompletion,
    resolve_route,
    unsupported_settings,
)

MSGS = [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "Hi"}]
OPENAI_OK = {"model": "gpt-4o-mini-2024-07-18",
             "choices": [{"message": {"role": "assistant", "content": '{"answer": 42}'},
                          "finish_reason": "stop",
                          "logprobs": {"content": [{"token": "{", "logprob": -0.01},
                                                   {"token": "42", "logprob": -0.2}]}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 5}}
ANTHROPIC_OK = {"model": "claude-haiku-4-5-20251001", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"answer": 7}'}],
                "usage": {"input_tokens": 20, "output_tokens": 6}}


def run(handler, **kw):
    seen = {}

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return handler(request)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(wrapped)) as c:
            return await acompletion(client=c, **kw)

    return asyncio.run(go()), seen


@pytest.mark.parametrize(
    "model_id,api_base,base,protocol",
    [("openai/gpt-4o-mini", None, "https://api.openai.com/v1", "openai"),
     ("openrouter/meta-llama/llama-3.1-8b-instruct", None, "https://openrouter.ai/api/v1",
      "openai"),
     ("gemini/gemini-3.6-flash", None, GEMINI_OPENAI_BASE, "openai"),
     ("ollama_chat/gemma3:270m", "http://127.0.0.1:11434", "http://127.0.0.1:11434/v1",
      "openai"),
     ("ollama_chat/gemma3:270m", "http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1",
      "openai"),
     ("openai/my-ft", "http://127.0.0.1:8080/v1/", "http://127.0.0.1:8080/v1", "openai"),
     ("anthropic/claude-x", None, "https://api.anthropic.com", "anthropic")],
)
def test_routing(model_id, api_base, base, protocol):
    r = resolve_route(model_id, api_base)
    assert r.base_url == base and r.protocol == protocol
    assert r.model == model_id.split("/", 1)[1]


def test_routing_rejects_unknown():
    with pytest.raises(ProviderError):
        resolve_route("nonsense")
    with pytest.raises(ProviderError):
        resolve_route("bedrock/x")


def test_openai_request_and_parse():
    resp, seen = run(lambda r: httpx.Response(200, json=OPENAI_OK), model="openai/gpt-4o-mini",
                     messages=MSGS, api_key="sk-test-123", temperature=0.7, max_tokens=50,
                     seed=3, response_format={"type": "json_object"}, logprobs=True,
                     num_retries=0, unknown_param="dropped")
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer sk-test-123"
    body = seen["body"]
    assert body["model"] == "gpt-4o-mini" and body["seed"] == 3 and body["temperature"] == 0.7
    assert body["messages"] == MSGS and body["stream"] is False
    assert "unknown_param" not in body and "num_retries" not in body
    assert "sk-test" not in seen["url"]
    assert resp.choices[0].message.content == '{"answer": 42}'
    assert resp.model == "gpt-4o-mini-2024-07-18" and resp.usage.prompt_tokens == 12
    assert [t.logprob for t in resp.choices[0].logprobs.content] == [-0.01, -0.2]
    assert resp.sent_params["seed"] == 3


def test_gemini_drops_unsupported_params():
    _, seen = run(lambda r: httpx.Response(200, json=OPENAI_OK), model="gemini/gemini-3.6-flash",
                  messages=MSGS, api_key="AIza-key", seed=1, logprobs=True, temperature=0.5)
    assert seen["url"] == GEMINI_OPENAI_BASE + "/chat/completions"
    assert "seed" not in seen["body"] and "logprobs" not in seen["body"]
    assert seen["body"]["temperature"] == 0.5


def test_local_endpoint_without_key_sends_no_auth_header():
    _, seen = run(lambda r: httpx.Response(200, json=OPENAI_OK), model="openai/my-ft",
                  api_base="http://127.0.0.1:8080/v1", messages=MSGS, api_key="not-needed")
    assert "authorization" not in seen["headers"]


def test_anthropic_request_and_parse():
    resp, seen = run(lambda r: httpx.Response(200, json=ANTHROPIC_OK),
                     model="anthropic/claude-haiku-4-5-20251001", messages=MSGS,
                     api_key="sk-ant-xyz", temperature=1.4, max_tokens=64, seed=5,
                     response_format={"type": "json_object"})
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant-xyz"
    assert seen["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    body = seen["body"]
    assert body["system"] == "Be careful."
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["temperature"] == 1.0  # clamped to Anthropic's 0..1 range
    assert "seed" not in body and "response_format" not in body
    assert resp.choices[0].message.content == '{"answer": 7}'
    assert resp.choices[0].finish_reason == "end_turn"
    assert resp.usage.prompt_tokens == 20 and resp.usage.completion_tokens == 6


def test_rate_limit_error_carries_retry_after():
    def handler(r):
        return httpx.Response(429, headers={"retry-after": "7"},
                              json={"error": {"message": "Rate limit reached"}})

    with pytest.raises(ProviderError) as ei:
        run(handler, model="openai/gpt-4o-mini", messages=MSGS, api_key="k")
    assert ei.value.status_code == 429 and ei.value.retry_after == 7.0
    assert "Rate limit" in str(ei.value)


def test_auth_error_masks_key():
    key = "sk-proj-SECRETSECRETSECRET1234567890"

    def handler(r):
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key: {key}"}})

    with pytest.raises(ProviderError) as ei:
        run(handler, model="openai/gpt-4o-mini", messages=MSGS, api_key=key)
    assert key not in str(ei.value) and ei.value.status_code == 401


def test_connection_error_and_timeout():
    def refuse(r):
        raise httpx.ConnectError("Connection refused")

    def slow(r):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ProviderError, match="Connection error"):
        run(refuse, model="openai/x", api_base="http://127.0.0.1:9/v1", messages=MSGS)
    with pytest.raises(ProviderError) as ei:
        run(slow, model="openai/x", api_base="http://127.0.0.1:9/v1", messages=MSGS)
    assert ei.value.status_code == 408


def test_invalid_payloads():
    with pytest.raises(ProviderError, match="invalid JSON"):
        run(lambda r: httpx.Response(200, text="<html>"), model="openai/x", messages=MSGS)
    with pytest.raises(ProviderError, match="no choices"):
        run(lambda r: httpx.Response(200, json={"choices": []}), model="openai/x",
            messages=MSGS)


def test_openai_content_given_as_parts_keeps_only_the_text():
    parts = [{"type": "reasoning", "text": "maybe 30"},
             {"type": "text", "text": '{"answer": 42}'}]
    payload = {"choices": [{"message": {"role": "assistant", "content": parts},
                            "finish_reason": "stop"}]}
    resp, _ = run(lambda r: httpx.Response(200, json=payload), model="openai/x",
                  api_base="http://127.0.0.1:9/v1", messages=MSGS)
    assert resp.choices[0].message.content == '{"answer": 42}'


def test_openai_missing_content_is_an_empty_string():
    payload = {"choices": [{"message": {"role": "assistant", "content": None},
                            "finish_reason": "length"}]}
    resp, _ = run(lambda r: httpx.Response(200, json=payload), model="openai/x",
                  api_base="http://127.0.0.1:9/v1", messages=MSGS)
    assert resp.choices[0].message.content == ""
    assert resp.choices[0].finish_reason == "length"


def _picky(rejects, message, status=400, protocol="openai"):
    """Fake server that refuses requests containing one of ``rejects``."""
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append({k: v for k, v in body.items() if k not in ("messages", "system")})
        if any(k in body for k in rejects):
            return httpx.Response(status, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": message}})
        return httpx.Response(200, json=ANTHROPIC_OK if protocol == "anthropic" else OPENAI_OK)

    return handler, calls


@pytest.mark.parametrize("model,message,rejected", [
    ("anthropic/claude-sonnet-5-5", "`temperature` is deprecated for this model.",
     "temperature"),
    ("anthropic/claude-opus-5-5", "temperature: Extra inputs are not permitted", "temperature"),
    ("anthropic/claude-haiku-5-5", "`top_p` is deprecated for this model.", "top_p"),
    ("openai/gpt-6", "Unsupported value: 'temperature' does not support 0.7 with this model. "
                     "Only the default (1) value is supported.", "temperature"),
    ("openai/o9-mini", "Unsupported parameter: 'seed' is not supported with this model.",
     "seed"),
    ("openrouter/vendor/model", "logprobs is not supported for this model", "logprobs"),
    ("openai/local-ft", "response_format type json_object is not supported", "response_format"),
])
def test_refused_setting_is_dropped_for_any_model(model, message, rejected):
    protocol = "anthropic" if model.startswith("anthropic/") else "openai"
    handler, calls = _picky({rejected}, message, protocol=protocol)
    kw = {"api_base": "http://127.0.0.1:9/v1"} if model == "openai/local-ft" else {}
    resp, _ = run(handler, model=model, messages=MSGS, api_key="k", temperature=0.7, top_p=0.9,
                  seed=3, logprobs=True, response_format={"type": "json_object"}, **kw)
    assert resp.dropped_params == [rejected]
    assert rejected not in calls[-1] and len(calls) == 2
    assert unsupported_settings(model, kw.get("api_base")) == {rejected}

    handler2, calls2 = _picky({rejected}, message, protocol=protocol)
    resp2, _ = run(handler2, model=model, messages=MSGS, api_key="k", temperature=0.7,
                   top_p=0.9, seed=3, logprobs=True, response_format={"type": "json_object"},
                   **kw)
    assert len(calls2) == 1 and resp2.dropped_params == [rejected]  # remembered: no 400


def test_several_refused_settings_and_max_tokens_rename():
    def handler(request):
        body = json.loads(request.content)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "temperature is deprecated"}})
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Use 'max_completion_tokens'"
                                                                  " instead of max_tokens."}})
        return httpx.Response(200, json=OPENAI_OK)

    resp, seen = run(handler, model="openai/gpt-6", messages=MSGS, api_key="k",
                     temperature=0.7, max_tokens=300)
    assert resp.dropped_params == ["temperature", "max_tokens"]
    assert seen["body"]["max_completion_tokens"] == 300 and "max_tokens" not in seen["body"]


def test_unnamed_rejection_falls_back_to_minimal_request():
    handler, calls = _picky({"temperature"}, "This model does not support sampling settings.")
    resp, _ = run(handler, model="openai/gpt-6", messages=MSGS, api_key="k", temperature=0.7,
                  seed=3)
    assert set(resp.dropped_params) == {"temperature", "seed"} and len(calls) == 2


def test_unrelated_bad_request_is_not_hidden():
    handler, calls = _picky({"messages"}, "messages: field required")
    with pytest.raises(ProviderError) as ei:
        run(handler, model="openai/gpt-6", messages=MSGS, api_key="k", temperature=0.7)
    assert ei.value.status_code == 400 and len(calls) == 1
    assert unsupported_settings("openai/gpt-6") == set()


def test_runner_uses_provider_client_end_to_end(monkeypatch):
    """Runner -> providers.acompletion -> mocked HTTP, incl. retry on 429 and cost."""
    from src.cases import single_case
    from src.context_engine import VariationConfig, build_windows
    from src.llm_runner import RunParams, run_audit
    from src.model_registry import cloud_model_specs
    from src.schemas import SchemaConfig, SchemaType

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {}})
        return httpx.Response(200, json=OPENAI_OK)

    real_client = httpx.AsyncClient

    def fake_client(*a, **k):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("src.providers.httpx.AsyncClient", fake_client)
    schema = SchemaConfig(type=SchemaType.PERCENTAGE)
    windows = build_windows(single_case("x"), VariationConfig(), schema, n=2, design_seed=1)
    model = cloud_model_specs({"openai": "sk-test"})[0]
    df = asyncio.run(run_audit(windows, [model], schema, RunParams()))
    assert df["parse_success"].all() and (df["parsed_value_num"] == 42).all()
    assert calls["n"] == 3  # one 429 retried, then two successes
    assert df["cost_usd"].notna().all() and (df["cost_usd"] > 0).all()
    assert df["token_prob"].notna().all()


def test_runner_records_settings_the_provider_dropped(monkeypatch):
    """Real HTTP path: Claude refuses temperature -> every row is answered and marked."""
    from src.cases import single_case
    from src.context_engine import VariationConfig, build_windows
    from src.llm_runner import RunParams, call_failed_mask, run_audit
    from src.model_registry import custom_cloud_spec
    from src.schemas import SchemaConfig, SchemaType

    handler, calls = _picky({"temperature"}, "`temperature` is deprecated for this model.",
                            protocol="anthropic")
    real_client = httpx.AsyncClient
    monkeypatch.setattr("src.providers.httpx.AsyncClient",
                        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    schema = SchemaConfig(type=SchemaType.PERCENTAGE)
    windows = build_windows(single_case("x"), VariationConfig(), schema, n=10, design_seed=1)
    model = custom_cloud_spec("anthropic/claude-sonnet-5-5")
    df = asyncio.run(run_audit(windows, [model], schema, RunParams(concurrency=1),
                               api_keys={"anthropic": "sk-ant-test"}))
    assert not call_failed_mask(df).any() and df["parse_success"].all()
    assert (df["dropped_params"] == "temperature").all() and df["temperature"].isna().all()
    assert len(calls) == 11  # one rejection, then remembered

