"""Capability probe against a fake OpenAI-compatible endpoint (no network)."""

import asyncio
from types import SimpleNamespace

from src.model_registry import CustomModelSpec, apply_probe, probe


def _resp(text, logprobs=None):
    choice = SimpleNamespace(message=SimpleNamespace(content=text), logprobs=logprobs,
                             finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="x", usage=None)


def _spec():
    return CustomModelSpec(id="p", model="my-model",
                           api_base="http://127.0.0.1:8080/v1").to_model_spec()


def test_probe_reports_capabilities():
    calls = []

    async def fake(**kw):
        calls.append(kw)
        if kw.get("response_format"):
            raise ValueError("response_format not supported")
        if kw.get("logprobs"):
            return _resp("OK", SimpleNamespace(content=[SimpleNamespace(logprob=-0.1)]))
        if kw.get("seed") == 1234:
            return _resp("banana")
        return _resp("OK")

    spec = _spec()
    rep = asyncio.run(probe(spec, completion_fn=fake,
                            http_get=lambda url, key: {"data": [{"id": "my-model"}]}))
    assert rep.reachable and rep.completion_ok and rep.model_listed
    assert rep.seed_deterministic is True and rep.json_mode is False
    assert rep.logprobs is True and rep.system_role is True
    adapted = apply_probe(spec, rep)
    assert adapted.supports_json_mode is False and adapted.supports_logprobs is True
    assert adapted.probed_capabilities["seed_deterministic"] is True
    assert all(c["api_base"] == "http://127.0.0.1:8080/v1" for c in calls)
    assert all(c["api_key"] == "not-needed" for c in calls)


def test_probe_unreachable():
    async def down(**kw):
        raise ConnectionError("Connection refused")

    def no_models(url, key):
        raise ConnectionError("refused")

    rep = asyncio.run(probe(_spec(), completion_fn=down, http_get=no_models))
    assert not rep.completion_ok and any("refused" in m for m in rep.messages)


def test_probe_seed_not_honoured_and_no_system_role():
    counter = {"n": 0}

    async def fake(**kw):
        if kw["messages"][0]["role"] == "system":
            raise ValueError("system role not supported")
        counter["n"] += 1
        return _resp(f"word{counter['n']}")

    spec = _spec()
    rep = asyncio.run(probe(spec, completion_fn=fake, http_get=lambda u, k: {"data": []}))
    assert rep.seed_deterministic is False and rep.system_role is False
    assert apply_probe(spec, rep).system_prompt_supported is False
    assert any("Seed is not honoured" in m for m in rep.messages)


def test_probe_warns_on_unknown_model_name():
    async def fake(**kw):
        return _resp("OK")

    rep = asyncio.run(probe(_spec(), completion_fn=fake,
                            http_get=lambda u, k: {"data": [{"id": "other-model"}]}))
    assert rep.model_listed is False
    assert any("other-model" in m for m in rep.messages)
