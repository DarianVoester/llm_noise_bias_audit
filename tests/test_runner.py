"""Tests for the async runner: mock mode, df contract, retries, masking, kwargs."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from src.cases import single_case
from src.context_engine import VariationConfig, VariationMode, build_windows
from src.llm_runner import (
    DF_COLUMNS,
    RunParams,
    call_failed_mask,
    call_failure_hint,
    call_failure_summary,
    dropped_params_summary,
    estimate_run,
    mock_model_specs,
    parse_success_summary,
    run_audit,
    run_audit_sync,
)
from src.model_spec import ModelSource, ModelSpec
from src.schemas import SchemaConfig, SchemaType

PCT = SchemaConfig(type=SchemaType.PERCENTAGE)
CASES = single_case("Estimate the success probability.", ground_truth=60.0)


def _windows(n=20, mode=VariationMode.OCCASION, **kw):
    return build_windows(CASES, VariationConfig(mode=mode, **kw), PCT, n=n, design_seed=1)


def test_mock_run_contract_and_determinism():
    params = RunParams(run_id="t1")
    df1 = run_audit_sync(_windows(), mock_model_specs(), PCT, params)
    df2 = run_audit_sync(_windows(), mock_model_specs(), PCT, RunParams(run_id="t1"))
    assert list(df1.columns) == DF_COLUMNS
    assert len(df1) == 60
    assert df1["parse_success"].mean() > 0.85
    a = df1["parsed_value_num"].to_numpy()
    b = df2["parsed_value_num"].to_numpy()
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])
    assert set(df1["seed"]) == set(range(1, 21))


def test_mock_noise_profiles_differ():
    df = run_audit_sync(_windows(n=40), mock_model_specs(), PCT, RunParams())
    sd = df[df.parse_success].groupby("model")["parsed_value_num"].std()
    assert sd["mock/noisy-novice"] > sd["mock/steady-sage"]


def test_mock_anchoring_effect_visible():
    windows = _windows(n=40, mode=VariationMode.ANCHOR, anchor_text="Someone said 99%.")
    df = run_audit_sync(windows, mock_model_specs(), PCT, RunParams())
    d = df[(df.model == "mock/anchored-optimist") & df.parse_success]
    means = d.groupby("context_variant")["parsed_value_num"].mean()
    assert means["anchored"] > means["baseline"] + 5


@pytest.mark.parametrize("stype", list(SchemaType))
def test_mock_supports_all_schemas(stype):
    opts = ["A", "B", "C"] if stype in (SchemaType.ENUM, SchemaType.RANKING) else []
    schema = SchemaConfig(type=stype, options=opts)
    w = build_windows(CASES, VariationConfig(), schema, n=10, design_seed=1)
    df = run_audit_sync(w, mock_model_specs(), schema, RunParams())
    assert df["parse_success"].mean() > 0.7


def _fake_response(text, model="fake-model-2026"):
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop",
                             logprobs=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], model=model, usage=usage,
                           _hidden_params={"response_cost": 0.0001})


CUSTOM = ModelSpec(id="custom-x", display_name="X", source=ModelSource.CUSTOM,
                   model_route="openai/x", api_base="http://127.0.0.1:8080/v1",
                   supports_seed=True, supports_json_mode=False)


def test_real_path_kwargs_and_parsing():
    seen = []

    async def fake(**kwargs):
        seen.append(kwargs)
        return _fake_response('{"answer": 55}')

    df = asyncio.run(run_audit(_windows(n=3), [CUSTOM], PCT, RunParams(base_seed=100),
                               completion_fn=fake))
    assert df["parse_success"].all() and (df["parsed_value_num"] == 55).all()
    assert sorted(k["seed"] for k in seen) == [101, 102, 103]
    assert all("response_format" not in k for k in seen)  # model has no JSON mode
    assert all(k["api_base"] == "http://127.0.0.1:8080/v1" for k in seen)
    assert all(k["messages"][0]["role"] == "system" for k in seen)
    assert df["model_version_returned"].iloc[0] == "fake-model-2026"
    assert df["parse_method"].iloc[0] == "json"


def test_format_repair_turns_fraction_into_percent():
    """0.42 is rejected; the model is reminded once in the SAME conversation and answers 42."""
    seen = []

    async def fake(**kwargs):
        seen.append(kwargs)
        return _fake_response('{"answer": 0.42}' if len(seen) == 1 else '{"answer": 42}')

    df = asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(), completion_fn=fake))
    row = df.iloc[0]
    assert row["parse_success"] and row["parsed_value_num"] == 42.0
    assert row["format_retries"] == 1 and row["error"] is None
    assert len(seen) == 2
    second = seen[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": '{"answer": 0.42}'}
    assert second[-1]["role"] == "user" and "fraction" in second[-1]["content"]
    assert seen[1]["seed"] == seen[0]["seed"]
    assert "1st reply" in row["raw_output"] and "0.42" in row["raw_output"]
    assert row["prompt_tokens"] == 20 and row["completion_tokens"] == 10


def test_format_repair_happens_only_once_and_can_be_disabled():
    calls = {"n": 0}

    async def stubborn(**kwargs):
        calls["n"] += 1
        return _fake_response('{"answer": 0.42}')

    df = asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(), completion_fn=stubborn))
    assert calls["n"] == 2 and not df["parse_success"].iloc[0]
    assert "fraction" in df["error"].iloc[0]

    calls["n"] = 0
    asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(repair_format=False),
                          completion_fn=stubborn))
    assert calls["n"] == 1


def test_mock_format_repair_raises_parse_rate():
    df = run_audit_sync(_windows(n=40), mock_model_specs(), PCT, RunParams())
    assert df["parse_success"].all()
    assert (df["format_retries"] > 0).any()
    off = run_audit_sync(_windows(n=40), mock_model_specs(), PCT, RunParams(repair_format=False))
    assert off["parse_success"].mean() < 1.0 and (off["format_retries"] == 0).all()


def test_empty_reply_is_explained_and_not_repaired():
    """No answer text (thinking used up the token limit): explain it, do not re-ask."""
    calls = {"n": 0}

    async def empty(**kwargs):
        calls["n"] += 1
        resp = _fake_response("")
        resp.choices[0].finish_reason = "length"
        return resp

    df = asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(max_tokens=64),
                               completion_fn=empty))
    row = df.iloc[0]
    assert calls["n"] == 1 and row["format_retries"] == 0 and not row["parse_success"]
    assert row["error"].startswith("parse: empty response") and "64 tokens" in row["error"]
    assert not call_failed_mask(df).any()  # the call itself worked


def test_call_failures_are_summarised_with_a_hint():
    class Gone(Exception):
        status_code = 404

    async def failing(**kwargs):
        raise Gone("HTTP 404: model: retired-model")

    df = asyncio.run(run_audit(_windows(n=2), [CUSTOM], PCT, RunParams(),
                               completion_fn=failing))
    assert call_failed_mask(df).all()
    summary = call_failure_summary(df)
    assert len(summary) == 1 and summary["n"].iloc[0] == 2
    assert "HTTP 404" in summary["error"].iloc[0]
    assert "retired" in call_failure_hint(summary["error"].iloc[0])
    assert call_failure_hint("something unexpected") == ""

    ok = run_audit_sync(_windows(n=5), mock_model_specs(), PCT, RunParams())
    assert not call_failed_mask(ok).any() and call_failure_summary(ok).empty


class _BadRequest(Exception):
    status_code = 400


def test_rejected_setting_is_dropped_and_recorded():
    """'temperature is deprecated for this model' -> same call again without temperature."""
    seen = []

    async def picky(**kwargs):
        seen.append(kwargs)
        if "temperature" in kwargs:
            raise _BadRequest("HTTP 400: `temperature` is deprecated for this model.")
        return _fake_response('{"answer": 55}')

    df = asyncio.run(run_audit(_windows(n=6), [CUSTOM], PCT, RunParams(concurrency=1),
                               completion_fn=picky))
    assert df["parse_success"].all() and df["error"].isna().all()
    assert (df["dropped_params"] == "temperature").all() and df["temperature"].isna().all()
    assert len(seen) == 7  # rejected once, then remembered for the rest of the run
    assert all("seed" in k and "max_tokens" in k for k in seen)
    summary = dropped_params_summary(df)
    assert summary["n"].iloc[0] == 6 and summary["dropped_params"].iloc[0] == "temperature"

    ok = run_audit_sync(_windows(n=3), mock_model_specs(), PCT, RunParams())
    assert ok["dropped_params"].isna().all() and dropped_params_summary(ok).empty


def test_max_tokens_is_renamed_when_the_model_asks_for_it():
    seen = []

    async def newer(**kwargs):
        seen.append(kwargs)
        if "max_tokens" in kwargs:
            raise _BadRequest("HTTP 400: Unsupported parameter: 'max_tokens' is not supported "
                              "with this model. Use 'max_completion_tokens' instead.")
        return _fake_response('{"answer": 55}')

    df = asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(max_tokens=300),
                               completion_fn=newer))
    assert df["parse_success"].all() and df["dropped_params"].iloc[0] == "max_tokens"
    assert seen[-1]["max_completion_tokens"] == 300 and "max_tokens" not in seen[-1]


def test_other_bad_requests_still_fail():
    calls = {"n": 0}

    async def broken(**kwargs):
        calls["n"] += 1
        raise _BadRequest("HTTP 400: messages: field required")

    df = asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(),
                               completion_fn=broken))
    assert calls["n"] == 1 and call_failed_mask(df).all()
    assert df["dropped_params"].isna().all()


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}

    class RateLimit(Exception):
        status_code = 429

    async def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimit("slow down")
        return _fake_response('{"answer": 10}')

    async def no_sleep(_):
        return None

    monkeypatch.setattr("src.llm_runner.asyncio.sleep", no_sleep)
    df = asyncio.run(run_audit(_windows(n=1), [CUSTOM], PCT, RunParams(), completion_fn=flaky))
    assert df["parse_success"].iloc[0] and calls["n"] == 3


def test_errors_are_masked():
    secret = "sk-proj-SECRETSECRETSECRET1234567890"

    class AuthError(Exception):
        status_code = 401

    async def failing(**kwargs):
        raise AuthError(f"Incorrect API key provided: {kwargs.get('api_key')}")

    cloud = ModelSpec(id="openai/gpt-x", display_name="x", source=ModelSource.CLOUD,
                      model_route="openai/gpt-x", provider="openai")
    df = asyncio.run(run_audit(_windows(n=1), [cloud], PCT, RunParams(),
                               api_keys={"openai": secret}, completion_fn=failing))
    err = df["error"].iloc[0]
    assert secret not in err and "REDACTED" in err
    assert not df["parse_success"].iloc[0]


def test_system_prompt_merge_when_unsupported():
    seen = []

    async def fake(**kwargs):
        seen.append(kwargs)
        return _fake_response('{"answer": 1}')

    model = CUSTOM.model_copy(update={"system_prompt_supported": False})
    asyncio.run(run_audit(_windows(n=1), [model], PCT, RunParams(), completion_fn=fake))
    assert len(seen[0]["messages"]) == 1 and seen[0]["messages"][0]["role"] == "user"


def test_estimate_and_summary():
    w = _windows(n=10)
    est = estimate_run(w, mock_model_specs(), RunParams())
    assert est.calls == 30 and est.est_cost_usd == 0.0
    df = run_audit_sync(w, mock_model_specs(), PCT, RunParams())
    s = parse_success_summary(df)
    assert set(s.columns) == {"model", "context_variant", "n", "parse_success_rate"}
