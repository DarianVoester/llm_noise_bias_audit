"""UI smoke tests with Streamlit's AppTest (mock mode, no network, no keys)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch, tmp_path):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Ollama is (almost certainly) unreachable in CI; point it at a closed loopback port.
    monkeypatch.setattr("src.local_models.OLLAMA_API_BASE", "http://127.0.0.1:9")


def _app() -> AppTest:
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                           default_timeout=120)
    at.run()
    return at


def test_app_starts_without_errors():
    at = _app()
    assert not at.exception, at.exception
    assert "LLM Noise" in at.title[0].value
    assert at.session_state["schema_type"] == "percentage"  # fits the example prompt
    assert at.sidebar.toggle[0].value is True  # practice mode forced without real models
    assert len(at.tabs) >= 4


def test_full_mock_run_produces_results():
    at = _app()
    run = at.button(key="run_main")
    run.click().run()
    assert not at.exception, at.exception
    df = at.session_state["df_results"]
    assert df is not None and len(df) == 30  # 3 mock models x 10 windows
    res = at.session_state["eval_result"]
    assert res is not None and res.ok, getattr(res, "error", None)
    assert res.summary_metrics


def test_failed_model_calls_show_the_provider_error(monkeypatch):
    """A rejected call must surface the provider's message, not look like a format problem."""
    import httpx

    real_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(404, json={"error": {"message": "The model does not exist"}})

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-000000000000")
    monkeypatch.setattr("src.providers.httpx.AsyncClient",
                        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    at = _app()
    next(t for t in at.toggle if "Practice" in t.label).set_value(False).run()
    next(m for m in at.multiselect if m.label == "Models to test").set_value(
        ["openai/gpt-4o-mini"]).run()
    at.button(key="run_main").click().run()
    assert not at.exception, at.exception
    df = at.session_state["df_results"]
    assert len(df) == 10 and not df["parse_success"].any()
    assert at.session_state["eval_result"] is None
    errors = " ".join(e.value for e in at.error)
    assert "Every model call failed" in errors and "10 of 10 model calls failed" in errors
    assert any("HTTP 404" in c.value for c in at.code)
    assert any("retired" in c.value for c in at.caption)
    assert not any("could not be parsed" in w.value for w in at.warning)


def test_model_that_rejects_temperature_still_gets_evaluated(monkeypatch):
    import json

    import httpx

    real_client = httpx.AsyncClient
    ok = {"model": "m", "choices": [{"message": {"role": "assistant",
                                                 "content": '{"answer": 42}'},
                                     "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 50, "completion_tokens": 8}}

    def handler(request):
        if "temperature" in json.loads(request.content):
            return httpx.Response(400, json={"error": {
                "message": "`temperature` is deprecated for this model."}})
        return httpx.Response(200, json=ok)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-000000000000")
    monkeypatch.setattr("src.providers.httpx.AsyncClient",
                        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    at = _app()
    next(t for t in at.toggle if "Practice" in t.label).set_value(False).run()
    next(m for m in at.multiselect if m.label == "Models to test").set_value(
        ["openai/gpt-4o-mini"]).run()
    at.button(key="run_main").click().run()
    assert not at.exception, at.exception
    df = at.session_state["df_results"]
    assert df["parse_success"].all() and (df["dropped_params"] == "temperature").all()
    res = at.session_state["eval_result"]
    assert res is not None and res.ok, getattr(res, "error", None)
    assert any("does not accept `temperature`" in w.value for w in at.warning)
    assert not any("model calls failed" in e.value for e in at.error)


def test_disclaimer_is_shown_in_sidebar():
    at = _app()
    assert not at.exception, at.exception
    labels = [e.label for e in at.sidebar.expander]
    assert any("Disclaimer" in lbl for lbl in labels), labels
    text = " ".join(m.value for m in at.sidebar.markdown)
    assert "without any warranty" in text
    assert "DISCLAIMER.md" in text


def test_disclaimer_file_exists_and_is_linked():
    root = Path(__file__).resolve().parent.parent
    disclaimer = (root / "DISCLAIMER.md").read_text(encoding="utf-8")
    for needle in ("AS IS", "commercial activity", "gross negligence", "GDPR"):
        assert needle in disclaimer
    assert "DISCLAIMER.md" in (root / "README.md").read_text(encoding="utf-8")


def test_docs_make_no_review_or_responsibility_promises():
    """Docs must not claim the maintainer reviewed every line or is responsible for the code."""
    root = Path(__file__).resolve().parent.parent
    risky = ("reviews, tests and is responsible", "is responsible for what is merged",
             "must have read, understood and tested every line", "reviewed/edited by the human",
             "must be reviewed by a human", "expect an acknowledgement within")
    for name in ("README.md", "SPEC.md", "SECURITY.md", "CONTRIBUTING.md", "DISCLAIMER.md"):
        text = (root / name).read_text(encoding="utf-8")
        for phrase in risky:
            assert phrase not in text, f"{name} contains: {phrase!r}"


def test_start_here_tab_and_recipe_button_configure_the_app():
    at = _app()
    assert not at.exception, at.exception
    assert at.tabs[0].label == "Start here"
    recipe_btn = next(b for b in at.button if b.key == "recipe_anchoring")
    recipe_btn.click().run()
    assert not at.exception, at.exception
    assert at.session_state["var_mode"] == "anchor"
    assert at.session_state["preset_id"] == "anchoring_framing_effect"
    assert at.session_state["n_windows"] == 20
    assert any("Settings applied" in s.value for s in at.success)


def test_start_page_runs_the_test_and_shows_the_summary():
    """A first test needs one click on the start page, without visiting another tab."""
    at = _app()
    assert any("Does a model give the same answer every time?" in m.value
               for m in at.tabs[0].markdown)  # the default settings are the first recipe
    at.button(key="run_start").click().run()
    assert not at.exception, at.exception
    res = at.session_state["eval_result"]
    assert res is not None and res.ok and res.interpretation
    assert any(res.interpretation[0] in i.value for i in at.tabs[0].info)


def test_run_is_blocked_while_the_settings_check_reports_an_error():
    at = _app()
    at.selectbox(key="preset_id").set_value("anchoring_framing_effect").run()
    assert at.button(key="run_main").disabled and at.button(key="run_start").disabled
    assert any("compares two groups" in w.value for w in at.tabs[0].warning)


def test_settings_check_flags_incompatible_combination():
    at = _app()
    at.selectbox(key="preset_id").set_value("anchoring_framing_effect").run()
    assert not at.exception, at.exception
    assert any("compares two groups" in e.value for e in at.error)
    assert any(b.key == "switch_preset" for b in at.button)


def test_metric_help_is_shown_on_results():
    at = _app()
    at.button(key="run_main").click().run()
    assert not at.exception, at.exception
    assert at.metric, "no metrics rendered"
    assert any(m.help for m in at.metric)


def test_malicious_editor_code_disables_run():
    at = _app()
    at.text_area(key="code_area").set_value("import os\n").run()
    assert not at.exception
    run = at.button(key="run_main")
    assert run.disabled
    assert any("Security check failed" in e.value for e in at.error)


def test_restart_and_stop_buttons_hand_over_to_the_start_script(monkeypatch):
    """The sidebar buttons must start the script detached, and never touch a running app here."""
    calls = []
    monkeypatch.setattr("ui.app_control.subprocess.Popen",
                        lambda args, **kw: calls.append((args, kw)))
    at = _app()
    at.button(key="app_restart").click().run()
    assert not at.exception, at.exception
    assert any("restarting" in i.value for i in at.sidebar.info)
    at.button(key="app_stop").click().run()
    assert any("stopping" in i.value for i in at.sidebar.info)
    assert [c[0][-1] for c in calls] == ["restart", "stop"]
    assert all(c[0][-2].endswith("scripts/start_app.sh") for c in calls)
    assert all(c[1]["start_new_session"] for c in calls)


def test_app_control_rejects_unknown_commands():
    from ui.app_control import launch

    with pytest.raises(ValueError):
        launch("rm -rf /")
