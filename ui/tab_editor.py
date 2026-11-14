"""Tab 2 - choose the analysis, check the settings, optional code editor, run button.

The run panel (:func:`run_panel`) is also used by the start page, so a first test needs
one click and no tab change.
"""

from __future__ import annotations

import streamlit as st

from src.config import CONFIRM_ABOVE_CALLS, CONFIRM_ABOVE_COST_USD
from src.evaluator import run_sandboxed, validate_code
from src.export import build_run_config
from src.guidance import PRESET_QUESTIONS, Tip, check_setup, setup_ok, suggest_preset
from src.llm_runner import (
    RunParams,
    call_failed_mask,
    dropped_params_summary,
    estimate_run,
    run_audit_sync,
)
from src.model_spec import ModelSpec
from src.presets import PRESETS_BY_ID, Preset, presets_for_schema
from src.security import mask_secrets
from ui.state import (
    ICON_ERROR,
    ICON_INFO,
    ICON_OK,
    ICON_WARNING,
    TAB_QUESTION,
    TAB_RESULTS,
    TAB_RUN,
    TAB_START,
    session_secrets,
)
from ui.tab_setup import SetupResult

BADGE = {"recommended": "fits your answer format",
         "caveat": "works only roughly with your answer format",
         "incompatible": "does not fit your answer format"}


def short_name(preset_id: str) -> str:
    """Name of the statistical method without the bracketed details."""
    return PRESETS_BY_ID[preset_id].name.split(" (")[0]


def analysis_label(preset_id: str, compat: str = "recommended") -> str:
    """Everyday question an analysis answers, plus a note if it does not fit."""
    question = PRESET_QUESTIONS.get(preset_id, short_name(preset_id))
    return question if compat == "recommended" else f"{question}  ({BADGE[compat]})"


def render_guide(preset: Preset, compat: str) -> None:
    st.markdown(preset.plain_english_summary)
    st.caption(f"Statistical method: {preset.name} · {BADGE[compat]}")
    with st.expander("How to read the numbers of this analysis", expanded=False):
        st.markdown(preset.how_to_interpret)
        st.markdown(f"**Connection to the book *Noise*:** {preset.kahneman_concept}")
        reqs = []
        if preset.requires_ground_truth:
            reqs.append("the correct answer (tab 1, or a ground_truth column in the file)")
        if preset.requires_cases > 1:
            reqs.append(f"at least {preset.requires_cases} different questions (a file in "
                        "tab 1)")
        if preset.min_n > 2:
            reqs.append(f"at least {preset.min_n} repetitions per model")
        if reqs:
            st.caption("Needs: " + "; ".join(reqs) + ".")


def _code_editor(preset: Preset) -> str:
    ss = st.session_state
    if ss.editor_preset_id != preset.id or "code_area" not in ss:
        ss["code_area"] = preset.code
        ss.editor_preset_id = preset.id
    c1, c2 = st.columns([4, 1])
    c1.markdown("This is the Python code that computes the result. You can ignore it. If you "
                "know Python you may change it; it runs in a restricted environment (no "
                "imports needed: `np`, `pd`, `stats`, `px`, `go`, `math` are available).")
    if c2.button("Undo my changes", width="stretch"):
        ss["code_area"] = preset.code
    code = st.text_area("evaluate(df_results, ground_truth)", key="code_area", height=420,
                        label_visibility="collapsed")
    if not validate_code(code):
        st.caption("Security check passed (the code is checked before it runs and gets an "
                   "isolated process with a 5 s limit; best effort).")
    return code


def _show_code_violations(code: str) -> None:
    """Shown outside the collapsed editor so a blocked run is never unexplained."""
    violations = validate_code(code)
    if violations:
        st.error("Security check failed - this code will not run:\n\n" +
                 "\n".join(f"- {v}" for v in violations[:8]) +
                 "\n\nOpen the code section above and press **Undo my changes** to restore "
                 "the original code.")


def _run_models(setup: SetupResult, models: list[ModelSpec]) -> None:
    ss = st.session_state
    params = RunParams(temperature=ss.temperature, concurrency=ss.concurrency,
                       max_tokens=ss.max_tokens)
    total = len(setup.windows) * len(models)
    bar = st.progress(0.0, text=f"Sending {total} requests…")

    def progress(done: int, n: int) -> None:
        bar.progress(done / n, text=f"{done} of {n} answers received")

    assert setup.schema is not None
    df = run_audit_sync(setup.windows, models, setup.schema, params,
                        api_keys=dict(ss.api_keys), progress=progress)
    bar.empty()
    ss.df_results = df
    ss.run_meta = build_run_config(
        run_id=params.run_id,
        models=[m.model_dump(mode="json") for m in models],
        schema=setup.schema.model_dump(mode="json"),
        variation=setup.variation.model_dump(mode="json") if setup.variation else None,
        n_windows=ss.n_windows, temperature=ss.temperature, design_seed=ss.design_seed,
        cases=[c.__dict__ for c in setup.cases or []], ground_truth=setup.ground_truth,
        mock_mode=ss.mock_mode,
    )


def _evaluate(code: str, ground_truth: float | None) -> None:
    ss = st.session_state
    with st.spinner("Computing the result…"):
        res = run_sandboxed(code, ss.df_results, ground_truth)
    if res.error:
        res.error = mask_secrets(res.error, session_secrets())
    ss.eval_result = res


TIP_STYLE = {"error": (st.error, ICON_ERROR), "warning": (st.warning, ICON_WARNING),
             "info": (st.info, ICON_INFO), "ok": (st.success, ICON_OK)}


def _has_truth(setup: SetupResult) -> bool:
    return setup.ground_truth is not None or \
        any(c.ground_truth is not None for c in setup.cases or [])


def setup_tips(setup: SetupResult, models: list[ModelSpec], preset: Preset) -> list[Tip]:
    """Findings of the settings check for the current question, models and analysis."""
    ss = st.session_state
    return check_setup(schema=setup.schema.type if setup.schema else None,
                       mode=setup.variation.mode if setup.variation else None,
                       preset_id=preset.id, n_windows=ss.n_windows,
                       temperature=ss.temperature, n_models=len(models),
                       n_cases=len(setup.cases or []), has_truth=_has_truth(setup))


def _settings_check(setup: SetupResult, models: list[ModelSpec], preset: Preset) -> None:
    """Show whether the current combination of settings makes sense."""
    ss = st.session_state
    tips = setup_tips(setup, models, preset)
    st.markdown("#### 2 · Do these settings fit together?")
    for tip in tips:
        fn, icon = TIP_STYLE[tip.level]
        fn(tip.message, icon=icon)
    if setup.schema is not None and setup.variation is not None and not setup_ok(tips):
        best = suggest_preset(setup.schema.type, setup.variation.mode, len(setup.cases or []),
                              _has_truth(setup), len(models))
        if best != preset.id:
            st.caption(f"Suggestion: the analysis **{analysis_label(best)}** fits your "
                       "current question.")
            if st.button("Use the suggested analysis", key="switch_preset"):
                ss["_pending_preset"] = best
                st.rerun()


def _duration(seconds: float) -> str:
    if seconds < 5:
        return "a few seconds"
    if seconds < 90:
        return f"about {round(seconds)} seconds"
    return f"about {round(seconds / 60)} minutes"


def run_panel(setup: SetupResult, models: list[ModelSpec], preset: Preset, code: str, *,
              key: str, on_start_page: bool = False) -> None:
    """What the run will do, anything that blocks it, and the run button itself."""
    ss = st.session_state
    est = estimate_run(setup.windows, models,
                       RunParams(temperature=ss.temperature, concurrency=ss.concurrency,
                                 max_tokens=ss.max_tokens))
    if est.est_cost_usd == 0 and not est.unknown_cost_models:
        cost = "free"
    else:
        cost = f"roughly ${est.est_cost_usd:.4f}" + \
            (" plus models with unknown prices" if est.unknown_cost_models else "")
    if models and setup.windows:
        st.markdown(f"{len(models)} model(s) × {len(setup.windows)} fresh conversations each = "
                    f"**{est.calls} requests** · cost: {cost} · takes "
                    f"{_duration(est.est_seconds)}")

    code_ok = not validate_code(code)
    tips = setup_tips(setup, models, preset)
    blockers: list[str] = []
    if setup.errors:
        blockers.append(f"Something is missing in tab **{TAB_QUESTION}**"
                        + (": " + " ".join(setup.errors) if on_start_page else "."))
    if not models:
        blockers.append("No model is selected. Choose at least one in the sidebar on the left.")
    elif on_start_page:  # tab 2 shows these in its settings check right above
        blockers += [t.message for t in tips if t.level == "error"]
    if not code_ok and on_start_page:
        blockers.append(f"The analysis code was changed and did not pass the security check. "
                        f"See tab **{TAB_RUN}**.")
    for msg in blockers:
        st.warning(msg, icon=ICON_WARNING)

    needs_confirm = est.calls > CONFIRM_ABOVE_CALLS or est.est_cost_usd > CONFIRM_ABOVE_COST_USD
    confirmed = st.checkbox("I understand the number of requests and the possible costs.",
                            key=f"confirm_{key}") if needs_confirm else True
    ready = bool(models) and bool(setup.windows) and not setup.errors and setup_ok(tips)
    c1, c2 = st.columns(2)
    run_clicked = c1.button("Run the test", type="primary", width="stretch",
                            key=f"run_{key}",
                            disabled=not (ready and confirmed and code_ok),
                            help="Sends the question to every selected model, as often as set "
                                 "in the sidebar, and then computes the result.")
    eval_clicked = False
    if not on_start_page:
        eval_clicked = c2.button(
            "Recalculate without asking again", width="stretch", key=f"reeval_{key}",
            disabled=ss.df_results is None or not code_ok,
            help="Applies the chosen analysis to the answers you already have. No new "
                 "requests, so it is free. Use it after switching the analysis.")
    if run_clicked:
        try:
            _run_models(setup, models)
        except Exception as exc:  # never show raw provider errors (may contain keys)
            st.error("Run failed: " + mask_secrets(f"{type(exc).__name__}: {exc}",
                                                   session_secrets())[:500])
            return
    if run_clicked or eval_clicked:
        df = ss.df_results
        failed = int(call_failed_mask(df).sum())
        if failed and failed == len(df):
            ss.eval_result = None  # no answers: an analysis would only add noise
            st.error("Every model call failed, so there is nothing to analyse. Open the tab "
                     f"**{TAB_RESULTS}** to read the provider's error message.", icon=ICON_ERROR)
            return
        _evaluate(code, setup.ground_truth)
        if failed:
            st.warning(f"{failed} of {len(df)} model calls failed. The provider's error "
                       f"message is shown in the tab **{TAB_RESULTS}**.", icon=ICON_WARNING)
        for row in dropped_params_summary(df).itertuples(index=False):
            st.warning(f"**{row.model}** does not accept `{row.dropped_params}`, so its calls "
                       "ran without that setting.", icon=ICON_WARNING)
        where = "The summary is right below. Charts" if on_start_page else "The summary, charts"
        st.success(f"Done. {where} and all details are in the tab **{TAB_RESULTS}** at the "
                   "top of the page.", icon=ICON_OK)


def render_editor_tab(setup: SetupResult, models: list[ModelSpec]) -> tuple[Preset, str]:
    """Render tab 2 and return the chosen analysis and its (possibly edited) code."""
    ss = st.session_state
    pending = ss.pop("_pending_preset", None)
    if pending in PRESETS_BY_ID:
        ss["preset_id"] = pending
    schema_type = setup.schema.type if setup.schema else None
    options = presets_for_schema(schema_type) if schema_type else \
        [(p, "recommended") for p in PRESETS_BY_ID.values()]
    compat = {p.id: c for p, c in options}
    if ss.get("preset_id") not in compat:
        ss.pop("preset_id", None)
        if schema_type is not None and setup.variation is not None:
            ss["preset_id"] = suggest_preset(schema_type, setup.variation.mode,
                                             len(setup.cases or []), _has_truth(setup),
                                             len(models))
    st.markdown("#### 1 · What do you want to find out?")
    st.caption("The analysis decides which numbers are computed from the answers. Not sure? "
               f"Pick one of the ready-made tests in the tab **{TAB_START}**; it sets this "
               "for you.")
    preset_id = st.selectbox(
        "Analysis", [p.id for p, _ in options], key="preset_id",
        format_func=lambda i: analysis_label(i, compat[i]))
    preset = PRESETS_BY_ID[preset_id]
    render_guide(preset, compat[preset_id])
    _settings_check(setup, models, preset)
    with st.expander("For experts: view or edit the analysis code (optional)",
                     expanded=False):
        code = _code_editor(preset)
    _show_code_violations(code)

    st.divider()
    st.markdown("#### 3 · Run")
    run_panel(setup, models, preset, code, key="main")
    return preset, code
