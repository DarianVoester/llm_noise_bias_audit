"""Tab 3 - metrics, interpretation, tables, charts, raw data and exports."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.export import run_config_json, to_csv_bytes, to_excel_bytes
from src.guidance import CHART_TIPS, explain_metric
from src.llm_runner import (
    call_failed_mask,
    call_failure_hint,
    call_failure_summary,
    dropped_params_summary,
    parse_success_summary,
)
from src.presets import PRESETS_BY_ID
from ui.state import ICON_ERROR, ICON_INFO, ICON_OK, ICON_WARNING, TAB_RUN, TAB_START


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)


# Mid-tone, muted, colour-blind friendly palette that stays readable on BOTH the light
# and the dark background. Text, grid and axis colours are left to Streamlit's native
# chart theme, so they switch automatically with light / dark mode.
# Every colour has >= 3:1 contrast on the light (#FFFFFF) AND the dark (#121418) background
# (WCAG 2.x minimum for graphical objects).
CALM_PALETTE = ["#5B80AA", "#C27F52", "#5F9470", "#AE6F8D", "#7E7EB8", "#A98C45",
                "#5E9696", "#8A8A8A"]
CALM_DIVERGING = [[0.0, "#5B80AA"], [0.5, "#9A968E"], [1.0, "#C27F52"]]


def soften_figure(fig: go.Figure) -> go.Figure:
    """Re-colour a Plotly figure's data with the calm palette (charts come from the sandbox).

    Plotly Express writes explicit colours into every trace, so the app theme alone does
    not change them. Traces that share a legend group keep sharing one colour.
    """
    palette = CALM_PALETTE
    colour_of: dict[str, str] = {}
    for trace in fig.data:
        key = str(getattr(trace, "legendgroup", None) or getattr(trace, "name", None) or "")
        colour = colour_of.setdefault(key, palette[len(colour_of) % len(palette)])
        marker = getattr(trace, "marker", None)
        if marker is not None and not isinstance(getattr(marker, "color", None), (list, tuple)):
            try:
                marker.color = colour
            except ValueError:
                pass
        line = getattr(trace, "line", None)
        if line is not None and trace.type in ("box", "violin", "scatter"):
            try:
                line.color = colour
            except ValueError:
                pass
        if trace.type in ("box", "violin"):
            trace.fillcolor = colour + "55"  # translucent fill
    fig.update_layout(colorway=palette)
    if fig.layout.coloraxis is not None and fig.layout.coloraxis.colorscale is not None:
        fig.update_layout(coloraxis_colorscale=CALM_DIVERGING)
    return fig


def _column_help(table: pd.DataFrame) -> dict[str, object]:
    """Tooltips for table headers (hover the column name)."""
    return {c: st.column_config.Column(help=explain_metric(str(c)))
            for c in table.columns if explain_metric(str(c))}


def _call_failures(df: pd.DataFrame) -> None:
    """Show provider / network errors up front - they are not a format problem."""
    failures = call_failure_summary(df)
    if failures.empty:
        return
    n = int(failures["n"].sum())
    st.error(f"{n} of {len(df)} model calls failed, so no answer came back for them. The "
             "provider's message is shown below.", icon=ICON_ERROR)
    for row in failures.head(6).itertuples(index=False):
        st.markdown(f"**{row.model}** · {row.n} call(s)")
        st.code(row.error, language=None)
        hint = call_failure_hint(row.error)
        if hint:
            st.caption(hint)


def _dropped_settings(df: pd.DataFrame) -> None:
    """Tell the user which settings a model refused, so the run went on without them."""
    for row in dropped_params_summary(df).itertuples(index=False):
        text = (f"**{row.model}** does not accept `{row.dropped_params}`. {row.n} call(s) "
                "ran without it, so the model's own default applied.")
        if "temperature" in row.dropped_params:
            text += (" The temperature slider had no effect on this model - keep that in "
                     "mind when you compare its noise with other models.")
        st.warning(text, icon=ICON_WARNING)


def _parse_quality(df: pd.DataFrame) -> None:
    _call_failures(df)
    _dropped_settings(df)
    df = df[~call_failed_mask(df)]  # answer quality is about the replies that came back
    if df.empty:
        return
    rate = float(df["parse_success"].mean()) if len(df) else 0.0
    repaired = int(((df.get("format_retries", 0) > 0) & df["parse_success"]).sum()) \
        if len(df) else 0
    note = (f" {repaired} of them were fixed by one automatic format reminder (e.g. 0.42 became "
            "42 %)." if repaired else "")
    if rate >= 0.95:
        st.success(f"Answer quality: {rate:.0%} of the replies are in the required format."
                   + note, icon=ICON_OK)
    elif rate >= 0.8:
        st.warning(f"Answer quality: {rate:.0%} of the replies are in the required format."
                   + note + " Results are usable, but check the single answers below.",
                   icon=ICON_WARNING)
    else:
        st.error(f"Answer quality: only {rate:.0%} of the replies are in the required format."
                 + note + " Treat the results with great caution.", icon=ICON_ERROR)


def render_results_tab() -> None:
    ss = st.session_state
    df = ss.df_results
    res = ss.eval_result
    if df is None:
        st.info(f"Nothing to show yet. Open the tab **{TAB_START}**, choose a test and press "
                "**Run the test**. The result then appears here.", icon=ICON_INFO)
        return

    with st.expander("How to read this page", expanded=False):
        st.markdown(
            "Read from top to bottom. The further down, the more detail.\n\n"
            "1. **Answer quality** - did the models answer in the requested form? Below "
            "about 90 %, the results may be distorted.\n"
            "2. **In short** - the main result in one or two sentences. If you read only "
            "one thing, read this.\n"
            "3. **The numbers** - move the mouse over the small question mark next to a "
            "number to see what it means.\n"
            "4. **Per model** - one row per model; move the mouse over a column title for "
            "its meaning.\n"
            "5. **Charts** - a short reading guide is shown above them.\n"
            "6. **Every single answer** - for checking and for downloading.\n\n"
            "Keep in mind that these are estimates from a limited number of answers. With "
            "few repetitions, a difference can be pure chance.")
    _parse_quality(df)

    if res is not None and not res.ok:
        st.error(f"The analysis could not be computed: {res.error}")
        if res.error and "No module named" in res.error:
            st.caption("The Python environment is incomplete. Run `pip install --require-hashes "
                       "-r requirements.txt` or rebuild the container, then press "
                       f"**Recalculate without asking again** in the tab **{TAB_RUN}**.")
        for v in res.violations:
            st.caption(f"- {v}")
    elif res is not None:
        preset = PRESETS_BY_ID.get(ss.get("preset_id", ""))
        if preset:
            st.caption(f"Statistical method: {preset.name}")
        for w in res.warnings:
            st.warning(w, icon=ICON_WARNING)
        metrics = res.summary_metrics
        if "error" in metrics:
            st.error(metrics["error"], icon=ICON_ERROR)
        if res.interpretation:
            st.markdown("### In short")
            st.info("\n\n".join(res.interpretation))
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### The numbers")
            items = [(k, v) for k, v in metrics.items() if k != "error"]
            cols = st.columns(3)
            for i, (k, v) in enumerate(items):
                cols[i % 3].metric(k.replace("_", " "), _fmt(v), help=explain_metric(k))
        with right:
            st.markdown("### What the numbers mean")
            if preset:
                st.markdown(preset.how_to_interpret)
            else:
                st.caption("You changed the analysis code, so there is no reading guide for it.")
        if "per_model_table" in res.tables:
            st.markdown("### Per model")
            st.caption("Move the mouse over a column title for an explanation.")
            table = res.tables["per_model_table"]
            st.dataframe(table, width="stretch", hide_index=True,
                         column_config=_column_help(table))
        for name, table in res.tables.items():
            if name != "per_model_table":
                with st.expander(f"More detail: {name.replace('_', ' ')}"):
                    st.dataframe(table, width="stretch", hide_index=True,
                                 column_config=_column_help(table))
        if res.figures:
            st.markdown("### Charts")
            if preset and preset.id in CHART_TIPS:
                st.caption("How to read them: " + CHART_TIPS[preset.id])
        for fig_json in res.figures:
            # theme="streamlit": fonts, grid and background follow light / dark mode
            st.plotly_chart(soften_figure(pio.from_json(fig_json)), width="stretch",
                            theme="streamlit")
        if res.logs:
            with st.expander("Text printed by the analysis code"):
                st.code(res.logs, language=None)

    st.divider()
    st.markdown("### Every single answer")
    st.caption("One row per answer. The most useful columns: `raw_output` (what the model "
               "wrote), `parsed_value_num` (the answer the app read from it), "
               "`context_variant` (which group, order or role the repetition had) and "
               "`parse_success` (whether the answer had the requested form).")
    rate = float(df["parse_success"].mean()) if len(df) else 0.0
    errors = int(df["error"].notna().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Requests sent", len(df), help="Total number of questions sent to the models.")
    c2.metric("Answers in the right form", f"{rate:.0%}",
              help="Share of answers that matched the requested answer format.")
    c3.metric("Problems", errors,
              help="Requests that failed (network, rate limit) or whose answer could not be "
                   "read.")
    answered = df[~call_failed_mask(df)]
    if len(answered) and float(answered["parse_success"].mean()) < 0.9:
        st.warning("Many answers could not be parsed, that is, they did not have the "
                   "requested form. Tick 'Show only the answers with problems' and read the "
                   "column `raw_output`. Usual causes: the model ignores the format, the "
                   "randomness (temperature) is too high, or a local model was set up with "
                   "the wrong chat template.")
    with st.expander("Right form per model and group"):
        st.caption("If one group (e.g. 'anchored') fails much more often than the other, the "
                   "comparison is biased.")
        st.dataframe(parse_success_summary(df), width="stretch", hide_index=True)
    only_fail = st.checkbox("Show only the answers with problems", key="only_fail")
    view = df[~df["parse_success"]] if only_fail else df
    st.dataframe(view, width="stretch", hide_index=True, height=320)
    st.markdown("### Download")

    sheets = dict(res.tables) if res is not None and res.ok else {}
    run_id = ss.run_meta.get("run_id", "run") if ss.run_meta else "run"
    d1, d2, d3 = st.columns(3)
    d1.download_button("All answers (CSV)", to_csv_bytes(df),
                       f"llm_noise_bias_audit_{run_id}.csv", "text/csv",
                       width="stretch")
    d2.download_button("Everything (Excel)", to_excel_bytes(df, sheets, ss.run_meta),
                       f"llm_noise_bias_audit_{run_id}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")
    d3.download_button("Settings of this run (JSON)", run_config_json(ss.run_meta or {}),
                       f"llm_noise_bias_audit_{run_id}_config.json", "application/json",
                       width="stretch")
    st.caption("The CSV file holds every answer. The Excel file adds the result tables and "
               "the settings. The settings file lets you or someone else repeat the run. The "
               "downloads are designed to leave out API keys and to escape spreadsheet "
               "formulas (best effort).")
