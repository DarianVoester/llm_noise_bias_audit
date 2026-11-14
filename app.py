"""LLM Noise & Bias Audit Studio - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ui.sidebar import render_sidebar
from ui.state import (
    TAB_MODELS,
    TAB_QUESTION,
    TAB_RESULTS,
    TAB_RUN,
    TAB_START,
    init_state,
)
from ui.tab_editor import render_editor_tab
from ui.tab_guide import apply_pending_recipe, render_guide_tab
from ui.tab_models import render_models_tab
from ui.tab_results import render_results_tab
from ui.tab_setup import render_setup_tab

st.set_page_config(page_title="LLM Noise & Bias Audit Studio",
                   page_icon=":material/query_stats:", layout="wide")
# Only the analysis-code editor is monospaced; questions are typed in the normal font.
st.markdown("<style>.st-key-code_area textarea {font-family: ui-monospace, monospace "
            "!important;}</style>", unsafe_allow_html=True)


def main() -> None:
    init_state()
    apply_pending_recipe()  # must run before any widget is created
    st.title("LLM Noise & Bias Audit Studio")
    st.markdown("Ask an AI model the same question many times and see how much its answers "
                "change. You do not need any technical knowledge: start on the first tab and "
                "follow the three steps.")
    models = render_sidebar()
    tab_guide, tab_setup, tab_editor, tab_results, tab_models = st.tabs(
        [TAB_START, TAB_QUESTION, TAB_RUN, TAB_RESULTS, TAB_MODELS])
    # The tabs are filled in the order the data flows, not in the order they are shown:
    # the start page needs the question (tab 1) and the analysis (tab 2) to offer its
    # own run button, and the results tab must come after any run.
    with tab_setup:
        setup = render_setup_tab()
    with tab_editor:
        preset, code = render_editor_tab(setup, models)
    with tab_guide:
        render_guide_tab(setup, models, preset, code)
    with tab_models:
        render_models_tab()
    with tab_results:
        render_results_tab()
    st.divider()
    st.caption("Free, non-commercial research prototype, provided AS IS without warranty; "
               "built with AI assistance. You are responsible for your API usage, costs and "
               "data. See the disclaimer in the sidebar. Inspired by *Noise* by Kahneman, "
               "Sibony and Sunstein.")


main()
