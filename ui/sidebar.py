"""Sidebar: which models to test, how often to ask, and the less common settings."""

from __future__ import annotations

import os

import streamlit as st

from src.config import MAX_CONCURRENCY, MAX_N_WINDOWS, PROVIDER_KEY_ENV
from src.model_spec import ModelSource, ModelSpec
from ui.app_control import render_app_control
from ui.legal import render_legal_notice
from ui.models_catalog import available_models, label
from ui.state import TAB_MODELS


def _api_keys() -> None:
    ss = st.session_state
    with st.expander("Use real AI models (API keys)", expanded=False):
        st.caption("An API key is a password from a provider such as OpenAI that lets this app "
                   "send questions on your account. The provider bills you for every request. "
                   "The app is designed to keep keys only in this browser session's memory "
                   "and not to write them to disk, logs or exports (best effort, no "
                   "guarantee). Prefer GitHub Codespaces Secrets.")
        for provider, env in PROVIDER_KEY_ENV.items():
            from_env = bool(os.environ.get(env))
            val = st.text_input(
                f"{provider.title()} key", type="password", key=f"key_{provider}",
                placeholder=f"set via {env}" if from_env else "",
                help=f"Or set the environment variable / Codespaces Secret {env}.",
            )
            if val:
                ss.api_keys[provider] = val.strip()
            elif provider in ss.api_keys and not val:
                ss.api_keys.pop(provider, None)
            if from_env:
                st.caption(f"{env} was found in the environment.")


def _model_choice() -> tuple[list[str], dict[str, ModelSpec], list[str]]:
    ss = st.session_state
    models, notes = available_models()
    has_real = any(m.source != ModelSource.MOCK for m in models)
    ss.mock_mode = st.toggle(
        "Practice mode (simulated models)", value=ss.mock_mode if has_real else True,
        disabled=not has_real,
        help="Practice models are simulations with known behaviour. They are free, answer "
             "instantly and need no key. Switch this off to test real AI models.")
    if not has_real:
        st.caption("Only practice models are available right now. To test real AI models, "
                   f"add an API key below or install a free local model in the tab "
                   f"**{TAB_MODELS}**.")

    visible = [m for m in models if (m.source == ModelSource.MOCK) == ss.mock_mode]
    by_id = {m.id: m for m in visible}
    default = [i for i in ss.selected_models if i in by_id] or list(by_id)[:3]
    chosen = st.multiselect("Models to test", options=list(by_id), default=default,
                            format_func=lambda i: label(by_id[i]),
                            help="Every model gets exactly the same questions. 2 or 3 models "
                                 "make comparisons easy.")
    if ss.mock_mode:
        st.caption("The three practice models behave differently on purpose: *Steady Sage* is "
                   "accurate and stable, *Anchored Optimist* answers too high and is easily "
                   "swayed, *Noisy Novice* scatters a lot. Good for learning what results "
                   "look like.")
    ss.selected_models = chosen
    return chosen, by_id, notes


def _more_settings() -> None:
    ss = st.session_state
    with st.expander("More settings"):
        st.slider("Requests sent at the same time", 1, MAX_CONCURRENCY, key="concurrency",
                  help="Only affects speed, not the results. 4 is fine for cloud models; "
                       "local models usually handle 1 or 2.")
        st.number_input("Longest allowed answer (tokens)", 64, 16384, step=64,
                        key="max_tokens",
                        help="A token is roughly three quarters of a word. Models that 'think' "
                             "before answering use part of this for thinking. If answers come "
                             "back empty, raise it (e.g. to 2048). A higher limit also raises "
                             "the cost estimate.")
        st.number_input("Random seed for the test plan", 0, 10**6, key="design_seed",
                        help="Decides which repetitions land in which group and in which order "
                             "facts are shown. The same number always gives the same plan. "
                             "Leave it alone unless you want a different random assignment.")
        new_id = st.text_input("Add another cloud model (provider/model)",
                               placeholder="openrouter/qwen/qwen3-8b", key="new_cloud_id",
                               help="For models that are not in the list. Needs the matching "
                                    "API key.")
        if st.button("Add model", key="add_cloud") and new_id.strip():
            if new_id.strip() not in ss.extra_cloud_ids:
                ss.extra_cloud_ids.append(new_id.strip())
            st.rerun()


def render_sidebar() -> list[ModelSpec]:
    """Render the sidebar and return the selected model specs."""
    with st.sidebar:
        st.markdown("## Settings")
        st.caption("The defaults work. Change them only if you want to.")
        models_area = st.container()
        st.slider("Repetitions per model", 1, MAX_N_WINDOWS, key="n_windows",
                  help="How often each model is asked the same question. Every repetition is "
                       "a fresh conversation, so the model cannot see its earlier answers. "
                       "10 is good for a first check, 20 or more for the anchoring and "
                       "wording tests. More repetitions give more dependable results but "
                       "take longer and, with paid models, cost more.")
        st.slider("Randomness of the answers (temperature)", 0.0, 2.0, step=0.05,
                  key="temperature",
                  help="How freely the model varies its wording. 0.7 is a good default. At 0 "
                       "most models repeat themselves, so they look more consistent than "
                       "they are; above 1.2 answers become erratic.")
        keys_area = st.container()
        _more_settings()
        notes_area = st.container()
        render_app_control()
        st.divider()
        render_legal_notice()
        st.caption("Research prototype · MIT · provided AS IS, no warranty · built with AI "
                   "assistance (Gemini, Claude). Use at your own responsibility. "
                   "Keep port 8501 **private**.")
        # Keys are read first because they decide which models can be offered above them.
        with keys_area:
            _api_keys()
        with models_area:
            chosen, by_id, notes = _model_choice()
        with notes_area:
            for n in notes:
                st.caption(f"Note: {n}")
    return [by_id[i] for i in chosen]
