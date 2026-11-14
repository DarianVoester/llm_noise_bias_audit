"""Session-state helpers. API keys live ONLY here (volatile, per browser session)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_DESIGN_SEED,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_WINDOWS,
)
from src.config import DEFAULT_TEMPERATURE as DEFAULT_TEMP

DEFAULTS: dict[str, Any] = {
    "api_keys": {},  # provider -> key; never persisted
    "mock_mode": True,
    "selected_models": [],
    "n_windows": DEFAULT_N_WINDOWS,
    "temperature": DEFAULT_TEMP,
    "concurrency": DEFAULT_CONCURRENCY,
    "design_seed": DEFAULT_DESIGN_SEED,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "session_custom_specs": [],  # BYOM "try it" models (not saved to disk)
    "extra_cloud_ids": [],
    "probe_results": {},
    "df_results": None,
    "run_meta": None,
    "eval_result": None,
    "editor_code": None,
    "editor_preset_id": None,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


def session_secrets() -> list[str]:
    """All secrets typed into the UI (used to mask error messages)."""
    return [v for v in st.session_state.get("api_keys", {}).values() if v]


# Plain line icons for message boxes (no emoji), defined once so they stay consistent.
ICON_ERROR = ":material/error:"
ICON_WARNING = ":material/warning:"
ICON_INFO = ":material/info:"
ICON_OK = ":material/check_circle:"

# Tab names, defined once so that every hint in the app points to the same words.
TAB_START = "Start here"
TAB_QUESTION = "1 · Question"
TAB_RUN = "2 · Analysis & run"
TAB_RESULTS = "3 · Results"
TAB_MODELS = "More models"
