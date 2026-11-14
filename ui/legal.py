"""Legal notice shown in the UI (short form of DISCLAIMER.md)."""

from __future__ import annotations

import streamlit as st

DISCLAIMER_TITLE = "Disclaimer & your responsibility"

DISCLAIMER_TEXT = """
**Free, non-commercial research prototype by a private individual, provided "AS IS"
without any warranty** (MIT License).

You run your **own copy** of this app and you are responsible for:

- **API keys & costs**: requests go to the provider *you* choose, billed to *your* account.
- **Your data**: prompts and CSV files are sent to the selected provider. Do not submit
  personal or confidential data without a legal basis (GDPR: *you* are the controller).
- **Provider terms & model licenses**: you must comply with them.
- **Analysis code** you run in the editor (the sandbox is best-effort only).
- **Port 8501**: keep it **private**.
- **Results**: statistics for research and education only. Not professional advice, and
  **not for decisions about individual people** (hiring, credit, etc.).

The code is largely **AI-generated** and has not necessarily been reviewed line by line.
Feature and security descriptions are design goals, not guarantees.

Liability is excluded to the extent permitted by law; where it cannot be excluded, it is
limited to intent and gross negligence. Details: `DISCLAIMER.md` in the repository.
"""


def render_legal_notice() -> None:
    """Render the collapsible disclaimer (call inside the sidebar context)."""
    with st.expander(DISCLAIMER_TITLE, expanded=False):
        st.markdown(DISCLAIMER_TEXT)
