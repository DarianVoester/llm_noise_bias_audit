"""Restart or stop the app from inside the app (sidebar buttons).

The work is done by ``scripts/start_app.sh``, started as a detached process: it has to
outlive the app it is about to stop. A short delay lets the message below reach the browser
before the server goes away.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import streamlit as st

from ui.state import ICON_INFO

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "start_app.sh"
START_HINT = "bash scripts/start_app.sh"


def available() -> bool:
    """The buttons need bash and the start script (i.e. Codespaces, Dev Container, Linux, macOS)."""
    return SCRIPT.is_file() and shutil.which("bash") is not None


def launch(command: str) -> None:
    """Run ``start_app.sh <command>`` detached from this process, after a one-second delay."""
    if command not in ("restart", "stop"):
        raise ValueError(f"unsupported command: {command}")
    subprocess.Popen(  # noqa: S603 - fixed argument list, no user input
        ["bash", "-c", 'sleep 1; exec bash "$0" "$1"', str(SCRIPT), command],
        cwd=SCRIPT.parent.parent, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)


def render_app_control() -> None:
    """Sidebar section with the restart and stop buttons."""
    if not available():
        return
    with st.expander("App: restart or stop"):
        st.caption("Restart the app if it hangs, behaves oddly or was updated. Results that "
                   "you have not downloaded and API keys you typed in are lost either way.")
        c1, c2 = st.columns(2)
        restart = c1.button("Restart", key="app_restart", width="stretch",
                            help="Stops the app and starts it again. Takes about 10 seconds.")
        stop = c2.button("Stop", key="app_stop", width="stretch",
                         help="Ends the app. Nothing keeps running in the background.")
        if restart:
            launch("restart")
            st.info("The app is restarting. Wait about 10 seconds, then reload this page "
                    "(F5).", icon=ICON_INFO)
        elif stop:
            launch("stop")
            st.info("The app is stopping. You can close this browser tab. To start it "
                    f"again, press Ctrl+Shift+B in VS Code or run `{START_HINT}` in a "
                    "terminal.", icon=ICON_INFO)
