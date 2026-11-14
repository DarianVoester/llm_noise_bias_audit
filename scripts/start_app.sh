#!/usr/bin/env bash
# Start, stop or restart the app. It runs in the background, so you can close the terminal.
#
#   bash scripts/start_app.sh            start it (does nothing if it already runs)
#   bash scripts/start_app.sh stop       stop it
#   bash scripts/start_app.sh restart    stop it and start it again
#   bash scripts/start_app.sh status     say whether it is running
#   bash scripts/start_app.sh log        show what the app has written to its log
#
# The same actions are available as VS Code tasks ("App: Start", "App: Stop", ...) and as
# buttons in the app's sidebar. Used by .devcontainer/devcontainer.json to start the app
# when a Codespace opens. In Codespaces the forwarded port 8501 stays PRIVATE.
#
# Settings (optional): STREAMLIT_PORT (default 8501),
#   STREAMLIT_LOG (default ~/.local/state/llm-noise-audit/streamlit.log)
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${STREAMLIT_PORT:-8501}"
# The log lives in a private directory of the user, not in the shared /tmp (where another
# local user could read it or plant a symlink under the predictable name).
umask 077
LOG="${STREAMLIT_LOG:-${XDG_STATE_HOME:-${HOME}/.local/state}/llm-noise-audit/streamlit.log}"
mkdir -p "$(dirname "${LOG}")"
HEALTH_URL="http://127.0.0.1:${PORT}/_stcore/health"

is_up() { curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; }

# Process ids of the app: whatever listens on the port, plus Streamlit processes that have
# not opened the port yet. The pattern is anchored to the start of the command line so that
# it can never match an unrelated command that merely mentions the app.
app_pids() {
  {
    if command -v ss >/dev/null 2>&1; then
      ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2
    fi
    pgrep -f '^[^ ]*python[^ ]* (-m )?[^ ]*streamlit run app\.py' 2>/dev/null
  } | sort -u
}
is_alive() { [[ -n "$(app_pids)" ]]; }
kill_app() { local pids; pids="$(app_pids)"; [[ -z "${pids}" ]] || kill "$@" ${pids} 2>/dev/null || true; }

where_to_open() {
  if [[ -n "${CODESPACE_NAME:-}" && -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]]; then
    echo "Open it here: https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
    echo "(or in VS Code: PORTS panel, globe icon next to port ${PORT}; keep the port private)"
  else
    echo "Open it here: http://localhost:${PORT}"
  fi
}

stop_app() {
  if ! is_alive; then
    echo "The app is not running."
    return 0
  fi
  echo "Stopping the app ..."
  kill_app
  for _ in $(seq 1 20); do
    is_alive || { echo "The app has stopped."; return 0; }
    sleep 0.5
  done
  kill_app -9   # it did not react within 10 s
  sleep 0.5
  echo "The app has stopped."
}

start_app() {
  if is_up; then
    echo "The app is already running."
    where_to_open
    return 0
  fi
  if is_alive; then   # a process exists but does not answer: replace it
    echo "The app does not respond. Starting it afresh ..."
    stop_app
  fi
  if ! python -c "import streamlit" >/dev/null 2>&1; then
    echo "The app is not installed yet. Run this once, then try again:"
    echo "  pip install --require-hashes -r requirements.txt"
    return 1
  fi
  echo "Starting the app (this takes a few seconds) ..."
  # Loopback only: the app has no login (see SECURITY.md, port policy).
  setsid nohup python -m streamlit run app.py --server.port "${PORT}" \
    --server.address 127.0.0.1 >"${LOG}" 2>&1 </dev/null &
  for _ in $(seq 1 60); do
    if is_up; then
      echo "The app is running."
      where_to_open
      return 0
    fi
    sleep 0.5
  done
  echo "The app did not start within 30 seconds. The last lines of its log (${LOG}):"
  tail -n 20 "${LOG}" || true
  return 1
}

case "${1:-start}" in
  start)            start_app ;;
  stop|--stop)      stop_app ;;
  restart)          stop_app; start_app ;;
  status)
    if is_up; then echo "The app is running."; where_to_open
    elif is_alive; then echo "The app was started but does not respond. Try: bash scripts/start_app.sh restart"
    else echo "The app is not running. Start it with: bash scripts/start_app.sh"
    fi ;;
  log)
    if [[ -f "${LOG}" ]]; then tail -n 60 "${LOG}"; else echo "There is no log yet (${LOG})."; fi ;;
  *)
    echo "Unknown command: $1"
    echo "Use one of: start, stop, restart, status, log"
    exit 2 ;;
esac
