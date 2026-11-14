#!/usr/bin/env bash
# Install a pinned, checksum-verified Ollama release into ~/.local and start it on
# the loopback interface only. Opt-in: cloud-only users never need this.
#
# Usage:  bash scripts/install_ollama.sh           # install + start
#         bash scripts/install_ollama.sh --no-start
#
# Upgrade policy (SECURITY.md): bump OLLAMA_VERSION + both checksums together, only to
# releases that are at least 7 days old. Checksums come from the release's
# sha256sum.txt on https://github.com/ollama/ollama/releases
set -euo pipefail

OLLAMA_VERSION="0.40.13"   # released 2026-10-28
SHA256_AMD64="c9490e0f520be21271d969e6703f012268b1d39f8b70dd58854e5bfef7a75bb6"
SHA256_ARM64="913d9a28da6da86d551ce4ca531457767bb14d511cbc1d5ca93e9b3b8b9a8666"

PREFIX="${HOME}/.local"
HOST="127.0.0.1:11434"
START=1
[[ "${1:-}" == "--no-start" ]] && START=0

case "$(uname -m)" in
  x86_64|amd64) ARCH="amd64"; SHA="${SHA256_AMD64}" ;;
  aarch64|arm64) ARCH="arm64"; SHA="${SHA256_ARM64}" ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd is required. Installing via apt (needs sudo)..."
  # 'apt-get update' exits non-zero if ANY repository fails (e.g. the Yarn repo in some
  # devcontainer base images has an expired/missing GPG key). The Debian indexes are
  # still refreshed in that case, so a failed update must not stop the install.
  if ! sudo apt-get update -qq; then
    echo "Note: 'apt-get update' reported errors for a third-party repository (often Yarn)."
    echo "      Continuing - zstd comes from the official Debian repository."
  fi
  sudo apt-get install -y -qq zstd || true
  if ! command -v zstd >/dev/null 2>&1; then
    echo "ERROR: zstd could not be installed, so the Ollama archive cannot be unpacked."
    echo "Try manually:  sudo apt-get install -y zstd"
    echo "If apt complains about the Yarn repository, disable it first:"
    echo "  sudo mv /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.disabled"
    exit 1
  fi
fi

ARCHIVE="ollama-linux-${ARCH}.tar.zst"
URL="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/${ARCHIVE}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Downloading Ollama v${OLLAMA_VERSION} (${ARCH})..."
curl -fL --retry 3 --progress-bar -o "${TMP}/${ARCHIVE}" "${URL}"

echo "Verifying SHA-256..."
echo "${SHA}  ${TMP}/${ARCHIVE}" | sha256sum -c - || {
  echo "CHECKSUM MISMATCH - aborting. Do not use this download."; exit 1; }

mkdir -p "${PREFIX}"
if ! zstd -dc "${TMP}/${ARCHIVE}" | tar -xf - -C "${PREFIX}"; then
  echo "ERROR: unpacking the Ollama archive failed (see the messages above)."
  exit 1
fi
echo "Installed: $("${PREFIX}/bin/ollama" --version 2>/dev/null || echo "${PREFIX}/bin/ollama")"

case ":${PATH}:" in
  *":${PREFIX}/bin:"*) ;;
  *) echo "export PATH=\"${PREFIX}/bin:\$PATH\"" >> "${HOME}/.bashrc"
     export PATH="${PREFIX}/bin:${PATH}"
     echo "Added ${PREFIX}/bin to PATH (open a new terminal or 'source ~/.bashrc')." ;;
esac

if [[ "${START}" == "1" ]]; then
  if curl -fs "http://${HOST}/api/version" >/dev/null 2>&1; then
    echo "Ollama is already running on ${HOST}."
  else
    echo "Starting Ollama on ${HOST} (loopback only; port is NOT forwarded)..."
    OLLAMA_HOST="${HOST}" nohup "${PREFIX}/bin/ollama" serve >"${HOME}/.ollama-serve.log" 2>&1 &
    for _ in $(seq 1 30); do
      curl -fs "http://${HOST}/api/version" >/dev/null 2>&1 && break
      sleep 1
    done
    curl -fs "http://${HOST}/api/version" && echo && echo "Ollama is running." || {
      echo "Ollama did not start; see ~/.ollama-serve.log"; exit 1; }
  fi
fi

echo
echo "Next: pull a small model, e.g.  bash scripts/pull_starter_model.sh gemma3:270m"
echo "      or use the 'Models' tab in the app."
