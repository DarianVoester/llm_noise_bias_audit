"""Local Ollama management: status, installed models, pull, delete, GGUF import, hardware.

Talks to the Ollama HTTP API on the loopback interface only. Never shells out with
string interpolation; ``ollama create`` (GGUF import) uses an argument list.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.config import (
    CODESPACES_FREE_STORAGE_GB,
    LOCAL_MODEL_CATALOG,
    MODELFILE_TEMPLATE_DIR,
    OLLAMA_API_BASE,
    LocalModelInfo,
)
from src.model_spec import ModelSource, ModelSpec
from src.security import check_endpoint_url, is_valid_model_name, is_valid_ollama_tag

GGUF_MAGIC = b"GGUF"
CHAT_TEMPLATES = ("auto", "chatml", "llama3", "gemma", "mistral")


class OllamaError(RuntimeError):
    """Raised for Ollama API failures with a user-friendly message."""


@dataclass
class OllamaStatus:
    reachable: bool
    version: str | None
    base_url: str
    message: str


@dataclass
class InstalledModel:
    name: str
    size_bytes: int
    family: str | None
    parameter_size: str | None
    quantization: str | None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3


@dataclass
class PullProgress:
    status: str
    completed: int | None = None
    total: int | None = None

    @property
    def fraction(self) -> float | None:
        if self.completed is not None and self.total:
            return min(1.0, self.completed / self.total)
        return None


def _base(base_url: str | None) -> str:
    return check_endpoint_url(base_url or OLLAMA_API_BASE)


def _client(timeout: float | httpx.Timeout) -> httpx.Client:
    """Client for the local Ollama server.

    ``trust_env=False``: proxy variables (HTTP_PROXY, ALL_PROXY) and ~/.netrc are ignored,
    so requests to the loopback server are never routed through a proxy.
    """
    return httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False)


def ollama_status(base_url: str | None = None, client: httpx.Client | None = None) -> OllamaStatus:
    base = _base(base_url)
    c = client or _client(3.0)
    try:
        r = c.get(f"{base}/api/version")
        r.raise_for_status()
        version = r.json().get("version")
        return OllamaStatus(True, version, base, f"Ollama {version} is running.")
    except (httpx.HTTPError, ValueError):
        installed = shutil.which("ollama") is not None
        hint = ("Ollama is installed but not running. Start it with: "
                "OLLAMA_HOST=127.0.0.1:11434 ollama serve &") if installed else \
            "Ollama is not installed. Run: bash scripts/install_ollama.sh"
        return OllamaStatus(False, None, base, hint)
    finally:
        if client is None:
            c.close()


def list_installed(base_url: str | None = None,
                   client: httpx.Client | None = None) -> list[InstalledModel]:
    base = _base(base_url)
    c = client or _client(10.0)
    try:
        r = c.get(f"{base}/api/tags")
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            details = m.get("details") or {}
            out.append(InstalledModel(
                name=m.get("name") or m.get("model", "?"),
                size_bytes=int(m.get("size") or 0),
                family=details.get("family"),
                parameter_size=details.get("parameter_size"),
                quantization=details.get("quantization_level"),
            ))
        return sorted(out, key=lambda x: x.name)
    except httpx.HTTPError as exc:
        raise OllamaError(f"Could not list models: {exc}") from exc
    finally:
        if client is None:
            c.close()


def pull(tag: str, base_url: str | None = None,
         client: httpx.Client | None = None) -> Iterator[PullProgress]:
    """Stream download progress of a model (Ollama library tag or hf.co/... GGUF path)."""
    if not is_valid_ollama_tag(tag):
        raise OllamaError(f"Invalid model tag: {tag!r}")
    base = _base(base_url)
    c = client or _client(httpx.Timeout(10.0, read=None))
    try:
        with c.stream("POST", f"{base}/api/pull", json={"model": tag, "stream": True}) as r:
            if r.status_code >= 400:
                raise OllamaError(f"Pull failed ({r.status_code}): {r.read().decode()[:300]}")
            import json

            for line in r.iter_lines():
                if not line:
                    continue
                data: dict[str, Any] = json.loads(line)
                if "error" in data:
                    raise OllamaError(str(data["error"]))
                yield PullProgress(str(data.get("status", "")), data.get("completed"),
                                   data.get("total"))
    except httpx.HTTPError as exc:
        raise OllamaError(f"Pull failed: {exc}") from exc
    finally:
        if client is None:
            c.close()


def delete(tag: str, base_url: str | None = None, client: httpx.Client | None = None) -> None:
    if not is_valid_ollama_tag(tag):
        raise OllamaError(f"Invalid model tag: {tag!r}")
    base = _base(base_url)
    c = client or _client(30.0)
    try:
        r = c.request("DELETE", f"{base}/api/delete", json={"model": tag})
        if r.status_code >= 400:
            raise OllamaError(f"Delete failed ({r.status_code}): {r.text[:200]}")
    finally:
        if client is None:
            c.close()


# --- GGUF import -----------------------------------------------------------

def is_gguf_file(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def render_modelfile(gguf_path: Path, template: str = "auto",
                     template_dir: Path = MODELFILE_TEMPLATE_DIR,
                     adapter_path: Path | None = None) -> str:
    """Build an Ollama Modelfile. 'auto' lets Ollama read the chat template from GGUF metadata."""
    if template not in CHAT_TEMPLATES:
        raise OllamaError(f"Unknown chat template {template!r}; choose one of {CHAT_TEMPLATES}")
    for p in (gguf_path, adapter_path):
        # A line break in the path would add further instructions to the Modelfile.
        if p is not None and any(ch in str(p.resolve()) for ch in '\n\r"'):
            raise OllamaError("The file path must not contain line breaks or double quotes.")
    lines = [f"FROM {gguf_path.resolve()}"]
    if adapter_path is not None:
        lines.append(f"ADAPTER {adapter_path.resolve()}")
    if template != "auto":
        tmpl_file = template_dir / f"{template}.tmpl"
        if not tmpl_file.is_file():
            raise OllamaError(f"Template file not found: {tmpl_file}")
        lines.append(tmpl_file.read_text(encoding="utf-8").strip())
    return "\n".join(lines) + "\n"


def import_gguf(gguf_path: Path, name: str, template: str = "auto",
                adapter_path: Path | None = None, timeout_s: float = 1800.0) -> str:
    """Register a local .gguf file as an Ollama model via ``ollama create``."""
    if not is_valid_model_name(name) or not is_valid_ollama_tag(name.lower()):
        raise OllamaError(f"Invalid model name {name!r} (use lowercase letters, digits, '-', "
                          "'_', '.', optional ':tag')")
    if gguf_path.suffix.lower() != ".gguf" or not is_gguf_file(gguf_path):
        raise OllamaError("File is not a valid GGUF model (missing GGUF header).")
    if adapter_path is not None and not is_gguf_file(adapter_path):
        raise OllamaError("Adapter file is not a valid GGUF file (missing GGUF header).")
    exe = shutil.which("ollama")
    if exe is None:
        raise OllamaError("The 'ollama' command is not installed. Run scripts/install_ollama.sh")
    modelfile = render_modelfile(gguf_path, template, adapter_path=adapter_path)
    with tempfile.TemporaryDirectory(prefix="llmnoise-modelfile-") as tmp:
        mf_path = Path(tmp) / "Modelfile"
        mf_path.write_text(modelfile, encoding="utf-8")
        proc = subprocess.run([exe, "create", name.lower(), "-f", str(mf_path)],
                              capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise OllamaError(f"ollama create failed: {(proc.stderr or proc.stdout)[-400:]}")
    return name.lower()


# --- Hardware check --------------------------------------------------------

@dataclass
class HardwareInfo:
    total_ram_gb: float | None
    available_ram_gb: float | None
    free_disk_gb: float
    cpu_count: int


def hardware_info(path: str = ".") -> HardwareInfo:
    total = avail = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = {line.split(":")[0]: int(line.split()[1]) for line in fh if ":" in line}
        total = info.get("MemTotal", 0) / 1024**2
        avail = info.get("MemAvailable", 0) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    free = shutil.disk_usage(path).free / 1024**3
    return HardwareInfo(total, avail, free, os.cpu_count() or 1)


def fit_status(model: LocalModelInfo, hw: HardwareInfo) -> tuple[str, str]:
    """Traffic light: ('fits'|'tight'|'no', explanation)."""
    if hw.free_disk_gb < model.approx_disk_gb * 1.1:
        return "no", f"needs ~{model.approx_disk_gb:.1f} GB disk, {hw.free_disk_gb:.1f} GB free"
    ram = hw.total_ram_gb or 0.0
    if ram and ram < model.min_ram_gb:
        return "no", f"needs ~{model.min_ram_gb:.0f} GB RAM, machine has {ram:.1f} GB"
    if ram and ram < model.min_ram_gb * 1.6:
        return "tight", "will run, but slowly and may compete with the app for memory"
    return "fits", "should run comfortably on CPU"


def storage_summary(installed: list[InstalledModel]) -> tuple[float, float]:
    """(total installed model size in GB, Codespaces free-tier storage quota in GB)."""
    return sum(m.size_gb for m in installed), CODESPACES_FREE_STORAGE_GB


def catalog_entry(tag: str) -> LocalModelInfo | None:
    return next((m for m in LOCAL_MODEL_CATALOG if m.tag == tag), None)


def spec_for_installed(model: InstalledModel, base_url: str | None = None) -> ModelSpec:
    """ModelSpec for an installed Ollama model (Ollama's OpenAI-compatible /v1 API)."""
    cat = catalog_entry(model.name)
    return ModelSpec(
        id=f"ollama/{model.name}",
        display_name=(cat.display_name if cat else model.name) + " (local)",
        source=ModelSource.OLLAMA,
        model_route=f"ollama_chat/{model.name}",
        api_base=_base(base_url),
        supports_seed=True,
        supports_json_mode=True,
        max_concurrency=2,
        timeout_s=300.0,
        license=cat.license if cat else None,
        notes=cat.notes if cat else None,
    )

