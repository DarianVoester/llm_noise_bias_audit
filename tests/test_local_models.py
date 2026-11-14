"""Ollama manager tests against a mocked HTTP API (no network, no real downloads)."""

import json
from pathlib import Path

import httpx
import pytest

from src.config import LOCAL_MODEL_CATALOG
from src.local_models import (
    HardwareInfo,
    InstalledModel,
    OllamaError,
    delete,
    fit_status,
    import_gguf,
    is_gguf_file,
    list_installed,
    ollama_status,
    pull,
    render_modelfile,
    spec_for_installed,
    storage_summary,
)
from src.model_spec import ModelSource

TAGS = {"models": [
    {"name": "gemma3:270m", "size": 291_554_000,
     "details": {"family": "gemma3", "parameter_size": "268M", "quantization_level": "Q8_0"}},
    {"name": "my-finetune:latest", "size": 1_000_000_000, "details": {}},
]}


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/version":
        return httpx.Response(200, json={"version": "0.40.13"})
    if path == "/api/tags":
        return httpx.Response(200, json=TAGS)
    if path == "/api/pull":
        body = json.loads(request.content)
        if body["model"] == "does-not-exist":
            err = {"error": "pull model manifest: file does not exist"}
            return httpx.Response(200, text=json.dumps(err) + "\n")
        lines = [{"status": "pulling manifest"},
                 {"status": "downloading", "completed": 50, "total": 100},
                 {"status": "downloading", "completed": 100, "total": 100},
                 {"status": "success"}]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines) + "\n")
    if path == "/api/delete":
        return httpx.Response(200)
    return httpx.Response(404)


@pytest.fixture
def client():
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_status_reachable(client):
    st = ollama_status(client=client)
    assert st.reachable and st.version == "0.40.13" and "11434" in st.base_url


def test_status_unreachable():
    def down(request):
        raise httpx.ConnectError("refused")

    st = ollama_status(client=httpx.Client(transport=httpx.MockTransport(down)))
    assert not st.reachable and ("install_ollama" in st.message or "ollama serve" in st.message)


def test_status_rejects_public_base():
    with pytest.raises(ValueError):
        ollama_status("https://ollama.example.com")


def test_list_installed(client):
    models = list_installed(client=client)
    assert [m.name for m in models] == ["gemma3:270m", "my-finetune:latest"]
    assert models[0].parameter_size == "268M"
    used, quota = storage_summary(models)
    assert 1.0 < used < 1.5 and quota == 15.0


def test_pull_progress(client):
    events = list(pull("gemma3:270m", client=client))
    assert events[0].status == "pulling manifest" and events[-1].status == "success"
    assert events[1].fraction == 0.5


def test_pull_error_and_validation(client):
    with pytest.raises(OllamaError, match="does not exist"):
        list(pull("does-not-exist", client=client))
    with pytest.raises(OllamaError, match="Invalid"):
        list(pull("bad; rm -rf /", client=client))
    assert list(pull("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", client=client))


def test_delete(client):
    delete("gemma3:270m", client=client)
    with pytest.raises(OllamaError):
        delete("$(whoami)", client=client)


def test_spec_for_installed():
    spec = spec_for_installed(InstalledModel("gemma3:270m", 1, None, None, None))
    assert spec.source == ModelSource.OLLAMA and spec.model_route == "ollama_chat/gemma3:270m"
    assert spec.license == "Gemma Terms of Use" and spec.id == "ollama/gemma3:270m"


def test_fit_status():
    small, big = LOCAL_MODEL_CATALOG[0], LOCAL_MODEL_CATALOG[-1]
    hw = HardwareInfo(total_ram_gb=7.8, available_ram_gb=5.0, free_disk_gb=19.0, cpu_count=2)
    assert fit_status(small, hw)[0] == "fits"
    assert fit_status(big, hw)[0] == "tight"
    assert fit_status(big, HardwareInfo(4.0, 2.0, 19.0, 2))[0] == "no"
    assert fit_status(small, HardwareInfo(8.0, 5.0, 0.1, 2))[0] == "no"


def test_gguf_detection_and_modelfile(tmp_path: Path):
    good = tmp_path / "m.gguf"
    good.write_bytes(b"GGUF" + b"\x00" * 16)
    bad = tmp_path / "fake.gguf"
    bad.write_bytes(b"not a model")
    assert is_gguf_file(good) and not is_gguf_file(bad)
    mf = render_modelfile(good, "chatml")
    assert mf.startswith(f"FROM {good.resolve()}") and "<|im_start|>" in mf
    assert render_modelfile(good, "auto").strip() == f"FROM {good.resolve()}"
    assert "ADAPTER" in render_modelfile(good, "auto", adapter_path=good)
    with pytest.raises(OllamaError):
        render_modelfile(good, "unknown-template")


def test_import_gguf_validation(tmp_path: Path):
    bad = tmp_path / "fake.gguf"
    bad.write_bytes(b"nope")
    with pytest.raises(OllamaError, match="GGUF"):
        import_gguf(bad, "my-model")
    with pytest.raises(OllamaError, match="Invalid model name"):
        import_gguf(bad, "bad name; rm")


@pytest.mark.parametrize("tmpl", ["chatml", "llama3", "gemma", "mistral"])
def test_all_templates_exist(tmp_path, tmpl):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF")
    assert "TEMPLATE" in render_modelfile(g, tmpl)
