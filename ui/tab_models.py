"""Models tab - install free local models (Ollama) or connect your own model."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ENDPOINT_PRESETS, GGUF_UPLOAD_LIMIT_BYTES, LOCAL_MODEL_CATALOG
from src.local_models import (
    CHAT_TEMPLATES,
    OllamaError,
    delete,
    fit_status,
    hardware_info,
    import_gguf,
    list_installed,
    ollama_status,
    pull,
    storage_summary,
)
from src.model_registry import (
    CustomModelSpec,
    append_custom_model,
    apply_probe,
    load_custom_registry,
    probe,
    remove_custom_model,
)
from src.security import is_valid_ollama_tag, mask_secrets
from ui.models_catalog import _ollama_models
from ui.state import ICON_INFO

LIGHT = {"fits": "yes", "tight": "tight", "no": "too large"}


def _refresh() -> None:
    _ollama_models.clear()
    st.rerun()


def _pull_with_progress(tag: str) -> None:
    bar = st.progress(0.0, text=f"Downloading {tag}…")
    try:
        for ev in pull(tag):
            frac = ev.fraction
            bar.progress(frac if frac is not None else 0.0, text=f"{tag}: {ev.status}")
        bar.progress(1.0, text=f"{tag}: done")
        st.success(f"Installed {tag}.")
    except OllamaError as exc:
        st.error(str(exc))


def _ollama_section() -> None:
    st.markdown("### Free models that run on this machine")
    st.caption("Ollama is a free program that runs openly available AI models right here, "
               "without an account and without costs per request. They are slower and "
               "weaker than the big cloud models, but fine for trying things out.")
    status = ollama_status()
    if not status.reachable:
        st.warning(status.message)
        st.code("bash scripts/install_ollama.sh   # pinned version, checksum-verified, "
                "loopback only", language="bash")
        st.caption("Ollama is not installed yet. Open a terminal (in Codespaces: menu "
                   "**Terminal > New Terminal**), paste the line above and press Enter. When "
                   "it has finished, click **Check again**. On a standard Codespace, choose "
                   "small models (up to about 3B parameters).")
        if st.button("Check again", key="refresh_ollama_down"):
            _refresh()
        return
    st.success(status.message)
    hw = hardware_info()
    installed = list_installed()
    used, quota = storage_summary(installed)
    c1, c2, c3 = st.columns(3)
    c1.metric("Memory (RAM)", f"{hw.total_ram_gb or 0:.1f} GB")
    c2.metric("Free disk", f"{hw.free_disk_gb:.1f} GB")
    c3.metric("Models on disk", f"{used:.1f} GB", help=f"Free Codespaces storage: {quota:.0f} GB "
                                                     "per month for all your codespaces.")

    st.markdown("**Installed models**")
    if not installed:
        st.caption("None yet. Install one from the list below.")
    for m in installed:
        a, b, c = st.columns([5, 2, 2])
        a.markdown(f"`{m.name}` · {m.parameter_size or '?'} · {m.quantization or ''}")
        b.caption(f"{m.size_gb:.2f} GB")
        if c.button("Delete", key=f"del_{m.name}",
                    help=f"Delete {m.name} from this machine"):
            try:
                delete(m.name)
                _refresh()
            except OllamaError as exc:
                st.error(str(exc))

    st.markdown("**Models to start with** (small enough to run without a graphics card; "
                "names checked on 2026-11-08, they may change)")
    names = {m.name for m in installed}
    rows = []
    for info in LOCAL_MODEL_CATALOG:
        light, why = fit_status(info, hw)
        rows.append({"fits?": LIGHT[light], "model": info.tag,
                     "size": f"{info.params_b:g}B parameters",
                     "download": f"{info.approx_disk_gb:.2f} GB", "license": info.license,
                     "open source": "yes" if info.open_source_osi else "weights only",
                     "will it run here?": why,
                     "installed": "yes" if info.tag in names else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    choice = st.selectbox("Model to install", [i.tag for i in LOCAL_MODEL_CATALOG
                                                   if i.tag not in names] or ["(all installed)"])
    if st.button("Install this model", disabled=choice == "(all installed)"):
        _pull_with_progress(choice)
        _ollama_models.clear()
    st.caption("The first column says whether a model fits this machine. The models come "
               "from third parties under their own licenses; this project does not "
               "redistribute them.")
    if st.button("Refresh the list"):
        _refresh()


def _byom_pull_path(kind: str) -> None:
    hint = "e.g. qwen3:1.7b" if kind == "tag" else \
        "e.g. hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
    tag = st.text_input("Model", placeholder=hint, key=f"byom_{kind}").strip()
    if tag and not is_valid_ollama_tag(tag):
        st.error("Invalid name. Allowed: letters, digits, '.', '_', '-', '/', optional ':tag'.")
        return
    ok = st.checkbox("I checked the model's license and that it fits my disk/RAM.",
                     key=f"byom_ok_{kind}")
    if st.button("Download", key=f"byom_pull_{kind}", disabled=not (tag and ok)):
        if not ollama_status().reachable:
            st.error("Ollama is not running - see the section above.")
            return
        _pull_with_progress(tag)
        _ollama_models.clear()


def _byom_gguf() -> None:
    st.caption(f"Upload a .gguf file (max {GGUF_UPLOAD_LIMIT_BYTES / 1024**3:.0f} GB) or, for "
               "big files, use: `bash scripts/import_gguf.sh path/to/model.gguf my-model`")
    up = st.file_uploader("Model file (.gguf)", type=["gguf"], key="byom_gguf_file")
    name = st.text_input("Name for the model", placeholder="my-finetune", key="byom_gguf_name")
    tmpl = st.selectbox("Chat template", CHAT_TEMPLATES, key="byom_tmpl",
                        help="'auto' reads it from the GGUF metadata. Wrong templates are the "
                             "#1 cause of garbage output - try another if answers look odd.")
    if st.button("Import into Ollama", disabled=not (up and name)):
        if up.size > GGUF_UPLOAD_LIMIT_BYTES:
            st.error("File too large for upload - use scripts/import_gguf.sh instead.")
            return
        tmpdir = tempfile.mkdtemp(prefix="llmnoise-gguf-")  # outside the repository
        try:
            path = Path(tmpdir) / "upload.gguf"
            with open(path, "wb") as fh:
                shutil.copyfileobj(up, fh)
            with st.spinner("Creating Ollama model…"):
                created = import_gguf(path, name, tmpl)
            st.success(f"Created `{created}`. You can now select it in the sidebar (switch "
                       "practice mode off first).")
            _ollama_models.clear()
        except OllamaError as exc:
            st.error(str(exc))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)



def _byom_endpoint() -> None:
    ss = st.session_state
    preset = st.selectbox("Server type (pre-fills the URL)", ENDPOINT_PRESETS,
                          format_func=lambda p: p.name, key="byom_preset")
    if preset.note:
        st.caption(preset.note)
    c1, c2 = st.columns(2)
    mid = c1.text_input("Short name of your choice", placeholder="my-llamacpp", key="byom_id")
    model = c2.text_input("Model name the server expects", key="byom_model")
    base = st.text_input("Address of the server (base URL)", preset.api_base,
                         key=f"byom_base_{preset.name}")
    with st.expander("Advanced"):
        key_env = st.text_input("Name of the environment variable that holds the API key "
                                "(optional; the name, not the key itself)", key="byom_env")
        conc = st.number_input("Requests sent at the same time", 1, 32, 1, key="byom_conc")
        timeout = st.number_input("Seconds to wait for an answer", 5, 3600, 120,
                                  key="byom_timeout")
        public = st.checkbox("Allow public endpoint (prompts leave this machine)",
                             key="byom_public")
    try:
        spec = CustomModelSpec(id=mid or "x", model=model or "x", api_base=base,
                               api_key_env=key_env, max_concurrency=int(conc),
                               timeout_s=float(timeout), allow_public_endpoint=public)
        spec.checked_api_base()
    except ValueError as exc:
        st.error(mask_secrets(str(exc).strip().splitlines()[-1]))
        return
    valid = bool(mid and model)
    b1, b2, b3 = st.columns(3)
    if b1.button("Test the connection", disabled=not valid):
        with st.spinner("Testing the connection…"):
            ss[f"probe_{mid}"] = asyncio.run(probe(spec.to_model_spec()))
    rep = ss.get(f"probe_{mid}")
    if rep is not None:
        yn = {True: "yes", False: "no", None: "unknown"}
        st.markdown(
            f"Answers: **{yn[rep.completion_ok]}** · system role: {yn[rep.system_role]} · "
            f"seed reproducible: {yn[rep.seed_deterministic]} · JSON mode: {yn[rep.json_mode]} · "
            f"logprobs: {yn[rep.logprobs]} · response time: {rep.latency_s} s")
        for m in rep.messages:
            st.caption(m)
    if b2.button("Use until I close the app", disabled=not valid):
        ms = spec.to_model_spec()
        if rep is not None and rep.completion_ok:
            ms = apply_probe(ms, rep)
        ss.session_custom_specs = [s for s in ss.session_custom_specs if s["id"] != ms.id]
        ss.session_custom_specs.append(ms.model_dump())
        st.success("Added. Select it in the sidebar (switch practice mode off first).")
    if b3.button("Save permanently", disabled=not valid):
        try:
            append_custom_model(spec)
            st.success("Saved to models/models.toml (git-ignored).")
        except ValueError as exc:
            st.error(str(exc))


def _registry_section() -> None:
    specs, errors = load_custom_registry()
    st.markdown("### Saved models (file models/models.toml)")
    if not specs and not errors:
        st.caption("Nothing saved yet. Models you save permanently above are listed here. "
                   "The file can also be edited by hand; `models/models.example.toml` is a "
                   "template.")
    for e in errors:
        st.error(str(e))
    for s in specs:
        a, b = st.columns([5, 1])
        a.markdown(f"`custom/{s.id}`: model `{s.model}` at {s.api_base}")
        if b.button("Remove", key=f"rm_{s.id}"):
            remove_custom_model(s.id)
            st.rerun()
    if st.button("Reload the file"):
        st.rerun()


def render_models_tab() -> None:
    st.info("You do not need this page to try the app. The practice models work without "
            "any setup. Come here when you want to test real models for free on this "
            "machine, or connect a model of your own. Cloud models from OpenAI, Anthropic, "
            "Google or OpenRouter only need an API key, which you enter in the sidebar.",
            icon=ICON_INFO)
    _ollama_section()
    st.divider()
    st.markdown("### Connect a model of your own")
    st.caption("For people who already know which model they want. Four ways, none of them "
               "needs programming. Step-by-step guide: docs/BRING_YOUR_OWN_MODEL.md")
    t1, t2, t3, t4 = st.tabs(["Name from ollama.com", "Model from Hugging Face",
                              "Model file (.gguf)", "A model server that is already running"])
    with t1:
        _byom_pull_path("tag")
    with t2:
        _byom_pull_path("hf")
    with t3:
        _byom_gguf()
    with t4:
        _byom_endpoint()
    st.divider()
    _registry_section()

