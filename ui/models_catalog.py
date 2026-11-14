"""Builds the list of selectable models from all sources (cached briefly)."""

from __future__ import annotations

import streamlit as st

from src.llm_runner import mock_model_specs
from src.local_models import OllamaError, list_installed, ollama_status, spec_for_installed
from src.model_registry import cloud_model_specs, custom_cloud_spec, load_custom_registry
from src.model_spec import ModelSource, ModelSpec

SOURCE_LABELS = {
    ModelSource.MOCK: "practice model (free)",
    ModelSource.CLOUD: "cloud model",
    ModelSource.OLLAMA: "local model",
    ModelSource.CUSTOM: "your own model",
}


@st.cache_data(ttl=10, show_spinner=False)
def _ollama_models() -> tuple[list[dict], str | None]:
    st_ = ollama_status()
    if not st_.reachable:
        return [], st_.message
    try:
        return [spec_for_installed(m).model_dump() for m in list_installed()], None
    except OllamaError as exc:
        return [], str(exc)


def available_models() -> tuple[list[ModelSpec], list[str]]:
    """All usable models plus human-readable notes (e.g. registry errors)."""
    notes: list[str] = []
    models: list[ModelSpec] = list(mock_model_specs())
    keys = st.session_state.get("api_keys", {})
    models += cloud_model_specs(keys)
    for mid in st.session_state.get("extra_cloud_ids", []):
        try:
            models.append(custom_cloud_spec(mid))
        except ValueError as exc:
            notes.append(f"{mid}: {exc}")
    raw, msg = _ollama_models()
    models += [ModelSpec.model_validate(m) for m in raw]
    if msg:
        notes.append(f"Ollama: {msg}")
    specs, errors = load_custom_registry()
    for spec in specs:
        try:
            models.append(spec.to_model_spec())
        except ValueError as exc:
            notes.append(f"custom/{spec.id}: {exc}")
    notes += [f"models.toml - {e}" for e in errors]
    for spec in st.session_state.get("session_custom_specs", []):
        models.append(ModelSpec.model_validate(spec))
    probes = st.session_state.get("probe_results", {})
    adapted = []
    seen: set[str] = set()
    for m in models:
        if m.id in seen:
            continue
        seen.add(m.id)
        if m.id in probes:
            m = m.model_copy(update=probes[m.id])
        adapted.append(m)
    return adapted, notes


def label(spec: ModelSpec) -> str:
    return f"{spec.display_name}  ·  {SOURCE_LABELS[spec.source]}"
