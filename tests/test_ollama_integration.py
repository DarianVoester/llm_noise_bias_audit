"""Optional end-to-end test against a real local Ollama server.

Skipped by default (pytest marker 'ollama'). Run with:
    bash scripts/install_ollama.sh && bash scripts/pull_starter_model.sh gemma3:270m
    pytest -m ollama
"""

import asyncio

import pytest

from src.cases import single_case
from src.context_engine import VariationConfig, build_windows
from src.llm_runner import RunParams, run_audit
from src.local_models import list_installed, ollama_status, spec_for_installed
from src.schemas import SchemaConfig, SchemaType


@pytest.mark.ollama
def test_real_ollama_roundtrip():
    st = ollama_status()
    if not st.reachable:
        pytest.skip(st.message)
    models = [m for m in list_installed() if m.name.startswith("gemma3:270m")]
    if not models:
        pytest.skip("gemma3:270m not installed")
    schema = SchemaConfig(type=SchemaType.PERCENTAGE)
    windows = build_windows(single_case("What percentage of Earth's surface is water?", 71.0),
                            VariationConfig(), schema, n=3, design_seed=1)
    df = asyncio.run(run_audit(windows, [spec_for_installed(models[0])], schema,
                               RunParams(temperature=0.8, max_tokens=64)))
    assert len(df) == 3
    assert df["error"].isna().all() or df["error"].str.startswith("parse").all(), df["error"]
    assert (df["model_source"] == "ollama").all()
