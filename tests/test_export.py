"""Export safety: formula injection, secret removal, valid files."""

import io
import json

import pandas as pd

from src.export import build_run_config, run_config_json, to_csv_bytes, to_excel_bytes

DF = pd.DataFrame({
    "model": ["m"],
    "raw_output": ['=HYPERLINK("http://evil","click")'],
    "parsed_value_num": [-5.0],
    "error": ["AuthenticationError: sk-proj-abcdefghijklmnopqrstuvwxyz"],
})


def test_csv_escapes_formulas_and_masks_keys():
    text = to_csv_bytes(DF).decode("utf-8-sig")
    assert "'=HYPERLINK" in text
    assert "sk-proj-abc" not in text
    back = pd.read_csv(io.StringIO(text))
    assert back["parsed_value_num"].iloc[0] == -5.0  # numbers are untouched


def test_column_headers_are_escaped_too():
    pivot = pd.DataFrame({"model": ["m"], "=HYPERLINK(\"http://evil\")": [1], "+cmd": [2]})
    header = to_csv_bytes(pivot).decode("utf-8-sig").splitlines()[0]
    assert "'=HYPERLINK" in header and "'+cmd" in header
    sheet = pd.read_excel(io.BytesIO(to_excel_bytes(DF, {"pivot": pivot})), sheet_name="pivot")
    assert all(not str(c).startswith(("=", "+")) for c in sheet.columns)


def test_excel_roundtrip_with_config():
    cfg = build_run_config(models=["m"], api_keys={"openai": "sk-x"}, temperature=0.7)
    data = to_excel_bytes(DF, {"per_model_table": DF[["model"]]}, cfg)
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert set(sheets) == {"df_results", "per_model_table", "run_config"}
    assert sheets["df_results"]["raw_output"].iloc[0].startswith("'=")
    assert "api_keys" not in set(sheets["run_config"]["setting"])


def test_run_config_has_no_secrets():
    cfg = build_run_config(
        models=[{"id": "x", "api_key_env": "MY_KEY", "api_key": "sk-secret-value-123456"}],
        notes="token sk-proj-zzzzzzzzzzzzzzzzzzzzzzzz",
        openai_api_key="sk-abc",
    )
    text = run_config_json(cfg).decode()
    assert "sk-secret" not in text and "sk-proj-zzz" not in text and "sk-abc" not in text
    parsed = json.loads(text)
    assert parsed["models"][0]["api_key_env"] == "MY_KEY"
    assert "openai_api_key" not in parsed
