"""Safe CSV / Excel export and reproducible run configuration."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src import __version__
from src.security import escape_spreadsheet_cell, mask_secrets

# Never export anything whose key looks like a credential.
_FORBIDDEN_CONFIG_KEYS = ("key", "token", "secret", "password", "authorization")


def _safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        # pandas >= 3 uses a dedicated string dtype; check for non-numeric columns instead
        if not pd.api.types.is_numeric_dtype(out[col]) and \
                not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(escape_spreadsheet_cell).astype(object)
    if "error" in out.columns:
        out["error"] = out["error"].map(lambda v: mask_secrets(v) if isinstance(v, str) else v)
    # Headers too: analysis tables can have model output as column names (pivot tables).
    out.columns = [escape_spreadsheet_cell(c) for c in out.columns]
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV with BOM (opens correctly in Excel) and formula injection escaped."""
    return _safe_frame(df).to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(
    df: pd.DataFrame,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
    run_config: dict[str, Any] | None = None,
) -> bytes:
    """XLSX workbook: results, optional analysis tables and the run configuration."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _safe_frame(df).to_excel(writer, sheet_name="df_results", index=False)
        for name, sheet in (extra_sheets or {}).items():
            safe_name = "".join(c for c in name if c.isalnum() or c in " _-")[:31] or "sheet"
            _safe_frame(sheet).to_excel(writer, sheet_name=safe_name, index=False)
        if run_config:
            cfg = pd.DataFrame(
                [(k, json.dumps(v, default=str)) for k, v in sanitize_config(run_config).items()],
                columns=["setting", "value"],
            )
            _safe_frame(cfg).to_excel(writer, sheet_name="run_config", index=False)
    return buf.getvalue()


def sanitize_config(cfg: Any) -> Any:
    """Recursively drop credential-like keys and mask credential-like values."""
    if isinstance(cfg, dict):
        return {
            k: sanitize_config(v)
            for k, v in cfg.items()
            if not any(f in str(k).lower() for f in _FORBIDDEN_CONFIG_KEYS)
            or str(k).lower() == "api_key_env"
        }
    if isinstance(cfg, list):
        return [sanitize_config(v) for v in cfg]
    if isinstance(cfg, str):
        return mask_secrets(cfg)
    return cfg


def build_run_config(**parts: Any) -> dict[str, Any]:
    """Assemble a reproducible, key-free description of a run."""
    cfg = {
        "app": "LLM Noise & Bias Audit Studio",
        "app_version": __version__,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **parts,
    }
    return sanitize_config(cfg)


def run_config_json(cfg: dict[str, Any]) -> bytes:
    return json.dumps(sanitize_config(cfg), indent=2, default=str).encode("utf-8")
