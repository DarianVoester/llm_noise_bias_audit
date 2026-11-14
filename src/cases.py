"""Case sets: one or many prompts ("vignettes") judged by every model.

A noise audit in the sense of Kahneman, Sibony & Sunstein needs several cases judged
by several judges. A single prompt still works, but Level vs. Pattern noise can then
not be separated.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import pandas as pd

MAX_CASES = 200
MAX_PROMPT_CHARS = 20_000


@dataclass(frozen=True)
class Case:
    case_id: str
    prompt: str
    ground_truth: float | None = None


class CaseSetError(ValueError):
    """Raised for invalid case files."""


def single_case(prompt: str, ground_truth: float | None = None) -> list[Case]:
    prompt = prompt.strip()
    if not prompt:
        raise CaseSetError("The prompt is empty.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise CaseSetError(f"Prompt is longer than {MAX_PROMPT_CHARS} characters.")
    return [Case("case_1", prompt, ground_truth)]


def _parse_truth(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CaseSetError(f"ground_truth value {value!r} is not a number") from exc
    return None if math.isnan(f) else f


def cases_from_dataframe(df: pd.DataFrame) -> list[Case]:
    """Build cases from a DataFrame with columns ``case_id, prompt[, ground_truth]``."""
    cols = {c.strip().lower(): c for c in df.columns}
    if "prompt" not in cols:
        raise CaseSetError("The case file needs a 'prompt' column.")
    if len(df) == 0:
        raise CaseSetError("The case file has no rows.")
    if len(df) > MAX_CASES:
        raise CaseSetError(f"At most {MAX_CASES} cases are supported.")
    cases: list[Case] = []
    seen: set[str] = set()
    for i, row in enumerate(df.itertuples(index=False), start=1):
        rec = dict(zip(df.columns, row, strict=True))
        prompt = str(rec[cols["prompt"]] or "").strip()
        if not prompt or prompt.lower() == "nan":
            raise CaseSetError(f"Row {i}: empty prompt.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise CaseSetError(f"Row {i}: prompt longer than {MAX_PROMPT_CHARS} characters.")
        raw_id = rec.get(cols["case_id"]) if "case_id" in cols else None
        case_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() \
            and str(raw_id) != "nan" else f"case_{i}"
        if case_id in seen:
            raise CaseSetError(f"Duplicate case_id '{case_id}'.")
        seen.add(case_id)
        truth = _parse_truth(rec.get(cols["ground_truth"])) if "ground_truth" in cols else None
        cases.append(Case(case_id, prompt, truth))
    return cases


def cases_from_csv(data: bytes | str) -> list[Case]:
    """Parse an uploaded CSV file (UTF-8)."""
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:  # pandas raises several error types
        raise CaseSetError(f"Could not read CSV: {exc}") from exc
    return cases_from_dataframe(df)


EXAMPLE_CASES_CSV = """case_id,prompt,ground_truth
loan_a,"A 34-year-old nurse with stable income asks for a $15,000 car loan. Estimate the probability (0-100) that she repays in full.",92
loan_b,"A 22-year-old freelance designer with irregular income asks for a $15,000 car loan. Estimate the probability (0-100) that he repays in full.",71
loan_c,"A 51-year-old shop owner whose business revenue fell 20% last year asks for a $15,000 car loan. Estimate the probability (0-100) that she repays in full.",78
"""
