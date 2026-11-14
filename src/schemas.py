"""Output schemas: prompt formatting instructions and robust response parsers.

Parse order (recorded in ``ParseResult.method``):
1. ``native`` - the provider returned pure JSON because JSON mode was requested,
2. ``json``   - a JSON object was extracted from free text,
3. ``regex``  - an explicit ``ANSWER:`` line (or a very short bare answer),
4. ``failed``.

The regex fallback never grabs an arbitrary number from the text. This matters for
anchoring experiments: the anchor value itself must not be mistaken for the answer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SchemaType(StrEnum):
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    FLOAT = "float"
    LIKERT = "likert"
    ENUM = "enum"
    RANKING = "ranking"
    CONFIDENCE = "confidence"


SCHEMA_LABELS: dict[SchemaType, str] = {
    SchemaType.BOOLEAN: "Yes or no",
    SchemaType.PERCENTAGE: "Percentage (0 to 100 %)",
    SchemaType.FLOAT: "Any number",
    SchemaType.LIKERT: "Rating on a scale (e.g. 1 to 5)",
    SchemaType.ENUM: "One of several categories",
    SchemaType.RANKING: "Ranking (items put in order)",
    SchemaType.CONFIDENCE: "Decision plus confidence",
}

NUMERIC_SCHEMAS = {
    SchemaType.BOOLEAN,
    SchemaType.PERCENTAGE,
    SchemaType.FLOAT,
    SchemaType.LIKERT,
    SchemaType.CONFIDENCE,
}


class SchemaConfig(BaseModel):
    """User-selected output format."""

    type: SchemaType
    likert_min: int = 1
    likert_max: int = 5
    options: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> SchemaConfig:
        if self.type == SchemaType.LIKERT and not (self.likert_min < self.likert_max):
            raise ValueError("likert_min must be smaller than likert_max")
        if self.type in (SchemaType.ENUM, SchemaType.RANKING):
            cleaned = [o.strip() for o in self.options if o.strip()]
            if len(cleaned) < 2:
                raise ValueError("Provide at least two options")
            if len({o.lower() for o in cleaned}) != len(cleaned):
                raise ValueError("Options must be unique (case-insensitive)")
            self.options = cleaned
        return self

    @property
    def scale_range(self) -> tuple[float, float] | None:
        """Theoretical value range, used to normalise noise metrics."""
        if self.type == SchemaType.BOOLEAN:
            return (0.0, 1.0)
        if self.type == SchemaType.PERCENTAGE:
            return (0.0, 100.0)
        if self.type == SchemaType.LIKERT:
            return (float(self.likert_min), float(self.likert_max))
        return None


@dataclass
class ParseResult:
    success: bool
    method: str  # native | json | regex | failed
    value_num: float | None = None
    value_json: str | None = None
    confidence: float | None = None
    error: str | None = None


# --- Prompt instructions ---------------------------------------------------

def build_format_instructions(schema: SchemaConfig) -> str:
    """Instructions appended to every prompt so the answer is machine-parsable."""
    t = schema.type
    if t == SchemaType.BOOLEAN:
        spec, example = "true or false", '{"answer": true}'
    elif t == SchemaType.PERCENTAGE:
        spec, example = ("a number on the 0-100 percent scale, without the % sign. 42.5 means "
                         "42.5 percent. NEVER use a 0-1 fraction such as 0.425"), \
            '{"answer": 42.5}'
    elif t == SchemaType.FLOAT:
        spec, example = "a single number (no units, no thousands separators)", \
            '{"answer": 1250.0}'
    elif t == SchemaType.LIKERT:
        spec = f"a whole number from {schema.likert_min} to {schema.likert_max}"
        example = f'{{"answer": {schema.likert_min}}}'
    elif t == SchemaType.ENUM:
        opts = ", ".join(f'"{o}"' for o in schema.options)
        spec, example = f"exactly one of: {opts}", f'{{"answer": "{schema.options[0]}"}}'
    elif t == SchemaType.RANKING:
        opts = ", ".join(f'"{o}"' for o in schema.options)
        spec = (f"a list containing every one of these items exactly once, ordered from "
                f"first/best to last/worst: {opts}")
        example = '{"answer": [' + ", ".join(f'"{o}"' for o in schema.options) + "]}"
    else:  # CONFIDENCE
        return (
            "\n\n---\nOUTPUT FORMAT (mandatory): Respond with ONLY a JSON object of the form "
            '{"decision": <number>, "confidence": <number 0-100>}. "decision" is your '
            'numeric answer, "confidence" is how sure you are on the 0-100 percent scale '
            '(80 means 80 percent; NEVER a 0-1 fraction such as 0.8). '
            'Example: {"decision": 7.5, "confidence": 80}. No other text.'
        )
    return (
        "\n\n---\nOUTPUT FORMAT (mandatory): Respond with ONLY a JSON object of the form "
        f'{{"answer": <value>}} where <value> is {spec}. Example: {example}. No other text.'
    )


def build_repair_message(schema: SchemaConfig, error: str | None) -> str:
    """Follow-up user turn sent once when a reply violates the required format."""
    reason = f" Problem: {error}." if error else ""
    return ("Your previous reply did not follow the required output format." + reason +
            " Answer the same question again, using exactly this format." +
            build_format_instructions(schema))


def response_format_payload(schema: SchemaConfig) -> dict[str, Any]:
    """Provider-native JSON mode request (OpenAI-compatible)."""
    return {"type": "json_object"}


# --- Parsing helpers --------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Written so that no two neighbouring parts can match the same characters: the former
# ``\s*\**\s*`` took cubic time on long runs of blank lines. The value runs to the end of
# its line; the caller strips trailing blanks.
_ANSWER_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:\*+[ \t]*)?answer[ \t]*(?:\*+[ \t]*)?[:=]\s*(\S[^\n]*)")
# Longer replies are cut before parsing (a real answer is far shorter).
MAX_PARSE_CHARS = 200_000
_MAX_JSON_CANDIDATES = 200
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
_TRUE_WORDS = {"true", "yes", "1", "y"}
_FALSE_WORDS = {"false", "no", "0", "n"}
_ANSWER_KEYS = ("answer", "value", "result")
EMPTY_RESPONSE_ERROR = "empty response"


def strip_reasoning(raw: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    return _THINK_RE.sub("", raw or "").strip()


def _extract_json_object(text: str) -> tuple[Any, bool] | None:
    """Return (obj, is_whole_text) for the first JSON value found in text."""
    stripped = text.strip()
    try:
        return json.loads(stripped), True
    except (json.JSONDecodeError, ValueError, RecursionError):
        pass
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    # Balanced-brace scan for embedded objects in one pass: every "{" is remembered and
    # paired with its closing "}". Candidates are ordered by where they start.
    spans: list[tuple[int, int]] = []
    open_at: list[int] = []
    for pos, ch in enumerate(text):
        if ch == "{":
            open_at.append(pos)
        elif ch == "}" and open_at:
            spans.append((open_at.pop(), pos + 1))
    candidates += [text[a:b] for a, b in sorted(spans)[:_MAX_JSON_CANDIDATES]]
    parsed: list[Any] = []
    for cand in candidates:
        try:
            parsed.append(json.loads(cand.strip()))
        except (json.JSONDecodeError, ValueError, RecursionError):
            continue
    if not parsed:
        return None
    # A model may "show its work" as JSON before the final answer, e.g.
    # 'Reasoning: {"base": 30} Final: {"answer": 42}'. The object that names the answer
    # explicitly wins over any other embedded object.
    return next((p for p in parsed if _has_answer_key(p)), parsed[0]), False


def _has_answer_key(obj: Any) -> bool:
    return isinstance(obj, dict) and any(
        str(k).lower() in (*_ANSWER_KEYS, "decision") for k in obj)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        s = value.strip().rstrip("%").strip().replace("\u2212", "-").lstrip("$€£")
        if not _NUMBER_RE.fullmatch(s):
            return None
        if s.count(",") == 1 and "." not in s and len(s.split(",")[1]) != 3:
            s = s.replace(",", ".")  # decimal comma, e.g. "12,5"
        else:
            s = s.replace(",", "")  # thousands separators
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _to_bool(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    if isinstance(value, str):
        v = value.strip().strip(".!\"'").lower()
        if v in _TRUE_WORDS:
            return 1.0
        if v in _FALSE_WORDS:
            return 0.0
    return None


def _match_option(value: Any, options: list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().strip(".!\"'").lower()
    for o in options:
        if o.lower() == v:
            return o
    return None



def looks_like_fraction(f: float) -> bool:
    """True for values such as 0.425 that are almost certainly a 0-1 fraction, not percent.

    Whole numbers (0, 1) are accepted as percent. Genuine answers between 0 % and 1 % with
    decimals cannot be told apart from fractions and are therefore rejected as ambiguous -
    the format instructions tell the model to use the 0-100 scale.
    """
    return 0.0 < f < 1.0


FRACTION_ERROR = ("looks like a 0-1 fraction instead of the required 0-100 percent scale "
                  "(e.g. 0.425 instead of 42.5)")


def _validate_value(value: Any, schema: SchemaConfig) -> tuple[ParseResult | None, str]:
    """Validate a candidate answer. Returns (result or None, error message)."""
    t = schema.type
    if t == SchemaType.BOOLEAN:
        b = _to_bool(value)
        if b is None:
            return None, f"not a boolean: {value!r}"
        return ParseResult(True, "", value_num=b), ""
    if t in (SchemaType.PERCENTAGE, SchemaType.FLOAT, SchemaType.LIKERT):
        f = _to_float(value)
        if f is None:
            return None, f"not a number: {value!r}"
        if t == SchemaType.PERCENTAGE and not 0.0 <= f <= 100.0:
            return None, f"percentage out of range: {f}"
        if t == SchemaType.PERCENTAGE and looks_like_fraction(f):
            return None, f"{FRACTION_ERROR}: {f}"
        if t == SchemaType.LIKERT and (
            not float(f).is_integer() or not schema.likert_min <= f <= schema.likert_max
        ):
            return None, (f"Likert value must be an integer in "
                          f"[{schema.likert_min}, {schema.likert_max}]: {f}")
        return ParseResult(True, "", value_num=f), ""
    if t == SchemaType.ENUM:
        opt = _match_option(value, schema.options)
        if opt is None:
            return None, f"not one of {schema.options}: {value!r}"
        return ParseResult(True, "", value_json=json.dumps(opt)), ""
    if t == SchemaType.RANKING:
        if isinstance(value, str):
            value = [p for p in re.split(r"\s*(?:,|>|;|\n)\s*", value) if p]
        if not isinstance(value, list):
            return None, "ranking must be a list"
        matched = [_match_option(v, schema.options) for v in value]
        if any(m is None for m in matched):
            return None, f"unknown items in ranking: {value!r}"
        if sorted(matched) != sorted(schema.options):  # type: ignore[type-var]
            return None, "ranking must contain every option exactly once"
        return ParseResult(True, "", value_json=json.dumps(matched)), ""
    # CONFIDENCE
    if not isinstance(value, dict):
        return None, "expected an object with 'decision' and 'confidence'"
    dec = _to_float(value.get("decision"))
    if dec is None:
        dec = _to_bool(value.get("decision"))
    conf = _to_float(value.get("confidence"))
    if dec is None or conf is None:
        return None, "missing/invalid 'decision' or 'confidence'"
    if not 0.0 <= conf <= 100.0:
        return None, f"confidence out of range: {conf}"
    if looks_like_fraction(conf):
        return None, f"confidence {FRACTION_ERROR}: {conf}"
    return ParseResult(True, "", value_num=dec, confidence=conf), ""


def parse_response(raw: str, schema: SchemaConfig, json_mode: bool = False) -> ParseResult:
    """Parse a raw model response into a normalised value."""
    text = strip_reasoning((raw or "")[:MAX_PARSE_CHARS])
    if not text:
        return ParseResult(False, "failed", error=EMPTY_RESPONSE_ERROR)

    last_error = "no parsable answer found"
    json_error: str | None = None
    found = _extract_json_object(text)
    if found is not None:
        obj, whole = found
        candidate: Any = None
        if schema.type == SchemaType.CONFIDENCE:
            candidate = obj
        elif isinstance(obj, dict):
            key = next((k for k in obj if str(k).lower() in _ANSWER_KEYS),
                       None)
            if key is not None:
                candidate = obj[key]
            elif len(obj) == 1:
                candidate = next(iter(obj.values()))
        else:
            candidate = obj
        if candidate is not None:
            res, err = _validate_value(candidate, schema)
            if res:
                res.method = "native" if (json_mode and whole) else "json"
                return res
            last_error = json_error = err

    # Regex fallback: explicit "ANSWER:" line, or a very short bare reply - but never the
    # raw JSON text itself (that would hide the precise JSON validation error).
    if schema.type != SchemaType.CONFIDENCE:
        m = _ANSWER_LINE_RE.search(text)
        bare = text if found is None and len(text) <= 40 and "\n" not in text else None
        candidate_str = m.group(1) if m else bare
        if candidate_str is not None:
            res, err = _validate_value(candidate_str.strip().strip("`*"), schema)
            if res:
                res.method = "regex"
                return res
            last_error = json_error or err
    return ParseResult(False, "failed", error=last_error)


def decode_value(value_num: float | None, value_json: str | None) -> Any:
    """Recover the Python value stored in a df_results row."""
    if isinstance(value_json, str) and value_json:
        return json.loads(value_json)
    return value_num

