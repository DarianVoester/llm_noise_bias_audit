"""Generate N isolated context windows per case for five noise/bias trigger modes.

Windows are model-independent: every selected model receives exactly the same set
of windows, which turns model comparisons into a paired design.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from src.cases import Case
from src.config import DEFAULT_PERSONAS
from src.schemas import SchemaConfig, build_format_instructions

DEFAULT_SYSTEM_PROMPT = "You are a careful assistant. Follow the output format exactly."


class VariationMode(StrEnum):
    OCCASION = "occasion"
    ORDER = "order"
    ANCHOR = "anchor"
    FRAMING = "framing"
    PERSONA = "persona"


MODE_LABELS: dict[VariationMode, str] = {
    VariationMode.OCCASION: "Nothing: the identical question every time",
    VariationMode.ORDER: "The order of the facts",
    VariationMode.ANCHOR: "An irrelevant number is added (anchoring)",
    VariationMode.FRAMING: "The wording (framing)",
    VariationMode.PERSONA: "The role the model plays (persona)",
}

MODE_EXPLANATIONS: dict[VariationMode, str] = {
    VariationMode.OCCASION: (
        "Every repetition gets exactly the same question. Any difference between the answers "
        "is pure chance, like a judge who decides differently on Monday than on Friday "
        "('occasion noise'). Keep the randomness slider above 0, otherwise most models just "
        "repeat themselves."
    ),
    VariationMode.ORDER: (
        "You enter a few facts. Each repetition shows them in a different order (every fact "
        "appears in every position equally often). If the answers change with the order, the "
        "model gives too much weight to what it reads first or last."
    ),
    VariationMode.ANCHOR: (
        "A randomly chosen half of the repetitions gets one extra sentence with a number "
        "that has nothing to do with the question. If those answers drift towards the number, "
        "the model is 'anchored' by it."
    ),
    VariationMode.FRAMING: (
        "Half of the repetitions read wording A, the other half wording B - the same fact "
        "said in two ways (e.g. '90 % survive' vs. '10 % die'). Since the fact is the same, "
        "the answers should be the same too."
    ),
    VariationMode.PERSONA: (
        "The model is told to play a different role in each repetition (e.g. a cautious "
        "analyst, then an optimist). This shows how much the answer depends on the role - "
        "similar to judges who are generally stricter or milder ('level noise')."
    ),
}

CHUNK_SPLIT_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


class VariationConfig(BaseModel):
    mode: VariationMode = VariationMode.OCCASION
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    chunks_text: str = ""
    anchor_text: str = ""
    frame_a_text: str = ""
    frame_b_text: str = ""
    personas: list[str] = Field(default_factory=lambda: list(DEFAULT_PERSONAS))

    @model_validator(mode="after")
    def _check(self) -> VariationConfig:
        if self.mode == VariationMode.ORDER and len(self.chunks) < 2:
            raise ValueError("Order permutation needs at least two chunks separated by '---'.")
        if self.mode == VariationMode.ANCHOR and not self.anchor_text.strip():
            raise ValueError("Anchoring needs an anchor / distractor sentence.")
        if self.mode == VariationMode.FRAMING and not (
            self.frame_a_text.strip() and self.frame_b_text.strip()
        ):
            raise ValueError("Framing needs both a frame A and a frame B text.")
        if self.mode == VariationMode.PERSONA:
            self.personas = [p.strip() for p in self.personas if p.strip()]
            if len(self.personas) < 2:
                raise ValueError("Persona jitter needs at least two personas.")
        return self

    @property
    def chunks(self) -> list[str]:
        return [c.strip() for c in CHUNK_SPLIT_RE.split(self.chunks_text) if c.strip()]

    @property
    def anchor_value(self) -> float | None:
        """First number in the anchor sentence (used by the mock simulator and presets)."""
        m = _NUMBER_RE.search(self.anchor_text)
        return float(m.group(0).replace(",", ".")) if m else None


@dataclass
class ContextWindow:
    case_id: str
    window_id: int  # 1..N within the case
    variant: str
    system_prompt: str
    user_prompt: str
    persona: str | None = None
    ground_truth: float | None = None
    meta: dict[str, object] = field(default_factory=dict)


# --- Design helpers ----------------------------------------------------------

def balanced_orders(k: int, n: int, rng: random.Random) -> list[tuple[int, ...]]:
    """Return n chunk orders over k chunks with balanced positions.

    If k! <= n all permutations are cycled evenly. Otherwise a cyclic Latin square
    (every chunk appears in every position equally often) plus its reversed rows is
    used, so primacy and recency effects are balanced across windows.
    """
    if k < 1 or n < 1:
        return []
    if math.factorial(k) <= n:
        perms = list(itertools.permutations(range(k)))
        rng.shuffle(perms)
        return [perms[i % len(perms)] for i in range(n)]
    base = [tuple((i + j) % k for j in range(k)) for i in range(k)]
    rows = base + [tuple(reversed(r)) for r in base]
    rng.shuffle(rows)
    return [rows[i % len(rows)] for i in range(n)]


def balanced_groups(n: int, rng: random.Random) -> list[bool]:
    """Randomly assign floor(n/2) windows to group B; returns True for group B."""
    flags = [i < n // 2 for i in range(n)]
    rng.shuffle(flags)
    return flags


def _compose(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def build_windows(
    cases: list[Case],
    variation: VariationConfig,
    schema: SchemaConfig,
    n: int,
    design_seed: int,
) -> list[ContextWindow]:
    """Create N context windows per case. Deterministic for a given ``design_seed``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    fmt = build_format_instructions(schema)
    windows: list[ContextWindow] = []
    for case_idx, case in enumerate(cases):
        rng = random.Random(f"{design_seed}:{case_idx}:{case.case_id}")
        mode = variation.mode
        system = variation.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT

        if mode == VariationMode.ORDER:
            chunks = variation.chunks
            orders = balanced_orders(len(chunks), n, rng)
            for i, order in enumerate(orders, start=1):
                ctx = "\n\n".join(chunks[j] for j in order)
                windows.append(ContextWindow(
                    case.case_id, i, "order_" + "".join(chr(65 + j) for j in order), system,
                    _compose("Context:\n" + ctx, case.prompt) + fmt,
                    ground_truth=case.ground_truth,
                    meta={"order": list(order), "first_chunk": order[0]},
                ))
        elif mode in (VariationMode.ANCHOR, VariationMode.FRAMING):
            groups = balanced_groups(n, rng)
            for i, is_b in enumerate(groups, start=1):
                if mode == VariationMode.ANCHOR:
                    variant = "anchored" if is_b else "baseline"
                    user = _compose(variation.anchor_text, case.prompt) if is_b else case.prompt
                    meta: dict[str, object] = {"anchor_value": variation.anchor_value} \
                        if is_b else {}
                else:
                    variant = "frame_B" if is_b else "frame_A"
                    frame = variation.frame_b_text if is_b else variation.frame_a_text
                    user = _compose(frame, case.prompt)
                    meta = {}
                windows.append(ContextWindow(case.case_id, i, variant, system, user + fmt,
                                             ground_truth=case.ground_truth, meta=meta))
        elif mode == VariationMode.PERSONA:
            personas = list(variation.personas)
            offset = rng.randrange(len(personas))
            for i in range(1, n + 1):
                p_idx = (i - 1 + offset) % len(personas)
                windows.append(ContextWindow(
                    case.case_id, i, f"persona_{p_idx + 1}", personas[p_idx],
                    case.prompt + fmt, persona=personas[p_idx],
                    ground_truth=case.ground_truth,
                ))
        else:  # OCCASION
            for i in range(1, n + 1):
                windows.append(ContextWindow(case.case_id, i, "identical", system,
                                             case.prompt + fmt, ground_truth=case.ground_truth))
    return windows

