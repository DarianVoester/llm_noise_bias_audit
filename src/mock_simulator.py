"""Deterministic mock LLM that produces realistic noisy / biased answers.

The simulated judgment is built from interpretable components (all expressed in
units of the scale range so the same profile works for every schema):

    value = truth + level_bias + pattern(model, case) + persona(model, persona)
            + anchor_pull + framing_shift + order_effect + occasion_noise

This lets users and tests verify that the statistical presets recover known effects.
"""

from __future__ import annotations

import hashlib
import json
import random

from src.config import MOCK_MODELS, MockModelProfile
from src.context_engine import ContextWindow
from src.schemas import SchemaConfig, SchemaType

MOCK_PROFILES: dict[str, MockModelProfile] = {p.name: p for p in MOCK_MODELS}


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _latent_truth(case_id: str, ground_truth: float | None, lo: float, hi: float) -> float:
    """Ground truth if it lies on the answer scale, otherwise a stable pseudo-truth per case."""
    if ground_truth is not None and lo <= ground_truth <= hi:
        return ground_truth
    return lo + (hi - lo) * (0.3 + 0.4 * _rng("truth", case_id).random())


def simulate_response(
    profile: MockModelProfile,
    window: ContextWindow,
    schema: SchemaConfig,
    seed: int,
    temperature: float,
    repair: bool = False,
) -> str:
    """Return a raw text response for one context window (deterministic per seed).

    ``repair=True`` simulates the answer after a format reminder: the same judgement,
    but the model now complies with the requested format.
    """
    occ = _rng("occ", profile.name, window.case_id, window.window_id, window.variant, seed)
    pat = _rng("pattern", profile.name, window.case_id)
    temp_scale = max(temperature, 0.05) / 0.7

    if occ.random() < profile.parse_failure_rate and not repair:
        return "I'm not entirely sure how to answer this in the requested format."

    lo, hi = schema.scale_range or (0.0, 100.0)
    if schema.type == SchemaType.FLOAT:
        lo, hi = 0.0, max(abs(window.ground_truth or 0.0) * 2.0, 100.0)
    span = hi - lo
    truth = _latent_truth(window.case_id, window.ground_truth, lo, hi)

    z = truth + span * profile.level_bias
    z += span * pat.gauss(0.0, profile.pattern_sd)
    if window.persona:
        z += span * _rng("persona", profile.name, window.persona).gauss(0.0, profile.persona_sd)
    anchor = window.meta.get("anchor_value")
    if isinstance(anchor, (int, float)):
        if lo <= float(anchor) <= hi:
            z += profile.anchor_sensitivity * (float(anchor) - z)
        else:  # anchor outside the scale: generic upward pull
            z += span * 0.25 * profile.anchor_sensitivity
    if window.variant == "frame_B":
        z += span * profile.framing_shift
    first = window.meta.get("first_chunk")
    if isinstance(first, int):
        z += span * profile.order_sensitivity * (1 if first == 0 else -0.5)
    z += span * occ.gauss(0.0, profile.occasion_sd * temp_scale)
    z = min(max(z, lo), hi)
    frac = (z - lo) / span if span else 0.5

    t = schema.type
    if t == SchemaType.BOOLEAN:
        # truth acts as a probability of "true" (0..1)
        p_true = min(max(z, 0.0), 1.0)
        return json.dumps({"answer": occ.random() < p_true})
    if t == SchemaType.PERCENTAGE:
        return json.dumps({"answer": round(z, 1)})
    if t == SchemaType.FLOAT:
        return f"Here is my estimate.\n{json.dumps({'answer': round(z, 2)})}"
    if t == SchemaType.LIKERT:
        return json.dumps({"answer": int(round(z))})
    if t == SchemaType.ENUM:
        k = len(schema.options)
        idx = min(int(frac * k), k - 1)
        return json.dumps({"answer": schema.options[idx]})
    if t == SchemaType.RANKING:
        base = list(schema.options)
        scores = {o: i + pat.gauss(0, 0.4) + occ.gauss(0, 0.8 * temp_scale *
                                                       (1 + 5 * profile.occasion_sd))
                  for i, o in enumerate(base)}
        if isinstance(first, int):
            scores[base[min(first, len(base) - 1)]] -= profile.order_sensitivity * 10
        return json.dumps({"answer": sorted(base, key=lambda o: scores[o])})
    # CONFIDENCE: decision on the 0..100 scale, confidence shrinks with noise
    conf = 100 * (1 - min(0.9, 3 * profile.occasion_sd)) + occ.gauss(0, 5)
    return json.dumps({"decision": round(z, 1), "confidence": round(min(max(conf, 0), 100), 1)})
