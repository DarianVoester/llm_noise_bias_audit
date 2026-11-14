"""Tests for case loading and context window generation."""

from collections import Counter

import pytest

from src.cases import EXAMPLE_CASES_CSV, CaseSetError, cases_from_csv, single_case
from src.context_engine import (
    VariationConfig,
    VariationMode,
    balanced_orders,
    build_windows,
)
from src.schemas import SchemaConfig, SchemaType

PCT = SchemaConfig(type=SchemaType.PERCENTAGE)
CASES = single_case("Estimate the probability that the project finishes on time.")


def test_example_csv_loads():
    cases = cases_from_csv(EXAMPLE_CASES_CSV)
    assert [c.case_id for c in cases] == ["loan_a", "loan_b", "loan_c"]
    assert cases[0].ground_truth == 92.0


@pytest.mark.parametrize(
    "csv",
    ["case_id,text\n1,hello\n", "case_id,prompt\na,\n", "case_id,prompt\na,x\na,y\n",
     "case_id,prompt,ground_truth\na,x,abc\n"],
)
def test_bad_csv_rejected(csv):
    with pytest.raises(CaseSetError):
        cases_from_csv(csv)


def test_occasion_identical_prompts():
    w = build_windows(CASES, VariationConfig(), PCT, n=5, design_seed=1)
    assert len(w) == 5 and len({x.user_prompt for x in w}) == 1
    assert "OUTPUT FORMAT" in w[0].user_prompt
    assert [x.window_id for x in w] == [1, 2, 3, 4, 5]


def test_anchor_balanced_random_and_deterministic():
    cfg = VariationConfig(mode=VariationMode.ANCHOR,
                          anchor_text="A colleague guessed 95% yesterday.")
    w1 = build_windows(CASES, cfg, PCT, n=10, design_seed=7)
    w2 = build_windows(CASES, cfg, PCT, n=10, design_seed=7)
    assert Counter(x.variant for x in w1) == {"anchored": 5, "baseline": 5}
    assert [x.variant for x in w1] == [x.variant for x in w2]
    # random assignment, not simply "first half / second half"
    variants = [x.variant for x in w1]
    assert variants != ["baseline"] * 5 + ["anchored"] * 5 or \
        [x.variant for x in build_windows(CASES, cfg, PCT, 10, 8)] != variants
    anchored = [x for x in w1 if x.variant == "anchored"]
    assert all("95%" in x.user_prompt for x in anchored)
    assert anchored[0].meta["anchor_value"] == 95.0


def test_framing_two_texts():
    cfg = VariationConfig(mode=VariationMode.FRAMING, frame_a_text="90% survive.",
                          frame_b_text="10% die.")
    w = build_windows(CASES, cfg, PCT, n=6, design_seed=1)
    assert Counter(x.variant for x in w) == {"frame_A": 3, "frame_B": 3}
    assert all(("90% survive" in x.user_prompt) == (x.variant == "frame_A") for x in w)


def test_order_permutation_balanced_positions():
    orders = balanced_orders(4, 16, __import__("random").Random(0))
    first_positions = Counter(o[0] for o in orders)
    assert set(first_positions.values()) == {4}  # each chunk is first equally often


def test_order_all_permutations_when_small():
    cfg = VariationConfig(mode=VariationMode.ORDER, chunks_text="Chunk A\n---\nChunk B\n---\nC")
    w = build_windows(CASES, cfg, PCT, n=12, design_seed=3)
    assert Counter(x.variant for x in w) == {v: 2 for v in
                                            {"order_ABC", "order_ACB", "order_BAC",
                                             "order_BCA", "order_CAB", "order_CBA"}}


def test_persona_rotation():
    cfg = VariationConfig(mode=VariationMode.PERSONA, personas=["P1", "P2", "P3"])
    w = build_windows(CASES, cfg, PCT, n=9, design_seed=1)
    assert Counter(x.persona for x in w) == {"P1": 3, "P2": 3, "P3": 3}
    assert all(x.system_prompt == x.persona for x in w)


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": VariationMode.ORDER, "chunks_text": "only one"},
     {"mode": VariationMode.ANCHOR},
     {"mode": VariationMode.FRAMING, "frame_a_text": "a"},
     {"mode": VariationMode.PERSONA, "personas": ["one"]}],
)
def test_invalid_variation_configs(kwargs):
    with pytest.raises(ValueError):
        VariationConfig(**kwargs)


def test_multiple_cases():
    cases = cases_from_csv(EXAMPLE_CASES_CSV)
    w = build_windows(cases, VariationConfig(), PCT, n=4, design_seed=1)
    assert len(w) == 12 and {x.case_id for x in w} == {"loan_a", "loan_b", "loan_c"}
