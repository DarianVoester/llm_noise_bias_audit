"""All 6 statistical presets: metadata, validator-clean code, correct maths via the sandbox."""

import numpy as np
import pandas as pd
import pytest

from src.cases import EXAMPLE_CASES_CSV, cases_from_csv, single_case
from src.context_engine import VariationConfig, VariationMode, build_windows
from src.evaluator import run_sandboxed, validate_code
from src.llm_runner import RunParams, mock_model_specs, run_audit_sync
from src.presets import PRESETS, PRESETS_BY_ID, compatibility_matrix, presets_for_schema
from src.schemas import SchemaConfig, SchemaType

MULTI = cases_from_csv(EXAMPLE_CASES_CSV)


def mock_df(schema, cases=MULTI, variation=None, n=20):
    w = build_windows(cases, variation or VariationConfig(), schema, n=n, design_seed=1)
    return run_audit_sync(w, mock_model_specs(), schema, RunParams())


def hand_df(values_by_model, case_id="c1", truth=None, variant="identical", schema="percentage"):
    rows = []
    for model, values in values_by_model.items():
        for i, v in enumerate(values, start=1):
            rows.append({"model": model, "case_id": case_id, "context_window_id": i,
                         "context_variant": variant, "persona": None, "parse_success": True,
                         "parsed_value_num": float(v), "parsed_value_json": None,
                         "ground_truth": truth, "schema_type": schema,
                         "scale_min": 0.0, "scale_max": 100.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_metadata_complete_and_code_valid(preset):
    for field in ("plain_english_summary", "kahneman_concept", "how_to_interpret"):
        assert len(getattr(preset, field)) > 40
    assert preset.compatible_schemas
    assert validate_code(preset.code) == [], preset.id


def test_six_presets_and_matrix():
    assert len(PRESETS) == 6
    assert len(compatibility_matrix()) == 7
    ranking = presets_for_schema(SchemaType.RANKING)
    assert ranking[0][0].id == "ranking_stability" and ranking[0][1] == "recommended"


def test_mse_identity_exact():
    # values 50, 60, 70 vs truth 50: bias = 10, population variance = 200/3
    df = hand_df({"m": [50, 60, 70]}, truth=50.0)
    r = run_sandboxed(PRESETS_BY_ID["mse_decomposition"].code, df)
    assert r.ok, r.error
    row = r.tables["per_model_table"].iloc[0]
    assert row["bias_sq"] == pytest.approx(100.0, abs=1e-3)
    assert row["noise_sq"] == pytest.approx(200 / 3, abs=1e-3)
    assert row["mse"] == pytest.approx(np.mean((np.array([50, 60, 70]) - 50) ** 2), abs=1e-3)


def test_mse_uses_ui_ground_truth_and_reports_missing():
    df = hand_df({"m": [40, 60]})
    assert "error" in run_sandboxed(PRESETS_BY_ID["mse_decomposition"].code, df).summary_metrics
    r = run_sandboxed(PRESETS_BY_ID["mse_decomposition"].code, df, ground_truth=50.0)
    assert r.tables["per_model_table"].iloc[0]["bias_sq"] == pytest.approx(0.0)


def test_mpd_exact():
    df = hand_df({"stable": [50, 50, 50], "noisy": [0, 50, 100]})
    r = run_sandboxed(PRESETS_BY_ID["mpd_rms_noise"].code, df)
    assert r.ok, r.error
    t = r.tables["per_model_table"].set_index("model")
    assert t.loc["stable", "mean_pairwise_distance"] == 0
    assert t.loc["noisy", "mean_pairwise_distance"] == pytest.approx(200 / 3, abs=1e-3)
    assert r.summary_metrics["most_stable_model"] == "stable"


def test_noise_audit_recovers_level_noise():
    parts = []
    for model, offset in (("strict", -10), ("lenient", 10)):
        for case, base in (("c1", 30), ("c2", 60)):
            parts.append(hand_df({model: [base + offset] * 4}, case_id=case))
    r = run_sandboxed(PRESETS_BY_ID["noise_audit"].code, pd.concat(parts))
    assert r.ok, r.error
    m = r.summary_metrics
    assert m["level_noise_sd"] == pytest.approx(10.0, abs=1e-6)
    assert m["occasion_noise_sd"] == pytest.approx(0.0, abs=1e-6)
    assert m["stable_pattern_noise_sd"] == pytest.approx(0.0, abs=1e-6)


def test_noise_audit_needs_cases():
    r = run_sandboxed(PRESETS_BY_ID["noise_audit"].code, hand_df({"a": [1, 2], "b": [3, 4]}))
    assert r.ok and "error" in r.summary_metrics


def test_categorical_flip_rate():
    df = hand_df({"solid": [1, 1, 1, 1], "coin": [1, 0, 1, 0]}, schema="boolean")
    r = run_sandboxed(PRESETS_BY_ID["categorical_stability"].code, df)
    assert r.ok, r.error
    t = r.tables["per_model_table"].set_index("model")
    assert t.loc["solid", "flip_rate"] == 0 and t.loc["coin", "flip_rate"] == 0.5
    assert t.loc["coin", "entropy_normalized"] == pytest.approx(1.0)


def test_effect_detects_strong_anchor():
    schema = SchemaConfig(type=SchemaType.PERCENTAGE)
    var = VariationConfig(mode=VariationMode.ANCHOR, anchor_text="A colleague guessed 5%.")
    df = mock_df(schema, cases=single_case("Estimate the repayment probability.", 70.0),
                 variation=var, n=40)
    r = run_sandboxed(PRESETS_BY_ID["anchoring_framing_effect"].code, df)
    assert r.ok, r.error
    t = r.tables["per_model_table"].set_index("model")
    assert t.loc["mock/anchored-optimist", "shift"] < -15
    assert bool(t.loc["mock/anchored-optimist", "significant_5pct"])
    assert r.summary_metrics["group_a"] == "baseline"


@pytest.mark.parametrize("stype", [SchemaType.BOOLEAN, SchemaType.ENUM])
def test_effect_categorical_variants_run(stype):
    schema = SchemaConfig(type=stype, options=["A", "B", "C"] if stype == SchemaType.ENUM else [])
    var = VariationConfig(mode=VariationMode.FRAMING, frame_a_text="Gain.", frame_b_text="Loss.")
    r = run_sandboxed(PRESETS_BY_ID["anchoring_framing_effect"].code,
                      mock_df(schema, single_case("x", 0.5), var, n=30))
    assert r.ok, r.error
    assert {"Fisher exact", "chi-square"} & set(r.tables["per_model_table"]["test"])


def test_ranking_stability_exact():
    rows = []
    for i, rank in enumerate([["A", "B", "C"], ["A", "B", "C"], ["C", "B", "A"]], start=1):
        rows.append({"model": "m", "case_id": "c", "context_window_id": i,
                     "context_variant": "identical", "parse_success": True,
                     "parsed_value_num": float("nan"),
                     "parsed_value_json": '["' + '", "'.join(rank) + '"]',
                     "schema_type": "ranking"})
    r = run_sandboxed(PRESETS_BY_ID["ranking_stability"].code, pd.DataFrame(rows))
    assert r.ok, r.error
    row = r.tables["per_model_table"].iloc[0]
    # pair distances: (1,2)=0, (1,3)=1, (2,3)=1 -> mean 2/3
    assert row["mean_kendall_tau_distance"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["identical_to_modal_ranking"] == pytest.approx(2 / 3, abs=1e-3)


@pytest.mark.parametrize(
    "preset_id,stype",
    [("mse_decomposition", SchemaType.LIKERT), ("mpd_rms_noise", SchemaType.FLOAT),
     ("noise_audit", SchemaType.CONFIDENCE), ("categorical_stability", SchemaType.ENUM),
     ("ranking_stability", SchemaType.RANKING)],
)
def test_presets_run_on_mock_data(preset_id, stype):
    opts = ["A", "B", "C"] if stype in (SchemaType.ENUM, SchemaType.RANKING) else []
    schema = SchemaConfig(type=stype, options=opts, likert_max=7)
    r = run_sandboxed(PRESETS_BY_ID[preset_id].code, mock_df(schema), ground_truth=4.0)
    assert r.ok, r.error
    assert r.summary_metrics and r.figures

