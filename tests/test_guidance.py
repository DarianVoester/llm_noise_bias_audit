"""Recipes, setup check and metric glossary (pure Python, no Streamlit, no sandbox)."""

import re
from pathlib import Path

import pytest

from src.cases import EXAMPLE_CASES_CSV, cases_from_csv
from src.context_engine import VariationMode
from src.guidance import (
    CHART_TIPS,
    METRIC_GLOSSARY,
    MODE_PRESET_FIT,
    RECIPES,
    check_setup,
    explain_metric,
    setup_ok,
    suggest_preset,
)
from src.presets import PRESETS, PRESETS_BY_ID
from src.schemas import SchemaType

N_EXAMPLE_CASES = len(cases_from_csv(EXAMPLE_CASES_CSV))


def _check(**overrides):
    args = dict(schema=SchemaType.PERCENTAGE, mode=VariationMode.OCCASION,
                preset_id="mpd_rms_noise", n_windows=10, temperature=0.7, n_models=3,
                n_cases=1, has_truth=False)
    args.update(overrides)
    return check_setup(**args)


def _levels(tips):
    return {t.level for t in tips}


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: r.id)
def test_every_recipe_passes_its_own_setup_check(recipe):
    n_cases = N_EXAMPLE_CASES if recipe.use_case_file else 1
    tips = check_setup(schema=recipe.schema, mode=recipe.mode, preset_id=recipe.preset_id,
                       n_windows=recipe.n_windows, temperature=recipe.temperature,
                       n_models=max(recipe.min_models, 3), n_cases=n_cases,
                       has_truth=recipe.use_case_file)
    assert setup_ok(tips), [t.message for t in tips]
    assert "warning" not in _levels(tips), [t.message for t in tips]
    assert PRESETS_BY_ID[recipe.preset_id].compatibility(recipe.schema) == "recommended"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: r.id)
def test_recipe_session_values_use_known_widget_keys(recipe):
    values = recipe.session_values()
    assert values["schema_type"] == recipe.schema
    assert values["var_mode"] == recipe.mode
    assert isinstance(values["temperature"], float)
    assert isinstance(values["n_windows"], int)
    assert "base_prompt" not in recipe.session_values(load_example=False)


def test_good_default_setup_is_ok():
    tips = _check()
    assert _levels(tips) == {"ok"}


def test_incompatible_format_is_an_error():
    tips = _check(schema=SchemaType.RANKING, preset_id="mpd_rms_noise")
    assert not setup_ok(tips)


def test_missing_ground_truth_is_an_error():
    assert not setup_ok(_check(preset_id="mse_decomposition", has_truth=False))
    assert setup_ok(_check(preset_id="mse_decomposition", has_truth=True))


def test_noise_audit_needs_cases_and_models():
    assert not setup_ok(_check(preset_id="noise_audit", n_cases=1))
    assert not setup_ok(_check(preset_id="noise_audit", n_cases=3, n_models=1))
    assert setup_ok(_check(preset_id="noise_audit", n_cases=3, n_models=1,
                           mode=VariationMode.PERSONA))


def test_group_analysis_needs_group_modifier_and_vice_versa():
    assert not setup_ok(_check(preset_id="anchoring_framing_effect"))
    tips = _check(mode=VariationMode.ANCHOR, preset_id="mpd_rms_noise", n_windows=20)
    assert "warning" in _levels(tips)


def test_small_n_and_zero_temperature_warn():
    assert "warning" in _levels(_check(n_windows=4))
    assert "warning" in _levels(_check(mode=VariationMode.ANCHOR,
                                       preset_id="anchoring_framing_effect", n_windows=10))
    assert "warning" in _levels(_check(temperature=0.0))


def test_no_models_is_an_error():
    assert not setup_ok(_check(n_models=0))


@pytest.mark.parametrize("schema", list(SchemaType))
@pytest.mark.parametrize("mode", list(VariationMode))
def test_suggest_preset_is_always_compatible(schema, mode):
    pid = suggest_preset(schema, mode, n_cases=1, has_truth=False, n_models=1)
    assert PRESETS_BY_ID[pid].compatibility(schema) == "recommended"


def test_suggest_preset_prefers_effect_size_for_anchoring():
    assert suggest_preset(SchemaType.PERCENTAGE, VariationMode.ANCHOR, 1, False, 2) == \
        "anchoring_framing_effect"


def test_mode_fit_and_chart_tips_reference_real_presets():
    for pids in MODE_PRESET_FIT.values():
        assert set(pids) <= set(PRESETS_BY_ID)
    assert set(CHART_TIPS) == set(PRESETS_BY_ID)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_every_metric_and_column_in_preset_code_is_explained(preset):
    """Keys of summary_metrics and of per-model table rows must have a glossary entry."""
    keys = set(re.findall(r'"([a-z][a-z0-9_]*)":', preset.code))
    ignore = {"summary_metrics", "per_model_table", "tables", "figures", "warnings",
              "interpretation", "error", "component", "variance", "sd",
              "share_of_system_noise", "per_case", "decomposition", "x", "y", "case_id", "mode",
              "label", "judge", "boolean", "enum"}  # plot labels / schema names, not columns
    missing = sorted(k for k in keys - ignore if explain_metric(k) is None)
    assert not missing, f"{preset.id}: add to METRIC_GLOSSARY: {missing}"


def test_every_analysis_has_an_everyday_question():
    from src.guidance import PRESET_QUESTIONS

    assert set(PRESET_QUESTIONS) == set(PRESETS_BY_ID)
    assert all(q.endswith("?") for q in PRESET_QUESTIONS.values())


def test_glossary_entries_are_sentences():
    for key, text in METRIC_GLOSSARY.items():
        assert len(text) > 8 and text[0].isupper(), key


def test_user_guide_exists_and_covers_key_sections():
    guide = (Path(__file__).resolve().parent.parent / "docs" / "USER_GUIDE.md").read_text(
        encoding="utf-8")
    for heading in ("Your first run", "Which settings work well together",
                    "How to read the results", "Troubleshooting", "Glossary"):
        assert heading in guide
    for analysis in ("Mean pairwise distance", "MSE noise decomposition", "Noise audit",
                     "Categorical stability", "Anchoring & framing effect size",
                     "Ranking stability"):
        assert analysis in guide, analysis
    for recipe in RECIPES:
        assert recipe.question in guide, recipe.question
