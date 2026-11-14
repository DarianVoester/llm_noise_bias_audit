"""User guidance: ready-made recipes and a setup check for setting combinations.

Pure Python (no Streamlit) so it can be unit-tested. The UI turns recipes into
session-state values and shows the tips returned by :func:`check_setup`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.context_engine import VariationMode
from src.presets import PRESETS_BY_ID
from src.schemas import SchemaType

CASE_SOURCE_SINGLE = "One question"
CASE_SOURCE_FILE = "Several questions (CSV file)"

GROUP_MODES = (VariationMode.ANCHOR, VariationMode.FRAMING)


@dataclass(frozen=True)
class Recipe:
    """A research question mapped to settings that work well together."""

    id: str
    question: str
    why: str
    schema: SchemaType
    mode: VariationMode
    preset_id: str
    n_windows: int
    temperature: float
    use_case_file: bool = False
    min_models: int = 1
    example_prompt: str = ""
    extra_state: dict[str, str] = field(default_factory=dict)

    def session_values(self, load_example: bool = True) -> dict[str, object]:
        """Session-state values that configure the app for this recipe."""
        values: dict[str, object] = {
            "schema_type": self.schema,
            "var_mode": self.mode,
            "preset_id": self.preset_id,
            "n_windows": self.n_windows,
            "temperature": self.temperature,
            "case_source": CASE_SOURCE_FILE if self.use_case_file else CASE_SOURCE_SINGLE,
            "use_truth": False,
        }
        if load_example:
            if self.example_prompt:
                values["base_prompt"] = self.example_prompt
            values.update(self.extra_state)
        return values


_RANK_PROMPT = ("A small online shop can improve only one area per quarter. Rank these "
                "priorities from most to least important for growing revenue next year: "
                "Price, Quality, Delivery speed, Support.")
_RANK_CHUNKS = ("Fact 1: Customers often complain that deliveries take more than a week.\n---\n"
                "Fact 2: Prices are 10 % above the two main competitors.\n---\n"
                "Fact 3: Product return rates are low and reviews praise the quality.\n---\n"
                "Fact 4: The support inbox has a backlog of 300 unanswered emails.")
_BOOL_PROMPT = ("A mid-sized company plans to launch a new product in 6 months. The team has 8 "
                "people, the budget is fixed, and two similar past projects were late. Will the "
                "launch happen on time? Answer true or false.")
_PCT_PROMPT = ("A mid-sized company plans to launch a new product in 6 months. The team has 8 "
               "people, the budget is fixed, and two similar past projects were late. Estimate "
               "the probability (0-100) that the launch happens on time.")

RECIPES: list[Recipe] = [
    Recipe("self_consistency", "Does a model give the same answer every time?",
           "The simplest check and the best one to start with. The same question is asked "
           "in separate, fresh conversations, so every difference between the answers is "
           "pure chance. You do not need to know the correct answer.",
           SchemaType.PERCENTAGE, VariationMode.OCCASION, "mpd_rms_noise", 10, 0.7,
           example_prompt=_PCT_PROMPT),
    Recipe("accuracy", "How far off is a model, and is that bias or noise?",
           "Compares the answers with the correct values and splits the error in two: "
           "always off in the same direction (bias) or scattered at random (noise). The "
           "built-in example questions come with correct values.",
           SchemaType.PERCENTAGE, VariationMode.OCCASION, "mse_decomposition", 10, 0.7,
           use_case_file=True),
    Recipe("noise_audit", "Why do different models disagree?",
           "Several questions go to several models. The result separates three causes: one "
           "model is always stricter, models react differently to particular questions, or "
           "the answers simply wobble at random. Needs at least 2 models.",
           SchemaType.PERCENTAGE, VariationMode.OCCASION, "noise_audit", 10, 0.7,
           use_case_file=True, min_models=2),
    Recipe("anchoring", "Does an irrelevant number sway the answer?",
           "Half of the repetitions mention a number that has nothing to do with the "
           "question. If those answers move towards the number, the model lets itself be "
           "anchored. 20 repetitions give 10 per group.",
           SchemaType.PERCENTAGE, VariationMode.ANCHOR, "anchoring_framing_effect", 20, 0.7,
           example_prompt=_PCT_PROMPT),
    Recipe("framing", "Does the wording change the answer?",
           "Half of the repetitions read '70 of 100 were on time', the other half '30 of 100 "
           "were late'. It is the same fact, so the answers should not differ.",
           SchemaType.PERCENTAGE, VariationMode.FRAMING, "anchoring_framing_effect", 20, 0.7,
           example_prompt=_PCT_PROMPT),
    Recipe("yes_no", "Does a model flip its yes/no decision?",
           "Asks a yes/no question many times and counts how often the model departs from "
           "its own usual answer.",
           SchemaType.BOOLEAN, VariationMode.OCCASION, "categorical_stability", 20, 0.7,
           example_prompt=_BOOL_PROMPT),
    Recipe("ranking", "Does a model reshuffle its priorities?",
           "The same facts are shown in a different order each time. If the ranking changes "
           "with the order, the model gives too much weight to what it reads first or last.",
           SchemaType.RANKING, VariationMode.ORDER, "ranking_stability", 12, 0.7,
           example_prompt=_RANK_PROMPT,
           extra_state={"opts_ranking": "Price, Quality, Delivery speed, Support",
                        "chunks": _RANK_CHUNKS}),
]
RECIPES_BY_ID: dict[str, Recipe] = {r.id: r for r in RECIPES}

# The question each analysis answers, in everyday words (shown in the analysis picker).
PRESET_QUESTIONS: dict[str, str] = {
    "mse_decomposition": "How far off are the answers, and is that bias or noise?",
    "mpd_rms_noise": "How much do the answers differ from each other?",
    "noise_audit": "Why do the models disagree?",
    "categorical_stability": "How often does the model change its mind?",
    "anchoring_framing_effect": "Did an irrelevant detail shift the answers?",
    "ranking_stability": "Does the model keep the same order?",
}

# Which analysis fits which context modifier (shown in the guide and used by the check).
MODE_PRESET_FIT: dict[VariationMode, tuple[str, ...]] = {
    VariationMode.OCCASION: ("mpd_rms_noise", "mse_decomposition", "noise_audit",
                             "categorical_stability", "ranking_stability"),
    VariationMode.ORDER: ("ranking_stability", "mpd_rms_noise", "categorical_stability",
                          "mse_decomposition"),
    VariationMode.ANCHOR: ("anchoring_framing_effect",),
    VariationMode.FRAMING: ("anchoring_framing_effect",),
    VariationMode.PERSONA: ("noise_audit", "mpd_rms_noise", "categorical_stability",
                            "mse_decomposition"),
}


@dataclass(frozen=True)
class Tip:
    """One finding of the setup check. level: 'error' | 'warning' | 'info' | 'ok'."""

    level: str
    message: str


def check_setup(*, schema: SchemaType | None, mode: VariationMode | None, preset_id: str,
                n_windows: int, temperature: float, n_models: int, n_cases: int,
                has_truth: bool) -> list[Tip]:
    """Check whether the chosen settings fit together; return friendly tips.

    'error' = the analysis will not produce a result; 'warning' = it runs but the result
    is weak or misleading; 'info' = optional improvement.
    """
    tips: list[Tip] = []
    preset = PRESETS_BY_ID.get(preset_id)
    if preset is None or schema is None or mode is None:
        return tips

    compat = preset.compatibility(schema)
    if compat == "incompatible":
        tips.append(Tip("error", f"The analysis '{preset.name}' does not work with the answer "
                                 "format you chose. Pick a different analysis in tab 2."))
    elif compat == "caveat":
        tips.append(Tip("warning", f"'{preset.name}' only works approximately with this answer "
                                   "format."))

    if preset.requires_ground_truth and not has_truth:
        tips.append(Tip("error", "This analysis needs the correct answer. In tab 1, tick 'I know "
                                 "the correct answer', or upload a file of questions that has a "
                                 "ground_truth column."))
    if preset.requires_cases > n_cases:
        tips.append(Tip("error", f"This analysis needs at least {preset.requires_cases} different "
                                 "questions. In tab 1, choose 'Several questions (CSV file)' - "
                                 "the built-in example file has 3."))
    if preset_id == "noise_audit" and n_models < 2 and mode != VariationMode.PERSONA:
        tips.append(Tip("error", "A noise audit compares several judges. Select at least 2 "
                                 "models in the sidebar on the left (or let one model play "
                                 "several roles in tab 1)."))
    if preset_id == "anchoring_framing_effect" and mode not in GROUP_MODES:
        tips.append(Tip("error", "This analysis compares two groups of answers. In tab 1, under "
                                 "'What should change between the repetitions?', choose the "
                                 "irrelevant number (anchoring) or the wording (framing)."))
    if mode in GROUP_MODES and preset_id != "anchoring_framing_effect":
        tips.append(Tip("warning", "Anchoring and framing create two groups on purpose. Other "
                                   "analyses would count that intended difference as noise. "
                                   "Use 'Did an irrelevant detail shift the answers?'."))
    if mode == VariationMode.ORDER and schema != SchemaType.RANKING:
        tips.append(Tip("info", "Changing the order of the facts works with any answer "
                                "format, but it tells you most with the 'Ranking' format."))

    if n_windows < preset.min_n:
        tips.append(Tip("error", f"This analysis needs at least {preset.min_n} repetitions per "
                                 "model (slider in the sidebar on the left)."))
    if mode in GROUP_MODES and n_windows < 20:
        tips.append(Tip("warning", f"{n_windows} repetitions give only about {n_windows // 2} "
                                   "answers per group. Use 20 or more to detect real effects."))
    elif n_windows < 10:
        tips.append(Tip("warning", f"{n_windows} repetitions are few. Results become dependable "
                                   "from about 10 repetitions per model."))

    if temperature == 0 and mode == VariationMode.OCCASION:
        tips.append(Tip("warning", "Randomness (temperature) is 0 and every repetition is "
                                   "identical: most models then return (almost) the same text, "
                                   "so noise looks lower than it really is. Try 0.7."))
    elif temperature > 1.2:
        tips.append(Tip("info", "Randomness (temperature) above 1.2 makes answers erratic and "
                                "may break the answer format. 0.7 is a good default."))

    if n_models == 0:
        tips.append(Tip("error", "Select at least one model in the sidebar on the left."))
    elif n_models == 1 and preset_id in ("mpd_rms_noise", "mse_decomposition"):
        tips.append(Tip("info", "Tip: select 2 or 3 models to compare them side by side."))

    if not tips:
        tips.append(Tip("ok", "These settings fit together well."))
    return tips


def setup_ok(tips: list[Tip]) -> bool:
    """True if no tip blocks a meaningful result."""
    return not any(t.level == "error" for t in tips)


# Plain-English explanation of every metric / column the presets return.
METRIC_GLOSSARY: dict[str, str] = {
    # MSE decomposition
    "best_model": "Model with the lowest total error (RMSE).",
    "best_rmse": "Total error of the best model, in answer units. Lower is better.",
    "avg_noise_share": "Share of the squared error that is random scatter (noise) rather than "
                       "systematic skew (bias). Near 1 = mostly noise, near 0 = mostly bias.",
    "mean_bias": "Average answer minus the correct answer. Positive = too high, negative = too "
                 "low, 0 = no systematic skew.",
    "bias_sq": "Bias squared - the systematic part of the error.",
    "noise_sq": "Noise squared (variance across repetitions) - the random part of the error.",
    "mse": "Mean squared error = bias² + noise². Lower is better.",
    "rmse": "Root of the MSE: typical total error in answer units (e.g. percentage points).",
    "rmse_ci95": "Bootstrap 95 % confidence interval for the RMSE. Wide = uncertain; collect "
                 "more repetitions.",
    "noise_share_of_mse": "Share of this model's error that is noise. Near 1 = right on average "
                          "but unreliable; near 0 = consistent but systematically off.",
    # MPD / RMS noise
    "intra_model_mpd": "Average distance between two answers of the same model to the same "
                       "question. 0 = always identical. Lower is better.",
    "inter_model_mpd": "Average distance between the mean answers of two different models. "
                       "Much bigger than intra-model MPD = the choice of model matters a lot.",
    "mean_pairwise_distance": "Average absolute difference between two random answers of this "
                              "model to the same case. Lower is better.",
    "rms_noise_sd": "Typical scatter (standard deviation) around the model's own average.",
    "rms_noise_ci95": "Bootstrap 95 % confidence interval for the RMS noise. Wide = uncertain.",
    "rms_noise_pct_of_scale": "RMS noise as % of the answer scale. < 5 % small, 5-15 % "
                              "noticeable, > 15 % severe.",
    "worst_rms_noise_pct_of_scale": "RMS noise of the least stable model, as % of the scale. "
                                    "< 5 % small, 5-15 % noticeable, > 15 % severe.",
    "most_stable_model": "Model whose answers vary the least.",
    "least_stable_model": "Model whose answers vary the most.",
    # Noise audit
    "system_noise_sd": "Total disagreement in the 'system' of all judges (standard deviation). "
                       "Equals level + pattern + occasion noise (as variances).",
    "level_noise_sd": "How much judges differ in their general level (one always stricter). "
                      "Often fixable by calibration.",
    "stable_pattern_noise_sd": "How much judges react differently to specific cases. Hard to "
                               "fix; often the largest part.",
    "occasion_noise_sd": "How much the same judge varies on the identical prompt. Reduce with "
                         "lower temperature or by averaging several answers.",
    "judges": "Number of judges (models, or model + persona).",
    "cases": "Number of cases (questions) that were analysed.",
    "models": "Number of models that were analysed.",
    "mean_answer": "This judge's average answer across all cases.",
    "level_deviation": "This judge's average minus the overall average. Negative = stricter/"
                       "lower, positive = more lenient/higher.",
    # Categorical stability
    "lowest_flip_rate": "Flip rate of the most stable model. 0 % = never changes its answer.",
    "highest_flip_rate": "Flip rate of the least stable model. For two options, 50 % = coin "
                         "flip.",
    "categories_observed": "How many different answers appeared in total.",
    "mode_agreement": "Share of repetitions giving the model's most common answer. Higher is "
                      "better; 100 % = rock solid.",
    "flip_rate": "How often the model departs from its usual answer (1 - mode agreement).",
    "pairwise_disagreement": "Chance that two random repetitions give different answers.",
    "entropy_normalized": "Spread of answers: 0 = always the same, 1 = evenly spread (random).",
    # Anchoring / framing
    "group_a": "Reference group (e.g. 'baseline' or 'frame_A').",
    "group_b": "Manipulated group (e.g. 'anchored' or 'frame_B').",
    "models_with_significant_effect": "Number of models whose answers differ significantly "
                                      "between the groups (Bonferroni-corrected p < 0.05).",
    "strongest_effect_model": "Model most affected by the manipulation.",
    "strongest_effect_size": "Largest effect size (absolute). Cohen's d: 0.2 small, 0.5 "
                             "medium, 0.8 large.",
    "mean_a": "Average answer in group A.",
    "mean_b": "Average answer in group B.",
    "shift": "Mean of group B minus mean of group A, in answer units.",
    "effect": "Which effect-size measure is used for this model.",
    "effect_size": "Strength of the effect (Cohen's d, risk difference or Cramér's V).",
    "p_value": "Probability of a difference this large by pure chance (uncorrected).",
    "p_value_bonferroni": "The p-value corrected for testing several models (Bonferroni). "
                          "Below 0.05 = significant.",
    "significant_5pct": "True if the corrected p-value is below 0.05.",
    "n_a": "Valid answers in group A.",
    "n_b": "Valid answers in group B.",
    "test": "Which statistical test was used (depends on the answer format).",
    "statistic": "Raw test statistic (U or chi-square). Mainly for experts - read the p-value "
                 "and effect size instead.",
    "note": "Why no test result could be computed for this model.",
    # Ranking
    "worst_mean_tau_distance": "For the least stable model: share of item pairs two random "
                               "repetitions order differently. 0 = identical, 0.5 = random.",
    "worst_kendalls_w": "Agreement among all repetitions of the least stable model. 1 = perfect, "
                        "> 0.7 strong, 0 = none.",
    "mean_kendall_tau_distance": "Share of item pairs two random repetitions order differently. "
                                 "0 = identical, 0.5 = random, 1 = reversed.",
    "kendalls_w": "Overall agreement among all repetitions. 1 = perfect, > 0.7 strong.",
    "identical_to_modal_ranking": "Share of repetitions that produced exactly the most common "
                                  "ranking.",
    "modal_ranking": "The most common ranking.",
    "item_ranked_first": "An item that this model put in first place.",
    "share": "Share of repetitions in which this item was ranked first. A position effect "
             "shows up when it follows the order of the facts.",
    # Generic
    "dropped_params": "Settings the model refused (e.g. temperature). The call was repeated "
                      "without them, so the model's own default applied.",
    "format_retries": "Format reminders sent: 1 = the first reply broke the answer format "
                      "(e.g. 0.42 instead of 42) and the model was asked once more.",
    "n": "Number of valid answers used.",
    "model": "Model name.",
}


def explain_metric(key: str) -> str | None:
    """Plain-English help for a metric / column name, or None if unknown."""
    return METRIC_GLOSSARY.get(key)


# How to read the charts each preset produces.
CHART_TIPS: dict[str, str] = {
    "mse_decomposition": "**Bars:** each bar is a model's total error split into bias² "
                         "(systematic) and noise² (random) - shorter is better. **Scatter:** "
                         "dots on the diagonal are perfect answers; dots consistently above or "
                         "below it show bias, a wide cloud shows noise.",
    "mpd_rms_noise": "**Box plot:** each dot is one answer. A short box = the model answers "
                     "consistently; a tall box or scattered dots = noisy. Boxes at different "
                     "heights = models disagree with each other.",
    "noise_audit": "**Pie:** which kind of noise dominates. **Heat map:** rows are judges, "
                   "columns are cases. A row that is darker/lighter everywhere = level noise; "
                   "single cells that stand out = pattern noise.",
    "categorical_stability": "**Bars:** how often each answer was given per model. One tall bar "
                             "= stable; several similar bars = the model keeps changing its "
                             "mind.",
    "anchoring_framing_effect": "**Violins / histograms:** the two colours are the two groups. "
                                "If they overlap almost completely, the manipulation had no "
                                "effect; if one is clearly shifted, it did.",
    "ranking_stability": "**Bars:** lower tau distance and higher Kendall's W = more stable "
                         "rankings. Compare models side by side.",
}


def suggest_preset(schema: SchemaType, mode: VariationMode, n_cases: int, has_truth: bool,
                   n_models: int) -> str:
    """Best-fitting analysis for the current setup (first one without blocking errors)."""
    candidates = list(MODE_PRESET_FIT.get(mode, ())) + list(PRESETS_BY_ID)
    for pid in dict.fromkeys(candidates):
        if PRESETS_BY_ID[pid].compatibility(schema) != "recommended":
            continue
        tips = check_setup(schema=schema, mode=mode, preset_id=pid, n_windows=50,
                           temperature=0.7, n_models=max(n_models, 1), n_cases=n_cases,
                           has_truth=has_truth)
        if setup_ok(tips):
            return pid
    return next(pid for pid in PRESETS_BY_ID
                if PRESETS_BY_ID[pid].compatibility(schema) == "recommended")
