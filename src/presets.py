"""Pre-built statistical presets with plain-English explanations.

Every preset's ``code`` is a self-contained script that defines
``evaluate(df_results, ground_truth=None) -> dict`` and runs inside the sandbox
(no imports; ``np``, ``pd``, ``stats``, ``px``, ``go`` and ``math`` are pre-injected).

Return contract (all keys optional except summary_metrics):
    {"summary_metrics": {...}, "per_model_table": DataFrame, "tables": {name: DataFrame},
     "figures": [plotly figures], "warnings": [str], "interpretation": [str]}
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas import SchemaType

S = SchemaType


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    compatible_schemas: tuple[SchemaType, ...]
    caveat_schemas: tuple[SchemaType, ...]
    plain_english_summary: str
    kahneman_concept: str
    how_to_interpret: str
    code: str
    min_n: int = 2
    requires_ground_truth: bool = False
    requires_cases: int = 1

    def compatibility(self, schema: SchemaType) -> str:
        """'recommended', 'caveat' or 'incompatible'."""
        if schema in self.compatible_schemas:
            return "recommended"
        if schema in self.caveat_schemas:
            return "caveat"
        return "incompatible"


# Shared helper code, prepended to every preset so each script is self-contained.
HELPERS = '''
# ---------- shared helpers (safe to edit) ----------
def clean_numeric(df_results):
    """Rows that parsed successfully and have a numeric value."""
    d = df_results[df_results["parse_success"] == True].copy()
    d = d[pd.to_numeric(d["parsed_value_num"], errors="coerce").notna()]
    d["x"] = d["parsed_value_num"].astype(float)
    return d


def scale_span(df_results):
    """Width of the answer scale (e.g. 100 for percentages) or None if unbounded."""
    lo = pd.to_numeric(df_results.get("scale_min"), errors="coerce")
    hi = pd.to_numeric(df_results.get("scale_max"), errors="coerce")
    if lo is None or hi is None or lo.dropna().empty or hi.dropna().empty:
        return None
    span = float(hi.dropna().iloc[0]) - float(lo.dropna().iloc[0])
    return span if span > 0 else None


def bootstrap_ci(values, stat, n_boot=1000, seed=12345):
    """95% percentile bootstrap confidence interval of stat(values)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (None, None)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def parse_warnings(df_results, group_col="context_variant", gap=0.10):
    """Warn about failed parses, esp. if failure rates differ between groups."""
    out = []
    total = len(df_results)
    if total == 0:
        return ["No rows to analyse."]
    fail = 1.0 - float(df_results["parse_success"].mean())
    if fail > 0:
        out.append(f"{fail:.0%} of responses could not be parsed and were excluded.")
    if group_col in df_results.columns:
        rates = df_results.groupby(["model", group_col])["parse_success"].mean()
        for model in rates.index.get_level_values(0).unique():
            r = rates.loc[model]
            if len(r) > 1 and (r.max() - r.min()) > gap:
                out.append(f"{model}: parse success differs by {r.max() - r.min():.0%} between "
                           f"{group_col} groups - excluded rows may bias the comparison.")
    return out


def fmt(x, digits=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)
'''

# =============================================================================
# 1. Kahneman MSE decomposition
# =============================================================================
MSE_CODE = HELPERS + '''

def evaluate(df_results, ground_truth=None):
    """MSE = Bias^2 + Noise^2 per model (population SD, ddof=0), per case, then averaged."""
    d = clean_numeric(df_results)
    warnings = parse_warnings(df_results)
    # ground truth: per-case column wins, otherwise the global value from the UI
    if "ground_truth" in d.columns and pd.to_numeric(d["ground_truth"], errors="coerce").notna().any():
        d["truth"] = pd.to_numeric(d["ground_truth"], errors="coerce")
    else:
        d["truth"] = ground_truth
    d = d[pd.to_numeric(d["truth"], errors="coerce").notna()]
    if d.empty:
        return {"summary_metrics": {"error": "No ground truth available. Enter one in Tab 1 "
                                             "or add a ground_truth column to your case file."},
                "warnings": warnings}
    d["truth"] = d["truth"].astype(float)

    per_case = (d.groupby(["model", "case_id"])
                .agg(mean=("x", "mean"), noise=("x", lambda v: float(np.std(v, ddof=0))),
                     truth=("truth", "first"), n=("x", "size"))
                .reset_index())
    per_case["bias"] = per_case["mean"] - per_case["truth"]
    per_case["mse"] = per_case["bias"] ** 2 + per_case["noise"] ** 2

    rows = []
    for model, g in per_case.groupby("model"):
        dm = d[d["model"] == model]
        err = (dm["x"] - dm["truth"]).to_numpy()
        bias2 = float(np.mean(g["bias"] ** 2))
        noise2 = float(np.mean(g["noise"] ** 2))
        lo, hi = bootstrap_ci(np.abs(err), lambda v: float(np.sqrt(np.mean(v ** 2))))
        rows.append({
            "model": model, "n": int(len(dm)), "cases": int(len(g)),
            "mean_bias": fmt(float(np.mean(g["bias"]))),
            "bias_sq": fmt(bias2), "noise_sq": fmt(noise2), "mse": fmt(bias2 + noise2),
            "rmse": fmt(math.sqrt(bias2 + noise2)),
            "rmse_ci95": f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None else "n/a",
            "noise_share_of_mse": fmt(noise2 / (bias2 + noise2) if bias2 + noise2 > 0 else 0.0),
        })
    table = pd.DataFrame(rows).sort_values("mse")
    if (table["n"] < 5).any():
        warnings.append("Some models have fewer than 5 valid answers - estimates are very uncertain.")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Bias²", x=table["model"], y=table["bias_sq"]))
    fig.add_trace(go.Bar(name="Noise²", x=table["model"], y=table["noise_sq"]))
    fig.update_layout(barmode="stack", title="Error decomposition: MSE = Bias² + Noise²",
                      yaxis_title="squared error")
    scatter = px.scatter(table, x="mean_bias", y=table["noise_sq"].pow(0.5), text="model",
                         labels={"mean_bias": "Bias (mean − truth)", "y": "Noise (SD)"},
                         title="Bias vs. noise per model (ideal: bottom center)")
    scatter.add_vline(x=0, line_dash="dash")

    best = table.iloc[0]
    return {
        "summary_metrics": {
            "best_model": best["model"], "best_rmse": best["rmse"],
            "models": int(len(table)), "cases": int(d["case_id"].nunique()),
            "avg_noise_share": fmt(float(table["noise_share_of_mse"].mean())),
        },
        "per_model_table": table,
        "tables": {"per_case": per_case.round(4)},
        "figures": [fig, scatter],
        "warnings": warnings,
        "interpretation": [
            f"{best['model']} has the lowest total error (RMSE {best['rmse']}).",
            f"On average {float(table['noise_share_of_mse'].mean()):.0%} of the squared error "
            "is noise (scatter), the rest is bias (systematic skew).",
        ],
    }
'''

PRESET_MSE = Preset(
    id="mse_decomposition",
    name="Kahneman MSE noise decomposition (needs ground truth)",
    compatible_schemas=(S.PERCENTAGE, S.FLOAT, S.LIKERT, S.BOOLEAN, S.CONFIDENCE),
    caveat_schemas=(),
    plain_english_summary=(
        "Compares the answers with the correct value and splits the total error into two "
        "parts: a consistent skew in one direction (bias) and random scatter (noise)."
    ),
    kahneman_concept=(
        "The book's 'error equation': Mean Squared Error = Bias² + Noise². A key insight of "
        "'Noise' is that both parts hurt equally - reducing scatter improves accuracy exactly "
        "as much as reducing an equal amount of bias, even if you don't know the direction "
        "of the bias."
    ),
    how_to_interpret=(
        "- **Bias** = average answer − truth. Positive = systematically too high.\n"
        "- **Noise** = standard deviation across repetitions (how much the answers "
        "scatter for the same case).\n"
        "- **MSE / RMSE** = total error. RMSE is in the units of the answer (e.g. percentage "
        "points).\n"
        "- **Noise share** close to 1 → the model is roughly right on average but unreliable; "
        "close to 0 → reliable but consistently wrong.\n"
        "- For Boolean answers, truth should be 0 or 1 (or a probability)."
    ),
    code=MSE_CODE,
    min_n=2,
    requires_ground_truth=True,
)


# =============================================================================
# 2. Mean pairwise distance & RMS noise
# =============================================================================
MPD_CODE = HELPERS + '''

def mean_pairwise_distance(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n < 2:
        return float("nan")
    # O(n log n) formula for the mean absolute difference over all pairs
    i = np.arange(1, n + 1)
    return float(2.0 * np.sum((2 * i - n - 1) * v) / (n * (n - 1)))


def evaluate(df_results, ground_truth=None):
    """Intra-model disagreement (same case, same model) and inter-model disagreement."""
    d = clean_numeric(df_results)
    warnings = parse_warnings(df_results)
    span = scale_span(df_results)
    if d.empty:
        return {"summary_metrics": {"error": "No valid numeric answers."}, "warnings": warnings}

    rows = []
    for model, dm in d.groupby("model"):
        mpds, sds = [], []
        for _, dc in dm.groupby("case_id"):
            if len(dc) >= 2:
                mpds.append(mean_pairwise_distance(dc["x"]))
                sds.append(float(np.std(dc["x"], ddof=1)))
        mpd = float(np.mean(mpds)) if mpds else float("nan")
        rms = float(np.sqrt(np.mean(np.square(sds)))) if sds else float("nan")
        resid = (dm["x"] - dm.groupby("case_id")["x"].transform("mean")).to_numpy()
        lo, hi = bootstrap_ci(resid, lambda v: float(np.sqrt(np.mean(v ** 2))))
        rows.append({
            "model": model, "n": int(len(dm)),
            "mean_pairwise_distance": fmt(mpd), "rms_noise_sd": fmt(rms),
            "rms_noise_ci95": f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None else "n/a",
            "rms_noise_pct_of_scale": fmt(100 * rms / span, 1) if span else None,
        })
    table = pd.DataFrame(rows).sort_values("rms_noise_sd")

    # inter-model: distance between model means for the same case
    means = d.groupby(["case_id", "model"])["x"].mean().unstack()
    inter = []
    for _, row in means.iterrows():
        vals = row.dropna().to_numpy()
        if len(vals) >= 2:
            inter.append(mean_pairwise_distance(vals))
    inter_mpd = float(np.mean(inter)) if inter else float("nan")
    intra_mpd = float(pd.to_numeric(table["mean_pairwise_distance"]).mean())

    fig = px.box(d, x="model", y="x", color="case_id" if d["case_id"].nunique() > 1 else None,
                 points="all", title="Answer distribution per model (each dot = one answer)",
                 labels={"x": "answer"})
    summary = {
        "intra_model_mpd": fmt(intra_mpd),
        "inter_model_mpd": fmt(inter_mpd) if inter else "needs ≥ 2 models",
        "most_stable_model": table.iloc[0]["model"],
        "least_stable_model": table.iloc[-1]["model"],
    }
    if span:
        summary["worst_rms_noise_pct_of_scale"] = table["rms_noise_pct_of_scale"].max()
    interp = [f"Two random answers of the same model differ on average by "
              f"{fmt(intra_mpd, 2)} units (intra-model noise)."]
    if inter:
        interp.append(f"The average answers of two different models differ by {fmt(inter_mpd, 2)} "
                      "units (inter-model disagreement).")
    return {"summary_metrics": summary, "per_model_table": table, "figures": [fig],
            "warnings": warnings, "interpretation": interp}
'''

PRESET_MPD = Preset(
    id="mpd_rms_noise",
    name="Mean pairwise distance & RMS noise (no ground truth needed)",
    compatible_schemas=(S.PERCENTAGE, S.FLOAT, S.LIKERT, S.BOOLEAN, S.CONFIDENCE),
    caveat_schemas=(),
    plain_english_summary=(
        "Measures how much the answers disagree with each other - without needing to know "
        "the right answer. If you asked the same model the same question twice, how far apart "
        "would the two answers typically be?"
    ),
    kahneman_concept=(
        "The 'noise audit': in the book, an insurance company asked underwriters to price the "
        "same cases and found a median difference of 55 % between two random underwriters, "
        "while executives expected about 10 %. Noise can be measured without knowing the truth "
        "- disagreement alone proves error exists."
    ),
    how_to_interpret=(
        "- **Mean pairwise distance (MPD)** = average absolute difference between two random "
        "answers of the same model for the same case. 0 = all repetitions agreed exactly.\n"
        "- **RMS noise (SD)** = typical scatter around the model's own average.\n"
        "- **% of scale**: for percentages, RMS noise of 2 = 2 % of the scale. Rule of thumb: "
        "< 5 % small, 5–15 % noticeable, > 15 % severe instability.\n"
        "- **Inter-model MPD** = how much different models disagree on average. If this is "
        "much bigger than intra-model MPD, the choice of model matters more than chance."
    ),
    code=MPD_CODE,
)


# =============================================================================
# 3. Noise audit: Level / Stable pattern / Occasion noise
# =============================================================================
NOISE_AUDIT_CODE = HELPERS + '''

def evaluate(df_results, ground_truth=None):
    """Two-way decomposition over judges x cases with replicates.

    judge = model (or model + persona when persona jitter was used)
    System noise² = Level noise² + Stable pattern noise² + Occasion noise²
    """
    d = clean_numeric(df_results)
    warnings = parse_warnings(df_results)
    use_persona = "persona" in d.columns and d["persona"].notna().any()
    d["judge"] = d["model"] + (" | " + d["context_variant"].astype(str) if use_persona else "")
    J, C = d["judge"].nunique(), d["case_id"].nunique()
    if J < 2 or C < 2:
        return {"summary_metrics": {
            "error": f"This analysis needs at least 2 judges and 2 cases (found {J} judge(s), "
                     f"{C} case(s)). Upload a case file with several cases in Tab 1 and "
                     "select at least two models (or use persona jitter)."},
            "warnings": warnings}

    cell = d.groupby(["judge", "case_id"])["x"].agg(["mean", "var", "size"]).reset_index()
    grid = cell.pivot(index="judge", columns="case_id", values="mean")
    if grid.isna().any().any():
        warnings.append("Some judge × case cells have no valid answer; those cases are ignored.")
        grid = grid.dropna(axis=1)
    grand = float(grid.values.mean())
    case_means = grid.mean(axis=0)
    judge_means = grid.mean(axis=1)

    level_var = float(np.var(judge_means.to_numpy(), ddof=0))
    resid = grid.to_numpy() - case_means.to_numpy()[None, :] - judge_means.to_numpy()[:, None] + grand
    interaction_var = float(np.mean(resid ** 2))
    occ = cell["var"].dropna()
    occasion_var = float(occ.mean()) if len(occ) else 0.0
    # cell means still contain occasion noise / n; remove that part (floor at 0)
    mean_n = float(cell["size"].mean())
    stable_pattern_var = max(0.0, interaction_var - occasion_var / max(mean_n, 1.0))
    system_var = level_var + stable_pattern_var + occasion_var

    parts = pd.DataFrame({
        "component": ["Level noise", "Stable pattern noise", "Occasion noise"],
        "variance": [level_var, stable_pattern_var, occasion_var],
    })
    parts["sd"] = np.sqrt(parts["variance"])
    parts["share_of_system_noise"] = parts["variance"] / system_var if system_var > 0 else 0.0
    parts = parts.round(4)

    judge_table = pd.DataFrame({
        "judge": list(judge_means.index),
        "mean_answer": judge_means.to_numpy().round(3),
        "level_deviation": (judge_means.to_numpy() - grand).round(3),
    }).sort_values("level_deviation")

    fig = px.pie(parts, names="component", values="variance",
                 title="Where does the system noise come from?")
    heat = px.imshow(grid, aspect="auto", color_continuous_scale="RdBu_r",
                     title="Mean answer per judge (rows) and case (columns)")
    top = parts.sort_values("variance", ascending=False).iloc[0]
    return {
        "summary_metrics": {
            "system_noise_sd": fmt(math.sqrt(system_var)),
            "level_noise_sd": fmt(math.sqrt(level_var)),
            "stable_pattern_noise_sd": fmt(math.sqrt(stable_pattern_var)),
            "occasion_noise_sd": fmt(math.sqrt(occasion_var)),
            "judges": int(J), "cases": int(grid.shape[1]),
        },
        "per_model_table": judge_table,
        "tables": {"decomposition": parts},
        "figures": [fig, heat],
        "warnings": warnings,
        "interpretation": [
            f"The largest source of noise is {str(top['component']).lower()} "
            f"({float(top['share_of_system_noise']):.0%} of the variance)."
        ],
    }
'''

PRESET_NOISE_AUDIT = Preset(
    id="noise_audit",
    name="Noise audit: level vs. pattern vs. occasion noise",
    compatible_schemas=(S.PERCENTAGE, S.FLOAT, S.LIKERT, S.CONFIDENCE),
    caveat_schemas=(S.BOOLEAN,),
    plain_english_summary=(
        "Finds out WHY the answers disagree: is one model simply always harsher than another, "
        "does each model have its own quirks on specific cases, or is it just random wobble?"
    ),
    kahneman_concept=(
        "The book splits 'system noise' into three parts: level noise (some judges are "
        "generally more severe), stable pattern noise (a judge reacts idiosyncratically to "
        "particular cases) and occasion noise (the same judge decides differently on another "
        "occasion). Here, judges are models (or model + persona) and cases are your prompts."
    ),
    how_to_interpret=(
        "- **Level noise** high → models differ in their general level (one is always "
        "stricter). A simple calibration offset could fix much of this.\n"
        "- **Stable pattern noise** high → models treat specific cases differently; hard to "
        "fix, and often the largest part in the book's studies.\n"
        "- **Occasion noise** high → the same model gives different answers to the identical "
        "prompt. Lower the temperature or average several answers.\n"
        "- Needs ≥ 2 cases and ≥ 2 judges. With Boolean answers it works on proportions and "
        "is only approximate."
    ),
    code=NOISE_AUDIT_CODE,
    requires_cases=2,
)


# =============================================================================
# 4. Categorical stability
# =============================================================================
CATEGORICAL_CODE = HELPERS + '''

def label_of(row):
    v = row["parsed_value"]
    if isinstance(v, list):
        return " > ".join(str(x) for x in v)
    if row.get("schema_type") == "boolean" and v in (0.0, 1.0):
        return "True" if v == 1.0 else "False"
    return str(v)


def evaluate(df_results, ground_truth=None):
    """Mode agreement, flip rate (1 - mode share), pairwise disagreement, normalised entropy."""
    d = df_results[df_results["parse_success"] == True].copy()
    warnings = parse_warnings(df_results)
    if d.empty:
        return {"summary_metrics": {"error": "No valid answers."}, "warnings": warnings}
    d["label"] = [label_of(r) for r in d.to_dict("records")]
    n_categories = max(int(d["label"].nunique()), 2)

    rows = []
    for (model, case), g in d.groupby(["model", "case_id"]):
        counts = g["label"].value_counts()
        c = counts.to_numpy()
        n = int(c.sum())
        p = c / n
        mode_share = float(p.max())
        pairwise = float(1.0 - np.sum(c * (c - 1)) / (n * (n - 1))) if n > 1 else 0.0
        entropy = float(-np.sum(p * np.log(p)))
        rows.append({"model": model, "case_id": case, "n": n, "mode": counts.index[0],
                     "mode_agreement": mode_share, "flip_rate": 1.0 - mode_share,
                     "pairwise_disagreement": pairwise,
                     "entropy_normalized": entropy / math.log(n_categories)})
    per_case = pd.DataFrame(rows)
    table = (per_case.groupby("model")
             .agg(n=("n", "sum"), mode_agreement=("mode_agreement", "mean"),
                  flip_rate=("flip_rate", "mean"),
                  pairwise_disagreement=("pairwise_disagreement", "mean"),
                  entropy_normalized=("entropy_normalized", "mean"))
             .reset_index().round(3).sort_values("flip_rate"))
    cis = []
    for model in table["model"]:
        g = d[d["model"] == model]
        is_mode = (g["label"] == g.groupby("case_id")["label"].transform(
            lambda s: s.value_counts().index[0])).astype(float).to_numpy()
        lo, hi = bootstrap_ci(is_mode, lambda v: float(1 - np.mean(v)))
        cis.append(f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None else "n/a")
    table["flip_rate_ci95"] = cis

    dist = d.groupby(["model", "label"]).size().reset_index(name="count")
    fig = px.bar(dist, x="model", y="count", color="label", barmode="stack",
                 title="How often did each model give each answer?")
    best, worst = table.iloc[0], table.iloc[-1]
    return {
        "summary_metrics": {
            "most_stable_model": best["model"], "lowest_flip_rate": float(best["flip_rate"]),
            "least_stable_model": worst["model"], "highest_flip_rate": float(worst["flip_rate"]),
            "categories_observed": int(d["label"].nunique()),
        },
        "per_model_table": table,
        "tables": {"per_case": per_case.round(3)},
        "figures": [fig],
        "warnings": warnings,
        "interpretation": [
            f"{worst['model']} departs from its own most common answer in "
            f"{float(worst['flip_rate']):.0%} of the repetitions."
        ],
    }
'''

PRESET_CATEGORICAL = Preset(
    id="categorical_stability",
    name="Categorical stability (flip rate & Shannon entropy)",
    compatible_schemas=(S.BOOLEAN, S.ENUM),
    caveat_schemas=(S.LIKERT,),
    plain_english_summary=(
        "For yes/no or pick-a-category questions: how often does the model change its mind "
        "when you ask it the same thing in separate, fresh conversations?"
    ),
    kahneman_concept=(
        "Occasion noise for categorical decisions - like the book's examples of judges or "
        "doctors who reach a different verdict on an identical case depending on the day. "
        "For a yes/no decision, any disagreement means some decisions are simply wrong."
    ),
    how_to_interpret=(
        "- **Mode agreement** = share of repetitions giving the most common answer. 100 % = rock "
        "solid.\n"
        "- **Flip rate** = 1 − mode agreement: how often the model deviates from its own usual "
        "answer. 0 % = always the same; for two options, 50 % = a coin flip.\n"
        "- **Pairwise disagreement** = chance that two random repetitions disagree.\n"
        "- **Normalised entropy**: 0 = complete agreement, 1 = answers spread evenly across "
        "the observed categories (pure randomness)."
    ),
    code=CATEGORICAL_CODE,
)


# =============================================================================
# 5. Anchoring / framing effect size
# =============================================================================
EFFECT_CODE = HELPERS + '''

def pick_groups(variants):
    v = set(variants)
    for a, b in (("baseline", "anchored"), ("frame_A", "frame_B")):
        if a in v and b in v:
            return a, b
    vs = sorted(v)
    return (vs[0], vs[1]) if len(vs) >= 2 else (None, None)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else float("nan")


def compare_numeric(a, b):
    if len(a) < 2 or len(b) < 2:
        return {"test": "Mann-Whitney U", "note": "fewer than 2 valid answers per group"}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"test": "Mann-Whitney U", "statistic": fmt(u.statistic), "p_value": fmt(u.pvalue, 4),
            "mean_a": fmt(np.mean(a)), "mean_b": fmt(np.mean(b)),
            "shift": fmt(np.mean(b) - np.mean(a)), "effect": "Cohen's d",
            "effect_size": fmt(cohens_d(a, b))}


def compare_boolean(a, b):
    a1, b1 = int(np.sum(a)), int(np.sum(b))
    _, p = stats.fisher_exact([[a1, len(a) - a1], [b1, len(b) - b1]])
    ra = a1 / len(a) if len(a) else float("nan")
    rb = b1 / len(b) if len(b) else float("nan")
    return {"test": "Fisher exact", "p_value": fmt(p, 4), "mean_a": fmt(ra), "mean_b": fmt(rb),
            "shift": fmt(rb - ra), "effect": "risk difference", "effect_size": fmt(rb - ra)}


def compare_categorical(g):
    table = pd.crosstab(g["context_variant"], g["parsed_value_json"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "chi-square", "note": "all answers identical - no effect measurable",
                "p_value": 1.0, "effect": "Cramér's V", "effect_size": 0.0}
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = float(table.to_numpy().sum())
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"test": "chi-square", "statistic": fmt(chi2), "p_value": fmt(p, 4),
            "effect": "Cramér's V", "effect_size": fmt(v)}
'''


EFFECT_CODE += '''

def evaluate(df_results, ground_truth=None):
    """Group A vs. group B per model."""
    warnings = parse_warnings(df_results)
    ok = df_results[df_results["parse_success"] == True].copy()
    if ok.empty:
        return {"summary_metrics": {"error": "No valid answers."}, "warnings": warnings}
    group_a, group_b = pick_groups(ok["context_variant"].unique())
    if group_a is None:
        return {"summary_metrics": {"error": "Needs two groups of answers. In tab 1, "
                                             "let an irrelevant number (anchoring) or the "
                                             "wording (framing) change between repetitions."},
                "warnings": warnings}
    schema = str(ok["schema_type"].iloc[0]) if "schema_type" in ok.columns else "float"
    rows = []
    for model, g in ok.groupby("model"):
        g = g[g["context_variant"].isin([group_a, group_b])]
        a = g[g["context_variant"] == group_a]["parsed_value_num"].astype(float).to_numpy()
        b = g[g["context_variant"] == group_b]["parsed_value_num"].astype(float).to_numpy()
        row = {"model": model, "n_a": int((g["context_variant"] == group_a).sum()),
               "n_b": int((g["context_variant"] == group_b).sum())}
        if schema == "enum":
            row.update(compare_categorical(g))
        elif schema == "boolean":
            row.update(compare_boolean(a, b))
        else:
            row.update(compare_numeric(a, b))
        rows.append(row)
    table = pd.DataFrame(rows)
    p = pd.to_numeric(table["p_value"], errors="coerce") if "p_value" in table.columns \\
        else pd.Series(float("nan"), index=table.index)
    table["p_value_bonferroni"] = (p * len(table)).clip(upper=1.0)
    table["significant_5pct"] = table["p_value_bonferroni"] < 0.05
    if (table[["n_a", "n_b"]].min(axis=1) < 10).any():
        warnings.append("Fewer than 10 answers per group - small effects cannot be detected reliably.")

    plot_df = ok[ok["context_variant"].isin([group_a, group_b])]
    if schema == "enum":
        fig = px.histogram(plot_df, x="parsed_value_json", color="context_variant",
                           barmode="group", facet_col="model",
                           title=f"Answers by group: {group_a} vs. {group_b}")
    else:
        fig = px.violin(plot_df, x="model", y="parsed_value_num", color="context_variant",
                        box=True, points="all", title=f"Answers by group: {group_a} vs. {group_b}")
    eff = pd.to_numeric(table["effect_size"], errors="coerce") if "effect_size" in table.columns \\
        else pd.Series(dtype=float)
    n_sig = int(table["significant_5pct"].sum())
    strongest = table.loc[eff.abs().idxmax(), "model"] if eff.notna().any() else None
    return {
        "summary_metrics": {
            "group_a": group_a, "group_b": group_b,
            "models_with_significant_effect": n_sig,
            "strongest_effect_model": strongest,
            "strongest_effect_size": fmt(eff.abs().max()) if eff.notna().any() else None,
        },
        "per_model_table": table,
        "figures": [fig],
        "warnings": warnings,
        "interpretation": [
            f"{n_sig} of {len(table)} model(s) show a statistically significant difference "
            f"between '{group_a}' and '{group_b}' (Bonferroni-corrected p < 0.05)."
        ],
    }
'''

PRESET_EFFECT = Preset(
    id="anchoring_framing_effect",
    name="Anchoring & framing effect size (Mann-Whitney U, Cohen's d, Fisher)",
    compatible_schemas=(S.PERCENTAGE, S.FLOAT, S.LIKERT, S.BOOLEAN, S.CONFIDENCE, S.ENUM),
    caveat_schemas=(),
    plain_english_summary=(
        "Compares two groups of repetitions - e.g. with vs. without an irrelevant number, "
        "or 'gain' vs. 'loss' wording - and measures how strongly that irrelevant change "
        "shifted the answers."
    ),
    kahneman_concept=(
        "Anchoring and framing are classic biases from Kahneman's research: an irrelevant "
        "number pulls estimates towards it, and logically equivalent wording ('90 % survive' "
        "vs. '10 % die') changes decisions. In 'Noise', such irrelevant context effects are a "
        "major source of occasion and pattern noise."
    ),
    how_to_interpret=(
        "- **Shift** = mean of group B − mean of group A, in answer units.\n"
        "- **Cohen's d**: |d| ≈ 0.2 small, 0.5 medium, ≥ 0.8 large effect.\n"
        "- **p-value** (Mann-Whitney U / Fisher / chi-square): below 0.05 means the difference "
        "is unlikely to be pure chance. Corrected for testing several models (Bonferroni).\n"
        "- **Risk difference** (Boolean): change in the share of 'True' answers, e.g. 0.30 = "
        "30 percentage points more 'True' in group B.\n"
        "- **Cramér's V** (categories): 0 = no association, ≥ 0.3 = substantial.\n"
        "- A large effect with a non-significant p-value usually means: collect more repetitions."
    ),
    code=EFFECT_CODE,
    min_n=4,
)


# =============================================================================
# 6. Ranking stability
# =============================================================================
RANKING_CODE = HELPERS + '''

def kendall_tau_distance(r1, r2):
    """Normalised Kendall tau distance: share of item pairs ordered differently (0..1)."""
    pos = {item: i for i, item in enumerate(r2)}
    seq = [pos[item] for item in r1]
    n = len(seq)
    if n < 2:
        return 0.0
    discordant = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
    return discordant / (n * (n - 1) / 2)


def kendalls_w(rankings):
    """Kendall's coefficient of concordance W (0 = no agreement, 1 = identical rankings)."""
    m = len(rankings)
    items = sorted(rankings[0])
    n = len(items)
    if m < 2 or n < 2:
        return float("nan")
    rank_sums = np.zeros(n)
    for r in rankings:
        pos = {item: i + 1 for i, item in enumerate(r)}
        rank_sums += np.array([pos[it] for it in items])
    s = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
    return 12.0 * s / (m ** 2 * (n ** 3 - n))


def evaluate(df_results, ground_truth=None):
    """Mean pairwise normalised Kendall tau distance and Kendall's W per model."""
    warnings = parse_warnings(df_results)
    d = df_results[df_results["parse_success"] == True].copy()
    d = d[[isinstance(v, list) for v in d["parsed_value"]]]
    if d.empty:
        return {"summary_metrics": {"error": "No valid rankings. Choose the 'Ranking' output "
                                             "format in Tab 1."}, "warnings": warnings}
    rows, first_rows = [], []
    for model, dm in d.groupby("model"):
        dists, ws = [], []
        for _, dc in dm.groupby("case_id"):
            ranks = list(dc["parsed_value"])
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    dists.append(kendall_tau_distance(ranks[i], ranks[j]))
            ws.append(kendalls_w(ranks))
        all_ranks = list(dm["parsed_value"])
        modal = pd.Series([" > ".join(r) for r in all_ranks]).value_counts()
        rows.append({
            "model": model, "n": int(len(dm)),
            "mean_kendall_tau_distance": fmt(float(np.mean(dists)) if dists else 0.0),
            "kendalls_w": fmt(float(np.nanmean(ws)) if ws else float("nan")),
            "identical_to_modal_ranking": fmt(float(modal.iloc[0]) / len(all_ranks)),
            "modal_ranking": modal.index[0],
        })
        firsts = pd.Series([r[0] for r in all_ranks]).value_counts(normalize=True)
        for item, share in firsts.items():
            first_rows.append({"model": model, "item_ranked_first": item, "share": float(share)})
    table = pd.DataFrame(rows).sort_values("mean_kendall_tau_distance")
    fig = px.bar(pd.DataFrame(first_rows), x="model", y="share", color="item_ranked_first",
                 barmode="stack", title="Which item was ranked first? (share of repetitions)")
    worst = table.iloc[-1]
    if d["context_variant"].astype(str).str.startswith("order_").any():
        warnings.append("Order permutation detected - compare which item is ranked first with "
                        "the chunk shown first (variant name, e.g. order_BAC) to spot primacy.")
    return {
        "summary_metrics": {
            "most_stable_model": table.iloc[0]["model"],
            "least_stable_model": worst["model"],
            "worst_mean_tau_distance": worst["mean_kendall_tau_distance"],
            "worst_kendalls_w": worst["kendalls_w"],
        },
        "per_model_table": table,
        "figures": [fig],
        "warnings": warnings,
        "interpretation": [
            f"For {worst['model']}, two random repetitions order "
            f"{float(worst['mean_kendall_tau_distance']):.0%} of all item pairs differently."
        ],
    }
'''


PRESET_RANKING = Preset(
    id="ranking_stability",
    name="Ranking stability (Kendall's tau distance & Kendall's W)",
    compatible_schemas=(S.RANKING,),
    caveat_schemas=(),
    plain_english_summary=(
        "For ranking tasks: does the model put the items in the same order every time, or does "
        "it reshuffle its priorities when the context or the chunk order changes?"
    ),
    kahneman_concept=(
        "Pattern noise in rankings - comparable to the book's hiring and performance-rating "
        "examples, where evaluators order the same candidates differently. The book recommends "
        "relative judgments and structured comparisons partly because they reduce such noise."
    ),
    how_to_interpret=(
        "- **Mean Kendall tau distance** = share of item pairs that two random repetitions order "
        "differently. 0 = identical rankings every time, 0.5 = as different as random "
        "orderings, 1 = completely reversed.\n"
        "- **Kendall's W** = overall agreement among all repetitions: 1 = perfect agreement, "
        "0 = no agreement. Above 0.7 is usually considered strong.\n"
        "- **Identical to modal ranking** = share of repetitions that produced exactly the most "
        "common ranking."
    ),
    code=RANKING_CODE,
)

# =============================================================================
# Registry
# =============================================================================
PRESETS: list[Preset] = [
    PRESET_MSE,
    PRESET_MPD,
    PRESET_NOISE_AUDIT,
    PRESET_CATEGORICAL,
    PRESET_EFFECT,
    PRESET_RANKING,
]
PRESETS_BY_ID: dict[str, Preset] = {p.id: p for p in PRESETS}


def presets_for_schema(schema: SchemaType) -> list[tuple[Preset, str]]:
    """All presets with their compatibility: recommended first, incompatible last."""
    order = {"recommended": 0, "caveat": 1, "incompatible": 2}
    pairs = [(p, p.compatibility(schema)) for p in PRESETS]
    return sorted(pairs, key=lambda pc: order[pc[1]])


def compatibility_matrix() -> list[dict[str, str]]:
    """Schema x preset matrix for documentation and the UI."""
    symbol = {"recommended": "yes", "caveat": "roughly", "incompatible": "-"}
    return [{"schema": s.value, **{p.id: symbol[p.compatibility(s)] for p in PRESETS}}
            for s in SchemaType]

