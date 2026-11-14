"""Start page: pick a ready-made test, run it and read the summary - all in one place."""

from __future__ import annotations

import streamlit as st

from src.context_engine import MODE_LABELS
from src.guidance import MODE_PRESET_FIT, RECIPES, RECIPES_BY_ID, Recipe
from src.model_spec import ModelSpec
from src.presets import PRESETS_BY_ID, Preset, compatibility_matrix
from src.schemas import SCHEMA_LABELS, SchemaType
from ui.state import ICON_OK, ICON_WARNING, TAB_QUESTION, TAB_RESULTS, TAB_RUN
from ui.tab_editor import analysis_label, run_panel, short_name
from ui.tab_setup import SetupResult

PENDING_KEY = "_pending_recipe"
USER_GUIDE_URL = "https://github.com/DarianVoester/llm_noise_bias_audit/blob/main/docs/USER_GUIDE.md"


def apply_pending_recipe() -> None:
    """Apply a recipe chosen on the previous run.

    Must run BEFORE any widget is created: Streamlit forbids changing a widget's
    session-state value after the widget exists in the same run.
    """
    ss = st.session_state
    pending = ss.pop(PENDING_KEY, None)
    if not pending:
        return
    recipe = RECIPES_BY_ID.get(pending["id"])
    if recipe is None:
        return
    for key, value in recipe.session_values(load_example=pending["examples"]).items():
        ss[key] = value
    ss.recipe_notice = recipe.id


def active_recipe() -> Recipe | None:
    """The ready-made test the current settings correspond to, if any."""
    ss = st.session_state
    for recipe in RECIPES:
        expected = recipe.session_values(load_example=False)
        if all(ss.get(k) == expected[k]
               for k in ("schema_type", "var_mode", "preset_id", "case_source")):
            return recipe
    return None


def _needs(recipe: Recipe) -> str:
    parts = [f"{recipe.n_windows} repetitions"]
    if recipe.use_case_file:
        parts.append("uses the 3 example questions")
    if recipe.min_models > 1:
        parts.append(f"needs at least {recipe.min_models} models")
    return " · ".join(parts)


def _recipe_cards(active: Recipe | None) -> None:
    st.markdown("### 1 · Choose what you want to find out")
    st.caption("Each test comes with settings that fit together and an example question, so "
               "you can try it straight away.")
    cols = st.columns(2)
    for i, recipe in enumerate(RECIPES):
        chosen = active is not None and recipe.id == active.id
        with cols[i % 2].container(border=True):
            st.markdown(f"**{recipe.question}**")
            st.caption(f"{recipe.why}  \n{_needs(recipe)}")
            if st.button("Selected" if chosen else "Choose this test",
                         key=f"recipe_{recipe.id}", width="stretch",
                         type="primary" if chosen else "secondary"):
                st.session_state[PENDING_KEY] = {
                    "id": recipe.id,
                    "examples": st.session_state.get("recipe_examples", True)}
                st.rerun()
    st.toggle("Load the example question together with the test", value=True,
              key="recipe_examples",
              help=f"Switch this off if you have already typed your own question in the tab "
                   f"'{TAB_QUESTION}' and want to keep it.")


def _run_step(setup: SetupResult, models: list[ModelSpec], preset: Preset, code: str,
              active: Recipe | None) -> None:
    st.markdown("### 2 · Run the test")
    if active is not None:
        st.markdown(f"Selected test: **{active.question}**")
    else:
        st.markdown(f"Selected: your own settings from the tabs **{TAB_QUESTION}** and "
                    f"**{TAB_RUN}** ({analysis_label(preset.id)})")
    if setup.cases and len(setup.cases) == 1:
        with st.expander("The question that will be asked"):
            st.markdown(f"> {setup.cases[0].prompt}")
            st.caption(f"Change it in the tab **{TAB_QUESTION}**.")
    run_panel(setup, models, preset, code, key="start", on_start_page=True)


def _result_step() -> None:
    ss = st.session_state
    st.markdown("### 3 · Read the result")
    res = ss.eval_result
    if ss.df_results is None:
        st.caption("The result of your test will appear here.")
    elif res is None:
        st.caption(f"The last run produced no answers. The tab **{TAB_RESULTS}** shows why.")
    elif not res.ok or "error" in res.summary_metrics:
        st.warning("The answers are in, but this analysis could not be computed from them. "
                   f"The tab **{TAB_RESULTS}** explains what is missing.", icon=ICON_WARNING)
    else:
        if res.interpretation:
            st.info("\n\n".join(res.interpretation))
        st.caption(f"This is the summary of your most recent run. Charts, all numbers and the "
                   f"downloads are in the tab **{TAB_RESULTS}** at the top of the page.")


def _how_it_works() -> None:
    with st.expander("How does this work?", expanded=False):
        st.markdown(
            "Ask a person the same question on Monday and on Friday and you may get two "
            "different answers. AI models do the same. This app makes that visible.\n\n"
            "1. **It asks one question many times.** Every repetition is a fresh conversation, "
            "so the model cannot see what it answered before.\n"
            "2. **If you like, it changes one detail** between the repetitions: the order of "
            "the facts, an irrelevant number, the wording, or the role the model plays.\n"
            "3. **It measures** how much the answers scatter (**noise**) and whether they are "
            "always off in the same direction (**bias**).\n\n"
            "**Words you will meet**\n"
            "- **Repetition** - one fresh conversation with a model (elsewhere also called a "
            "*context window*).\n"
            "- **Noise** - unwanted variation: same question, different answers.\n"
            "- **Bias** - a consistent error: answers are always too high or always too low.\n"
            "- **Correct answer** (*ground truth*) - the true value, if you happen to know "
            "it.\n"
            "- **Judge** - whoever gives the answers: a model, or a model playing a role.\n"
            "- **Answer format** - the form the answer must have, such as a percentage or "
            "yes/no. Answers in another form cannot be counted.")


def _combination_guide() -> None:
    with st.expander("Building your own test: which settings fit together?", expanded=False):
        st.markdown("**Which analysis works with which answer format** (yes = fits, roughly = "
                    "only an approximation, - = does not work)")
        header = "| Answer format | " + " | ".join(short_name(p) for p in PRESETS_BY_ID) + " |"
        lines = [header, "|---" * (len(PRESETS_BY_ID) + 1) + "|"]
        for row in compatibility_matrix():
            label = SCHEMA_LABELS[SchemaType(row["schema"])]
            lines.append(f"| {label} | " + " | ".join(row[p] for p in PRESETS_BY_ID) + " |")
        st.markdown("\n".join(lines))

        st.markdown("**What changes between the repetitions, and the analyses that suit it**")
        st.markdown("\n".join(
            f"- **{MODE_LABELS[m]}**: " + ", ".join(short_name(p) for p in pids)
            for m, pids in MODE_PRESET_FIT.items()))

        st.markdown(
            "**Values that work well**\n\n"
            "| Setting | Good value | Why |\n|---|---|---|\n"
            "| Repetitions per model | 10 for a noise check, 20 or more for anchoring and "
            "wording tests | Few answers give shaky results; those two tests split the "
            "repetitions into two groups |\n"
            "| Randomness (temperature) | 0.7 | At 0 most models repeat themselves; above 1.2 "
            "answers become erratic |\n"
            "| Models | 2 or 3 | Lets you compare; a noise audit needs at least 2 |\n"
            "| Questions | 1 for a quick check, 3 or more for a noise audit | The causes of "
            "disagreement can only be told apart with several questions |")
        st.caption(f"The tab **{TAB_RUN}** checks your settings and tells you if something "
                   "does not fit.")


def render_guide_tab(setup: SetupResult, models: list[ModelSpec], preset: Preset,
                     code: str) -> None:
    notice = st.session_state.pop("recipe_notice", None)
    if notice in RECIPES_BY_ID:
        st.success(f"Settings applied: **{RECIPES_BY_ID[notice].question}** Now press **Run "
                   "the test** in step 2 below.", icon=ICON_OK)
    st.markdown("New here? Three steps: choose a test, run it, read the result. The practice "
                "models are free and answer instantly, so nothing can go wrong. More "
                f"background is in the [user guide]({USER_GUIDE_URL}).")
    active = active_recipe()
    _recipe_cards(active)
    _run_step(setup, models, preset, code, active)
    _result_step()
    st.divider()
    _how_it_works()
    _combination_guide()
