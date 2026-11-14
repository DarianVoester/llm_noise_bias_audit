"""Tab 1 - the question, the answer format, what changes between repetitions, preview."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.cases import EXAMPLE_CASES_CSV, Case, CaseSetError, cases_from_csv, single_case
from src.config import DEFAULT_PERSONAS
from src.context_engine import (
    DEFAULT_SYSTEM_PROMPT,
    MODE_EXPLANATIONS,
    MODE_LABELS,
    ContextWindow,
    VariationConfig,
    VariationMode,
    build_windows,
)
from src.guidance import CASE_SOURCE_FILE, CASE_SOURCE_SINGLE
from src.schemas import SCHEMA_LABELS, SchemaConfig, SchemaType
from ui.state import ICON_INFO, ICON_WARNING, TAB_RUN, TAB_START

EXAMPLE_PROMPT = ("A mid-sized company plans to launch a new product in 6 months. The team "
                  "has 8 people, the budget is fixed, and two similar past projects were late. "
                  "Estimate the probability (0-100) that the launch happens on time.")


@dataclass
class SetupResult:
    cases: list[Case] | None
    schema: SchemaConfig | None
    variation: VariationConfig | None
    ground_truth: float | None
    windows: list[ContextWindow]
    errors: list[str]


SCHEMA_HELP = {
    SchemaType.BOOLEAN: "For questions with a yes or no answer. You can then see how often "
                        "the model changes its mind.",
    SchemaType.PERCENTAGE: "For probabilities and shares, from 0 to 100. The most flexible "
                           "format: it works with every analysis except ranking.",
    SchemaType.FLOAT: "For any other number, such as a price, a duration or a count.",
    SchemaType.LIKERT: "For ratings, such as 1 (very poor) to 5 (very good). Good for "
                       "'how strongly…' questions.",
    SchemaType.ENUM: "The model must pick exactly one of the options you list, for example "
                     "Low, Medium or High.",
    SchemaType.RANKING: "The model must put the items you list in order. Works best together "
                        "with 'The order of the facts' in step 3.",
    SchemaType.CONFIDENCE: "The model gives a decision and says how sure it is (0 to 100). "
                           "The confidence is analysed like a percentage.",
}


def _schema_inputs() -> tuple[SchemaConfig | None, float | None, list[str]]:
    errors: list[str] = []
    stype = st.selectbox("Answer format", list(SchemaType),
                         format_func=lambda s: SCHEMA_LABELS[s],
                         # The example prompt asks for a probability, so start with the
                         # matching format (a yes/no format would contradict the question).
                         key=_default("schema_type", SchemaType.PERCENTAGE),
                         help="Answers can only be compared if they all have the same form. "
                              "The app tells the model which form to use and reads the answer "
                              "back automatically.")
    st.caption(SCHEMA_HELP[stype])
    kwargs: dict = {}
    truth = None
    c1, c2 = st.columns(2)
    if stype == SchemaType.LIKERT:
        kwargs["likert_min"] = int(c1.number_input("Lowest rating", step=1,
                                                   key=_default("lik_min", 1)))
        kwargs["likert_max"] = int(c2.number_input("Highest rating", step=1,
                                                   key=_default("lik_max", 5)))
    if stype in (SchemaType.ENUM, SchemaType.RANKING):
        default = "Low, Medium, High" if stype == SchemaType.ENUM else "A, B, C, D"
        raw = st.text_input("Options, separated by commas",
                            key=_default(f"opts_{stype}", default),
                            help="At least two different options. Mention them in your "
                                 "question as well, so the model knows what they mean.")
        kwargs["options"] = [o.strip() for o in raw.split(",")]
    if stype not in (SchemaType.ENUM, SchemaType.RANKING):
        use_truth = c1.checkbox("I know the correct answer", key="use_truth",
                                help="Optional. Only needed if you want to measure how far "
                                     "off the models are. If you upload a file of questions "
                                     "with a ground_truth column, that column is used instead.")
        if use_truth:
            truth = float(c2.number_input("Correct answer", key=_default("truth_val", 50.0)))
    try:
        schema = SchemaConfig(type=stype, **kwargs)
    except ValueError as exc:
        errors.append(f"Answer format: {_short(exc)}")
        schema = None
    return schema, truth, errors


def _default(key: str, value: object) -> str:
    """Seed a widget's value in session state (lets recipes pre-fill widgets cleanly)."""
    st.session_state.setdefault(key, value)
    return key


def _short(exc: Exception) -> str:
    """First meaningful line of a (pydantic) validation error."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("Value error,"):
            return ln.removeprefix("Value error,").split("[")[0].strip()
    return lines[-1] if lines else str(exc)


def _variation_inputs() -> tuple[VariationConfig | None, list[str]]:
    mode = st.selectbox("What is different from one repetition to the next?",
                        list(VariationMode), format_func=lambda m: MODE_LABELS[m],
                        key="var_mode",
                        help="'Nothing' is the plain consistency check. The other choices "
                             "test whether a detail that should not matter changes the "
                             "answers.")
    st.info(MODE_EXPLANATIONS[mode], icon=ICON_INFO)
    kw: dict = {"mode": mode}
    if mode == VariationMode.ORDER:
        kw["chunks_text"] = st.text_area(
            "Facts to reorder (put a line with only --- between them)", height=160,
            key=_default("chunks",
                         "Fact 1: The team delivered the last project 3 weeks late.\n---\n"
                         "Fact 2: The team has grown by two senior engineers.\n---\n"
                         "Fact 3: The supplier confirmed all parts are in stock."),
            help="2 to 5 short facts. They are placed before your question, in a different "
                 "order each time.")
    elif mode == VariationMode.ANCHOR:
        kw["anchor_text"] = st.text_input(
            "Sentence with the irrelevant number (added to a random half of the repetitions)",
            key=_default("anchor", "For reference, a randomly generated number for today is 92."),
            help="The sentence should have nothing to do with the question, and its number "
                 "should be far from a sensible answer. Then any pull towards it is easy to "
                 "see.")
    elif mode == VariationMode.FRAMING:
        kw["frame_a_text"] = st.text_area(
            "Wording A", height=70,
            key=_default("fa", "Out of 100 similar launches, 70 were on time."),
            help="Wording A and wording B must state the same fact in different words.")
        kw["frame_b_text"] = st.text_area(
            "Wording B", height=70,
            key=_default("fb", "Out of 100 similar launches, 30 were late."))
    elif mode == VariationMode.PERSONA:
        raw = st.text_area("Roles (one per line)", height=120,
                           key=_default("personas", "\n".join(DEFAULT_PERSONAS)),
                           help="At least two. The repetitions take turns through the roles.")
        kw["personas"] = [p for p in raw.splitlines() if p.strip()]
    if mode != VariationMode.PERSONA:
        with st.expander("Background instruction for the model (rarely needs changing)"):
            kw["system_prompt"] = st.text_area(
                "Background instruction (system prompt)", height=70,
                key=_default("sys_prompt", DEFAULT_SYSTEM_PROMPT),
                help="A short instruction sent along with every repetition. It tells the "
                     "model how to behave in general.")
    try:
        return VariationConfig(**kw), []
    except ValueError as exc:
        return None, [f"Change between repetitions: {_short(exc)}"]


def _case_inputs(truth: float | None) -> tuple[list[Case] | None, list[str]]:
    source = st.radio("How many questions do you want to test?",
                      [CASE_SOURCE_SINGLE, CASE_SOURCE_FILE],
                      horizontal=True, key="case_source",
                      help="One question is enough for most tests. Only the noise audit "
                           "needs several, which you upload as a file.")
    if source == CASE_SOURCE_SINGLE:
        prompt = st.text_area("Your question", height=150,
                              key=_default("base_prompt", EXAMPLE_PROMPT),
                              help="Ask for one judgement, as you would ask a person. You do "
                                   "not have to explain the answer format; the app adds that "
                                   "for you.")
        try:
            return single_case(prompt, truth), []
        except CaseSetError as exc:
            return None, [str(exc)]
    st.caption("A CSV file is a simple table you can save from Excel or Google Sheets. It "
               "needs the columns `case_id` (a short name) and `prompt` (the question), and "
               "may have `ground_truth` (the correct answer).")
    st.download_button("Download an example file", EXAMPLE_CASES_CSV, "example_cases.csv",
                       "text/csv")
    up = st.file_uploader("Upload your CSV file", type=["csv"], key="case_file")
    if up is None:
        st.caption("No file uploaded, so the built-in example file is used.")
        data: bytes | str = EXAMPLE_CASES_CSV
    else:
        if up.size > 5 * 1024 * 1024:
            return None, ["The file is larger than 5 MB."]
        data = up.getvalue()
    try:
        cases = cases_from_csv(data)
    except CaseSetError as exc:
        return None, [str(exc)]
    if truth is not None:
        cases = [c if c.ground_truth is not None else Case(c.case_id, c.prompt, truth)
                 for c in cases]
    st.caption(f"{len(cases)} question(s) loaded.")
    return cases, []


def render_setup_tab() -> SetupResult:
    st.caption(f"Here you write your own test. If you only want to try the app, the tab "
               f"**{TAB_START}** fills this page in for you.")
    left, right = st.columns([1.1, 1])
    with right:
        st.markdown("#### 2 · How should the models answer?")
        schema, truth, errs1 = _schema_inputs()
        st.markdown("#### 3 · What should change between the repetitions?")
        variation, errs2 = _variation_inputs()
    with left:
        st.markdown("#### 1 · Your question")
        cases, errs3 = _case_inputs(truth)
    errors = errs1 + errs2 + errs3
    windows: list[ContextWindow] = []
    if not errors and cases and schema and variation:
        windows = build_windows(cases, variation, schema, st.session_state.n_windows,
                                st.session_state.design_seed)
    for e in errors:
        st.error(e)
    if variation and variation.mode == VariationMode.OCCASION and \
            st.session_state.temperature == 0:
        st.warning("Randomness (temperature) is set to 0 in the sidebar. Most models then "
                   "give (almost) identical answers, so they look more consistent than they "
                   "are.", icon=ICON_WARNING)

    with st.expander(f"Preview: exactly what each model will receive ({len(windows)} "
                     "conversations per model)", expanded=False):
        st.caption("Each box is one separate conversation. The lines about the answer format "
                   "were added by the app.")
        for w in windows[:12]:
            st.markdown(f"**Question `{w.case_id}` · repetition {w.window_id} · group "
                        f"`{w.variant}`**")
            st.code(f"Background instruction:\n{w.system_prompt}\n\n"
                    f"Question:\n{w.user_prompt}", language=None, wrap_lines=True)
        if len(windows) > 12:
            st.caption(f"… and {len(windows) - 12} more.")
    if not errors:
        st.caption(f"Finished here? Continue in the tab **{TAB_RUN}**.")
    return SetupResult(cases, schema, variation, truth, windows, errors)
