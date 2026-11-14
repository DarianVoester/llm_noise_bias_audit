"""Theme config: native light + dark mode with readable contrast (WCAG 2.x)."""

import tomllib
from pathlib import Path

from ui.tab_results import CALM_PALETTE, soften_figure

CONFIG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted([_luminance(a), _luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _theme():
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["theme"]


def test_both_modes_defined_and_base_not_pinned():
    theme = _theme()
    assert "base" not in theme, "setting [theme] base would pin one mode"
    assert {"light", "dark"} <= set(theme)


def test_body_text_contrast_is_high_in_both_modes():
    theme = _theme()
    for mode in ("light", "dark"):
        t = theme[mode]
        assert contrast(t["textColor"], t["backgroundColor"]) >= 15, mode  # WCAG AAA is 7
        assert contrast(t["textColor"], t["secondaryBackgroundColor"]) >= 10, mode


def test_status_box_text_is_readable():
    theme = _theme()
    for mode in ("light", "dark"):
        t = theme[mode]
        for colour in ("red", "green", "yellow", "blue", "orange"):
            ratio = contrast(t[f"{colour}TextColor"], t[f"{colour}BackgroundColor"])
            assert ratio >= 7, (mode, colour, ratio)  # WCAG AAA for normal text


def test_controls_and_links_stand_out():
    theme = _theme()
    for mode in ("light", "dark"):
        t = theme[mode]
        bg = t["backgroundColor"]
        assert contrast(t["primaryColor"], "#FFFFFF") >= 4.5, mode  # white text on buttons
        assert contrast(t["primaryColor"], bg) >= 3, mode  # sliders, focus rings
        assert contrast(t["linkColor"], bg) >= 7, mode
        assert contrast(t["borderColor"], bg) >= 3, mode  # widget outlines
        assert contrast(t["borderColor"], t["secondaryBackgroundColor"]) >= 3, mode


def test_chart_colours_visible_on_both_backgrounds():
    theme = _theme()
    for mode in ("light", "dark"):
        bg = theme[mode]["backgroundColor"]
        for colour in CALM_PALETTE:
            assert contrast(colour, bg) >= 3, (mode, colour)  # 3:1 for graphics
    assert theme["light"]["chartCategoricalColors"] == CALM_PALETTE


def test_soften_figure_recolours_traces():
    import plotly.express as px

    fig = px.bar(x=["a", "b"], y=[1, 2], color=["g1", "g2"])
    soften_figure(fig)
    assert {t.marker.color for t in fig.data} <= set(CALM_PALETTE)
