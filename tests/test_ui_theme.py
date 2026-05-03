from pathlib import Path

from ui.theme import get_app_css


def test_app_css_includes_polish_and_accessibility_guards():
    css = get_app_css()

    assert "-webkit-font-smoothing: antialiased" in css
    assert "text-wrap: balance" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "min-height: 44px" in css
    assert "scale(0.96)" in css
    assert "rgba(0, 0, 0, 0.1)" in css
    assert '[data-testid="stHeader"]' in css
    assert 'div[data-baseweb="select"] > div' in css
    assert "background: var(--ocr-surface) !important" in css
    assert '[role="listbox"]' in css
    assert 'div[data-baseweb="popover"] [role="option"]' in css
    assert '[data-testid="stExpander"] summary *' in css
    assert '[data-testid="stTextArea"] textarea' in css
    assert '[data-testid="stExpander"] summary' in css
    assert "background-color: var(--ocr-surface) !important" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_app_css_uses_specific_transitions():
    css = get_app_css()

    assert "transition: all" not in css
    assert "will-change: all" not in css
    assert "transition-property:" in css


def test_streamlit_theme_config_forces_light_widgets():
    config = Path(".streamlit/config.toml").read_text()

    assert 'base = "light"' in config
    assert 'primaryColor = "#2563EB"' in config
    assert 'secondaryBackgroundColor = "#FFFFFF"' in config
    assert 'textColor = "#0F172A"' in config
