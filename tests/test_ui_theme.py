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
    assert ".stButton > button *" in css
    assert '[data-testid="stFileUploaderDropzone"] button *' in css
    assert '[data-testid="stFileChips"]' in css
    assert '[data-testid="stFileChip"]' in css
    assert '[data-testid="stFileChipDeleteBtn"] button' in css
    assert 'button[aria-label^="Remove "]' in css
    assert 'button[aria-label^="Cancel upload"]' in css
    assert 'button[aria-label="Add files"]' in css
    assert '[data-testid="stFileUploaderDropzone"] button span' in css
    assert '-webkit-text-fill-color: #FFFFFF !important' in css
    assert 'button[kind="primary"]:hover' in css
    assert 'background: #0F172A !important' in css
    assert ".ocr-brand-row" in css
    assert ".ocr-brand-logo" in css
    assert ".ocr-app-description" in css
    assert "width: clamp(46px, 4vw, 56px) !important" in css
    assert "font-size: clamp(1.9rem, 3vw, 2.55rem) !important" in css
    assert "max-width: 54rem" in css
    assert "stroke: #FFFFFF !important" in css
    assert '[data-testid="stNumberInput"] input' in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "text-align: right" in css
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
