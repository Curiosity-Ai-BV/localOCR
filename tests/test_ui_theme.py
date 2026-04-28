from ui.theme import get_app_css


def test_app_css_includes_polish_and_accessibility_guards():
    css = get_app_css()

    assert "-webkit-font-smoothing: antialiased" in css
    assert "text-wrap: balance" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "min-height: 44px" in css
    assert "scale(0.96)" in css
    assert "rgba(0, 0, 0, 0.1)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_app_css_uses_specific_transitions():
    css = get_app_css()

    assert "transition: all" not in css
    assert "will-change: all" not in css
    assert "transition-property:" in css
