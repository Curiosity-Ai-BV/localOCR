"""Shared Streamlit theme helpers."""
from __future__ import annotations


def get_app_css() -> str:
    """Return the app-level CSS used by the Streamlit shell."""
    return """
    :root {
        --ocr-primary: #1E293B;
        --ocr-primary-strong: #0F172A;
        --ocr-accent: #2563EB;
        --ocr-accent-hover: #1D4ED8;
        --ocr-background: #F8FAFC;
        --ocr-surface: #FFFFFF;
        --ocr-surface-muted: #F1F5F9;
        --ocr-foreground: #0F172A;
        --ocr-muted: #475569;
        --ocr-border: #E2E8F0;
        --ocr-border-strong: #CBD5E1;
        --ocr-success: #047857;
        --ocr-warning: #B45309;
        --ocr-error: #B91C1C;
        --ocr-radius-sm: 10px;
        --ocr-radius-md: 14px;
        --ocr-radius-lg: 22px;
        --ocr-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05), 0 1px 3px rgba(15, 23, 42, 0.08);
        --ocr-shadow-md: 0 12px 28px rgba(15, 23, 42, 0.08), 0 2px 8px rgba(15, 23, 42, 0.04);
        --ocr-focus: 0 0 0 3px rgba(37, 99, 235, 0.22);
    }

    html,
    body,
    .stApp {
        background: var(--ocr-background);
        color: var(--ocr-foreground);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    .stApp {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(248, 250, 252, 0.96) 42%, #EEF2F7 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4 {
        color: var(--ocr-primary-strong);
        letter-spacing: 0;
        text-wrap: balance;
    }

    .stMarkdown p,
    .stCaptionContainer,
    [data-testid="stMarkdownContainer"] {
        text-wrap: pretty;
    }

    .stCaptionContainer,
    .ocr-result-meta,
    code,
    pre {
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.82);
        border-right: 1px solid rgba(148, 163, 184, 0.26);
        box-shadow: 8px 0 24px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 0.92rem;
        font-weight: 700;
        margin-top: 0.35rem;
        color: var(--ocr-primary);
    }

    .ocr-app-header {
        border-radius: var(--ocr-radius-lg);
        background: rgba(255, 255, 255, 0.86);
        box-shadow: var(--ocr-shadow-md);
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
    }

    .ocr-eyebrow {
        color: var(--ocr-accent);
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0 0 0.28rem;
        text-transform: uppercase;
    }

    .ocr-app-header h1 {
        font-size: clamp(1.8rem, 4vw, 2.55rem);
        line-height: 1.05;
        margin: 0;
    }

    .ocr-app-header p {
        color: var(--ocr-muted);
        margin: 0.55rem 0 0;
        max-width: 62rem;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148, 163, 184, 0.32);
        border-radius: var(--ocr-radius-md);
        box-shadow: var(--ocr-shadow-sm);
        background: rgba(255, 255, 255, 0.88);
    }

    [data-testid="stFileUploader"] section {
        transition-property: border-color, box-shadow, background-color;
        transition-duration: 180ms;
        transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(37, 99, 235, 0.42);
        box-shadow: var(--ocr-shadow-md);
    }

    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        min-height: 44px;
        border-radius: 12px;
        border: 1px solid rgba(37, 99, 235, 0.22);
        background: var(--ocr-accent);
        color: #FFFFFF;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.2);
        transition-property: background-color, border-color, box-shadow, transform;
        transition-duration: 180ms;
        transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
        touch-action: manipulation;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        background: var(--ocr-accent-hover);
        border-color: rgba(29, 78, 216, 0.44);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
        color: #FFFFFF;
    }

    .stButton > button:active,
    [data-testid="stDownloadButton"] > button:active {
        transform: scale(0.96);
    }

    .stButton > button:focus-visible,
    [data-testid="stDownloadButton"] > button:focus-visible,
    textarea:focus,
    input:focus,
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: var(--ocr-focus);
        outline: 2px solid transparent;
    }

    textarea,
    input,
    div[data-baseweb="select"] > div {
        min-height: 44px;
        border-radius: var(--ocr-radius-sm);
        transition-property: border-color, box-shadow, background-color;
        transition-duration: 160ms;
        transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
    }

    img {
        border-radius: 12px;
        outline: 1px solid rgba(0, 0, 0, 0.1);
        outline-offset: -1px;
    }

    .ocr-setup-status {
        display: grid;
        gap: 0.45rem;
        margin: 0.7rem 0 1rem;
    }

    .ocr-setup-heading {
        color: var(--ocr-muted);
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0;
        text-transform: uppercase;
    }

    .ocr-setup-row {
        align-items: center;
        background: rgba(248, 250, 252, 0.9);
        border-radius: 12px;
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.22);
        display: grid;
        gap: 0.18rem;
        grid-template-columns: 1fr auto;
        padding: 0.58rem 0.68rem;
    }

    .ocr-setup-label {
        color: var(--ocr-primary-strong);
        font-size: 0.9rem;
        font-weight: 700;
    }

    .ocr-setup-detail {
        color: var(--ocr-muted);
        font-size: 0.78rem;
        grid-column: 1 / -1;
    }

    .ocr-status-badge,
    .ocr-setup-badge {
        align-items: center;
        border-radius: 999px;
        display: inline-flex;
        font-size: 0.73rem;
        font-weight: 800;
        justify-content: center;
        line-height: 1;
        min-height: 24px;
        padding: 0 0.58rem;
        white-space: nowrap;
    }

    .ocr-status-done,
    .ocr-status-ready {
        background: rgba(4, 120, 87, 0.1);
        color: var(--ocr-success);
    }

    .ocr-status-error,
    .ocr-status-action {
        background: rgba(185, 28, 28, 0.1);
        color: var(--ocr-error);
    }

    .ocr-status-running,
    .ocr-status-check {
        background: rgba(180, 83, 9, 0.1);
        color: var(--ocr-warning);
    }

    .ocr-status-queued {
        background: rgba(71, 85, 105, 0.1);
        color: var(--ocr-muted);
    }

    .ocr-result-heading {
        align-items: center;
        display: flex;
        gap: 0.7rem;
        margin-bottom: 0.85rem;
        min-width: 0;
    }

    .ocr-result-title {
        color: var(--ocr-primary-strong);
        font-size: 1.02rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .ocr-result-meta {
        color: var(--ocr-muted);
        font-size: 0.82rem;
        margin: 0.65rem 0 0;
    }

    .ocr-empty-state {
        background: rgba(255, 255, 255, 0.86);
        border-radius: var(--ocr-radius-lg);
        box-shadow: var(--ocr-shadow-sm);
        color: var(--ocr-muted);
        padding: 1.15rem 1.25rem;
    }

    .ocr-empty-state strong {
        color: var(--ocr-primary-strong);
        display: block;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }

    .ocr-footer {
        color: var(--ocr-muted);
        font-size: 0.82rem;
        margin-top: 1.8rem;
        text-align: center;
    }

    .ocr-footer a {
        color: var(--ocr-accent);
        font-weight: 700;
        text-decoration: none;
    }

    .ocr-footer a:hover {
        text-decoration: underline;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .ocr-app-header {
            border-radius: 16px;
            padding: 1rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .stButton > button,
        [data-testid="stDownloadButton"] > button,
        [data-testid="stFileUploader"] section,
        textarea,
        input,
        div[data-baseweb="select"] > div {
            transition-duration: 0.01ms;
        }

        .stButton > button:active,
        [data-testid="stDownloadButton"] > button:active {
            transform: none;
        }
    }
    """


def render_app_theme() -> None:
    """Inject the shared Streamlit theme."""
    import streamlit as st

    st.markdown(f"<style>{get_app_css()}</style>", unsafe_allow_html=True)
