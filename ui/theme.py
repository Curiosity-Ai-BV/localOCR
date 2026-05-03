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
        --ocr-background-wash: #EFF4FA;
        --ocr-surface: #FFFFFF;
        --ocr-surface-muted: #F1F5F9;
        --ocr-surface-raised: rgba(255, 255, 255, 0.94);
        --ocr-foreground: #0F172A;
        --ocr-muted: #475569;
        --ocr-muted-soft: #64748B;
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
        background: var(--ocr-background) !important;
        color: var(--ocr-foreground) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    .stApp {
        background:
            linear-gradient(180deg, #FFFFFF 0%, var(--ocr-background) 44%, var(--ocr-background-wash) 100%) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.92);
        border-bottom: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        color: var(--ocr-muted);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
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
        background: var(--ocr-surface) !important;
        border-right: 1px solid var(--ocr-border);
        box-shadow: 6px 0 18px rgba(15, 23, 42, 0.035);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] .stCaptionContainer,
    [data-testid="stSidebar"] [data-testid="InputInstructions"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--ocr-muted) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: var(--ocr-primary) !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-top: 0.85rem;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2:first-child,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3:first-child {
        margin-top: 0;
    }

    [data-testid="stWidgetLabel"] p {
        color: var(--ocr-muted) !important;
        font-size: 0.82rem;
        font-weight: 700;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: var(--ocr-surface-muted) !important;
        border: 1px solid var(--ocr-border);
        color: var(--ocr-muted) !important;
        border-radius: var(--ocr-radius-md);
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: var(--ocr-primary) !important;
    }

    .ocr-app-header {
        background: var(--ocr-surface-raised);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 16px;
        box-shadow: var(--ocr-shadow-sm);
        margin-bottom: 1rem;
        padding: 1.05rem 1.2rem;
    }

    .ocr-brand-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        min-width: 0;
    }

    .ocr-brand-logo {
        width: clamp(46px, 4vw, 56px) !important;
        height: clamp(46px, 4vw, 56px) !important;
        flex: 0 0 auto;
        border-radius: 13px;
        max-width: 56px !important;
        object-fit: contain;
    }

    .ocr-brand-copy {
        min-width: 0;
    }

    .ocr-eyebrow {
        color: var(--ocr-muted-soft);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        line-height: 1.2;
        margin: 0 0 0.22rem;
        text-transform: uppercase;
    }

    .ocr-app-header h1 {
        color: var(--ocr-primary-strong);
        font-size: clamp(1.9rem, 3vw, 2.55rem) !important;
        font-weight: 800;
        line-height: 1.02;
        margin: 0;
    }

    .ocr-app-description {
        color: var(--ocr-muted);
        font-size: 0.98rem;
        line-height: 1.45;
        margin: 0.38rem 0 0;
        max-width: 54rem;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--ocr-border);
        border-radius: var(--ocr-radius-md);
        box-shadow: var(--ocr-shadow-sm);
        background: var(--ocr-surface-raised) !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--ocr-surface) !important;
        transition-property: border-color, box-shadow, background-color;
        transition-duration: 180ms;
        transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
    }

    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: var(--ocr-muted) !important;
    }

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary > div,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        color: var(--ocr-foreground) !important;
    }

    [data-testid="stExpander"] summary {
        min-height: 44px;
    }

    [data-testid="stExpander"] summary * {
        background-color: transparent !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] svg {
        color: var(--ocr-primary) !important;
        fill: var(--ocr-primary) !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(37, 99, 235, 0.42);
        box-shadow: var(--ocr-shadow-md);
    }

    .stButton > button,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stFileUploaderDropzone"] button {
        min-height: 44px;
        border-radius: 12px;
        border: 1px solid rgba(37, 99, 235, 0.22);
        background: var(--ocr-accent) !important;
        color: #FFFFFF !important;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.2);
        transition-property: background-color, border-color, box-shadow, transform;
        transition-duration: 180ms;
        transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
        touch-action: manipulation;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--ocr-accent-hover) !important;
        border-color: rgba(29, 78, 216, 0.44);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
        color: #FFFFFF !important;
    }

    .stButton > button *,
    .stButton > button:hover *,
    .stButton > button:active *,
    [data-testid="stDownloadButton"] > button *,
    [data-testid="stDownloadButton"] > button:hover *,
    [data-testid="stDownloadButton"] > button:active *,
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button:hover *,
    [data-testid="stFileUploaderDropzone"] button:active * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    .stButton > button p,
    [data-testid="stDownloadButton"] > button p,
    [data-testid="stFileUploaderDropzone"] button p {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stFileUploaderDropzone"] button div,
    [data-testid="stFileUploaderDropzone"] button svg,
    [data-testid="stFileUploaderDropzone"] button svg * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    .stButton > button:active,
    [data-testid="stDownloadButton"] > button:active,
    [data-testid="stFileUploaderDropzone"] button:active {
        transform: scale(0.96);
    }

    .stButton > button:focus-visible,
    [data-testid="stDownloadButton"] > button:focus-visible,
    [data-testid="stFileUploaderDropzone"] button:focus-visible,
    textarea:focus,
    input:focus,
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: var(--ocr-focus);
        outline: 2px solid transparent;
    }

    [data-testid="stFileChips"] {
        display: grid !important;
        gap: 0.45rem !important;
        grid-template-columns: minmax(0, 1fr);
        line-height: 1.2;
        max-height: 13rem;
        overflow-y: auto;
        padding-right: 0.1rem;
        width: 100%;
    }

    [data-testid="stFileChips"] > div,
    [data-testid="stFileChips"] > div > div {
        max-width: 100% !important;
        width: 100% !important;
    }

    [data-testid="stFileChip"] {
        align-items: center !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1px solid var(--ocr-border) !important;
        border-radius: 11px !important;
        box-shadow: none !important;
        display: grid !important;
        gap: 0.5rem !important;
        grid-template-columns: 30px minmax(0, 1fr) 28px;
        min-width: 0 !important;
        padding: 0.38rem 0.42rem !important;
        width: 100% !important;
    }

    [data-testid="stFileChip"] > div:first-child {
        background: var(--ocr-primary-strong) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        height: 30px !important;
        width: 30px !important;
    }

    [data-testid="stFileChip"] > div:first-child svg,
    [data-testid="stFileChip"] > div:first-child svg * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    [data-testid="stFileChipName"] {
        color: var(--ocr-primary-strong) !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    [data-testid="stFileChipName"] + div {
        color: var(--ocr-muted) !important;
        font-size: 0.78rem !important;
        font-variant-numeric: tabular-nums;
        line-height: 1.1 !important;
    }

    [data-testid="stFileChipDeleteBtn"] {
        align-self: center !important;
        justify-self: end !important;
    }

    [data-testid="stFileChipDeleteBtn"] button,
    [data-testid="stFileUploader"] button[aria-label^="Remove "],
    [data-testid="stFileUploader"] button[aria-label^="Cancel upload"],
    [data-testid="stFileChips"] button[aria-label="Add files"] {
        align-items: center !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: 1px solid transparent !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        color: var(--ocr-muted) !important;
        display: inline-flex !important;
        height: 28px !important;
        justify-content: center !important;
        line-height: 1 !important;
        min-height: 28px !important;
        min-width: 28px !important;
        padding: 0 !important;
        width: 28px !important;
    }

    [data-testid="stFileChips"] button[aria-label="Add files"] {
        background: rgba(37, 99, 235, 0.1) !important;
        background-color: rgba(37, 99, 235, 0.1) !important;
        border-color: rgba(37, 99, 235, 0.18) !important;
        color: var(--ocr-accent) !important;
        margin-top: 0.1rem;
        width: 34px !important;
        height: 34px !important;
    }

    [data-testid="stFileChipDeleteBtn"] button *,
    [data-testid="stFileUploader"] button[aria-label^="Remove "] *,
    [data-testid="stFileUploader"] button[aria-label^="Cancel upload"] *,
    [data-testid="stFileChips"] button[aria-label="Add files"] * {
        color: currentColor !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        background: transparent !important;
        background-color: transparent !important;
    }

    [data-testid="stFileChipDeleteBtn"] button:hover,
    [data-testid="stFileUploader"] button[aria-label^="Remove "]:hover,
    [data-testid="stFileUploader"] button[aria-label^="Cancel upload"]:hover {
        background: rgba(185, 28, 28, 0.08) !important;
        background-color: rgba(185, 28, 28, 0.08) !important;
        color: var(--ocr-error) !important;
    }

    [data-testid="stFileChips"] button[aria-label="Add files"]:hover {
        background: rgba(37, 99, 235, 0.14) !important;
        color: var(--ocr-accent-hover) !important;
    }

    textarea,
    input,
    div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    div[data-baseweb="textarea"],
    div[data-baseweb="input"] {
        background: var(--ocr-surface) !important;
        background-color: var(--ocr-surface) !important;
        border-color: var(--ocr-border) !important;
        color: var(--ocr-foreground) !important;
        min-height: 44px;
        border-radius: var(--ocr-radius-sm);
        transition-property: border-color, box-shadow, background-color;
        transition-duration: 160ms;
        transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
    }

    textarea:hover,
    input:hover,
    div[data-baseweb="select"] > div:hover,
    [data-testid="stNumberInput"] input:hover,
    div[data-baseweb="textarea"]:hover,
    div[data-baseweb="input"]:hover {
        border-color: var(--ocr-border-strong) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: var(--ocr-muted-soft) !important;
        opacity: 1 !important;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input {
        background: var(--ocr-surface) !important;
        background-color: var(--ocr-surface) !important;
        color: var(--ocr-foreground) !important;
        caret-color: var(--ocr-accent) !important;
    }

    [data-testid="stNumberInput"] {
        margin-bottom: 0.1rem;
    }

    [data-testid="stNumberInput"] input {
        font-variant-numeric: tabular-nums;
        text-align: right;
    }

    [data-testid="stNumberInput"] button {
        background: var(--ocr-surface-muted) !important;
        border-color: var(--ocr-border) !important;
        color: var(--ocr-primary) !important;
        min-height: 36px;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button * {
        color: var(--ocr-primary) !important;
        fill: var(--ocr-primary) !important;
        stroke: var(--ocr-primary) !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color: var(--ocr-foreground) !important;
        fill: var(--ocr-muted) !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [role="listbox"] {
        background: var(--ocr-surface) !important;
        background-color: var(--ocr-surface) !important;
        border: 1px solid var(--ocr-border) !important;
        border-radius: var(--ocr-radius-md);
        box-shadow: var(--ocr-shadow-md);
        color: var(--ocr-foreground) !important;
    }

    div[data-baseweb="popover"] div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] div,
    [role="listbox"] div,
    [role="option"] {
        background: var(--ocr-surface) !important;
        background-color: var(--ocr-surface) !important;
        color: var(--ocr-foreground) !important;
    }

    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] div:hover,
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [aria-selected="true"],
    [role="option"]:hover,
    [aria-selected="true"] {
        background: var(--ocr-surface-muted) !important;
        background-color: var(--ocr-surface-muted) !important;
        color: var(--ocr-primary-strong) !important;
    }

    [data-baseweb="slider"] [role="slider"] {
        box-shadow: var(--ocr-shadow-sm);
    }

    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: var(--ocr-foreground) !important;
        min-height: 32px;
    }

    [data-testid="stCheckbox"] input,
    [data-testid="stRadio"] input {
        accent-color: var(--ocr-accent);
    }

    [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div,
    [data-testid="stRadio"] [data-baseweb="radio"] > div {
        border-color: var(--ocr-border-strong) !important;
        background-color: var(--ocr-surface) !important;
    }

    [data-testid="stCheckbox"] [aria-checked="true"] > div,
    [data-testid="stRadio"] [aria-checked="true"] > div {
        border-color: var(--ocr-accent) !important;
        background-color: var(--ocr-accent) !important;
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
        letter-spacing: 0.04em;
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
        background: var(--ocr-surface-raised);
        border: 1px solid var(--ocr-border);
        border-radius: var(--ocr-radius-md);
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

    /* Final button foreground pass: Streamlit injects nested markdown labels inside buttons. */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button[kind],
    .stButton > button[kind="primary"],
    .stButton > button[kind="primaryFormSubmit"],
    button[kind="primary"] {
        background: var(--ocr-accent) !important;
        border-color: rgba(37, 99, 235, 0.36) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploaderDropzone"] button[kind]:hover,
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primaryFormSubmit"]:hover,
    button[kind="primary"]:hover {
        background: #0F172A !important;
        border-color: #0F172A !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stFileUploaderDropzone"] button :is(p, span, div),
    [data-testid="stFileUploaderDropzone"] button[kind] :is(p, span, div),
    .stButton > button[kind="primary"] :is(p, span, div),
    .stButton > button[kind="primaryFormSubmit"] :is(p, span, div),
    button[kind="primary"] :is(p, span, div) {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stFileUploaderDropzone"] button svg,
    [data-testid="stFileUploaderDropzone"] button svg *,
    [data-testid="stFileUploaderDropzone"] button[kind] svg,
    [data-testid="stFileUploaderDropzone"] button[kind] svg *,
    .stButton > button[kind="primary"] svg,
    .stButton > button[kind="primary"] svg *,
    .stButton > button[kind="primaryFormSubmit"] svg,
    .stButton > button[kind="primaryFormSubmit"] svg *,
    button[kind="primary"] svg,
    button[kind="primary"] svg * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .ocr-app-header {
            border-radius: 16px;
            padding: 0.9rem;
        }

        .ocr-brand-row {
            align-items: flex-start;
            gap: 0.7rem;
        }

        .ocr-brand-logo {
            width: 42px !important;
            height: 42px !important;
            border-radius: 12px;
        }

        .ocr-app-header h1 {
            font-size: clamp(1.65rem, 8vw, 2rem) !important;
        }

        .ocr-app-description {
            font-size: 0.92rem;
            line-height: 1.45;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .stButton > button,
        [data-testid="stDownloadButton"] > button,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileChips"] button,
        [data-testid="stFileUploader"] section,
        textarea,
        input,
        [data-testid="stNumberInput"] input,
        [data-testid="stNumberInput"] button,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"],
        div[data-baseweb="input"] {
            transition-duration: 0.01ms;
        }

        .stButton > button:active,
        [data-testid="stDownloadButton"] > button:active,
        [data-testid="stFileUploaderDropzone"] button:active,
        [data-testid="stFileChips"] button:active {
            transform: none;
        }
    }
    """


def render_app_theme() -> None:
    """Inject the shared Streamlit theme."""
    import streamlit as st

    st.markdown(f"<style>{get_app_css()}</style>", unsafe_allow_html=True)
