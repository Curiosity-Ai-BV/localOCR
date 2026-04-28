from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

from adapters.ollama_adapter import resolve_model_name
from core.models import Result
from core.pdf_utils import get_pdf_page_count, is_pdf_supported
from core.pipeline import BatchConfig, BatchJob, run_batch
from core.settings import Settings
from ui.components import (
    SidebarState,
    render_downloads,
    render_results,
    render_sidebar,
)
from ui.components.setup_status import PDF_PER_PAGE_MODE

st.set_page_config(
    page_title="Curiosity AI Scans",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --accent: #6C5CE7; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.15); }
    .stButton>button {
        background: var(--accent); color: #fff; border: 1px solid rgba(0,0,0,0.05);
        border-radius: 10px; padding: 0.5rem 0.9rem; font-weight: 600;
    }
    .stButton>button:hover { opacity: .95; }
    .download-panel {
        border: 1px dashed rgba(127,127,127,0.25); border-radius: 12px;
        padding: 0.9rem 1rem; background: rgba(127,127,127,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_TITLE = "Curiosity AI Scans"
SETTINGS = Settings.from_env()
PDF_SUPPORT = is_pdf_supported()


def _session_key(suffix: str) -> str:
    """Scope session state keys to the current Streamlit session id.

    Prevents results bleeding between users on a shared deployment.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        sid = getattr(ctx, "session_id", "local") if ctx else "local"
    except Exception:
        sid = "local"
    return f"{sid}:{suffix}"


def _get_state(key: str, default):
    k = _session_key(key)
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]


def _set_state(key: str, value) -> None:
    st.session_state[_session_key(key)] = value


def _jobs_from_uploads(uploaded_files) -> List[BatchJob]:
    jobs: List[BatchJob] = []
    for uf in uploaded_files:
        data = uf.read()
        uf.seek(0)
        if uf.name.lower().endswith(".pdf"):
            jobs.append(BatchJob(source=uf.name, data=data, kind="pdf"))
        else:
            img = Image.open(uf)
            img.load()
            jobs.append(BatchJob(source=uf.name, data=img, kind="image"))
    return jobs


def _count_items(uploaded_files, pdf_process_mode: str) -> int:
    total = 0
    for uf in uploaded_files:
        file_bytes = uf.read()
        uf.seek(0)
        if uf.name.lower().endswith(".pdf") and PDF_SUPPORT:
            if pdf_process_mode == PDF_PER_PAGE_MODE:
                try:
                    total += get_pdf_page_count(file_bytes)
                except Exception:
                    total += 1
            else:
                total += 1
        else:
            total += 1
    return total


def run_processing(state: SidebarState, results_placeholder) -> None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    _set_state("results", [])
    _set_state("display_entries", [])

    available, resolved_model, note = resolve_model_name(state.selected_model, settings=state.settings)
    if note:
        st.warning(note)
    if not available:
        st.error(
            f"Model '{state.selected_model}' is not available locally. "
            f"Run: ollama pull {state.selected_model}"
        )
        return

    total_items = _count_items(state.uploaded_files, state.pdf_process_mode)
    processed = 0
    durations: List[float] = []
    batch_start = time.perf_counter()

    cfg = BatchConfig(
        model=resolved_model,
        fields=state.fields,
        system_prompt=state.system_prompt,
        options=state.options,
        settings=state.settings,
        prompts=state.prompts,
        pdf_pages_separately=(state.pdf_process_mode == PDF_PER_PAGE_MODE),
    )

    results_list: List[Result] = _get_state("results", [])
    entries: List[Dict[str, Any]] = _get_state("display_entries", [])

    for uf in state.uploaded_files:
        file_bytes = uf.read()
        uf.seek(0)
        if uf.name.lower().endswith(".pdf"):
            if not PDF_SUPPORT:
                st.error(f"Cannot process PDF {uf.name}. Install PyMuPDF.")
                processed += 1
                progress_bar.progress(min(processed / max(total_items, 1), 1.0))
                continue
            job = BatchJob(source=uf.name, data=file_bytes, kind="pdf")
        else:
            img = Image.open(uf)
            img.load()
            job = BatchJob(source=uf.name, data=img, kind="image")

        for result in run_batch([job], cfg):
            status_text.text(f"Processing {result.source}")
            results_list.append(result)
            if result.latency_ms:
                durations.append(result.latency_ms / 1000.0)

            entry: Dict[str, Any] = {
                "filename": result.source,
                "content": result.text if not result.error else "",
                "duration_sec": round(result.latency_ms / 1000.0, 3) if result.latency_ms else None,
                "dimensions": result.dimensions,
                "encoded_bytes": result.encoded_bytes,
                "structured_data": ({"filename": result.source, **result.fields} if result.fields else None),
                "image_bytes": result.preview_image_bytes,
                "page_note": None,
                "status": "error" if result.error else "done",
                "error": result.error,
            }
            entries.append(entry)
            render_results(results_placeholder, entries, state.show_images, state.compact_view)

            processed += 1
            progress_bar.progress(min(processed / max(total_items, 1), 1.0))

    _set_state("results", results_list)
    _set_state("display_entries", entries)

    batch_elapsed = time.perf_counter() - batch_start
    status_text.text("Processing complete!")
    if durations:
        avg = sum(durations) / max(len(durations), 1)
        st.info(f"Total processing time: {batch_elapsed:.2f} s  |  Avg per item: {avg:.2f} s")


# --- Main ------------------------------------------------------------------

st.title(APP_TITLE)
st.caption("Local, private, minimalist vision scanning")

if not PDF_SUPPORT:
    st.warning("PDF support requires PyMuPDF. Install it with: pip install pymupdf")

_get_state("results", [])
_get_state("display_entries", [])

state = render_sidebar(SETTINGS, pdf_supported=PDF_SUPPORT)

results_placeholder = st.empty()

if state.uploaded_files and state.process_button:
    run_processing(state, results_placeholder)
else:
    render_results(
        results_placeholder,
        _get_state("display_entries", []),
        state.show_images,
        state.compact_view,
    )

results_list: List[Result] = _get_state("results", [])
if results_list:
    render_downloads(results_list)

if not state.uploaded_files and not results_list:
    st.info("Add files on the left to get started")
    st.caption("Upload images or PDFs, choose a local vision model, then run a scan.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; opacity: 0.7;">
        Made with love by Adrian - <a href="https://ad1x.com" target="_blank">ad1x.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
