"""Results rendering helpers for the Streamlit UI."""
from __future__ import annotations

from html import escape
from typing import Any, Dict, List

import streamlit as st

STATUS_BADGES = {
    "queued": ("Queued", "ocr-status-queued"),
    "running": ("Running", "ocr-status-running"),
    "done": ("Done", "ocr-status-done"),
    "error": ("Error", "ocr-status-error"),
}


def get_status_badge(status: str) -> tuple[str, str]:
    """Return the display label and CSS class for a result status."""
    return STATUS_BADGES.get(str(status).lower(), ("Check", "ocr-status-check"))


def format_result_metadata(entry: Dict[str, Any]) -> List[str]:
    """Format stable result metadata for display."""
    meta_bits: List[str] = []
    duration = entry.get("duration_sec")
    if isinstance(duration, (int, float)):
        meta_bits.append(f"{float(duration):.2f} s")

    dims = entry.get("dimensions")
    if (
        isinstance(dims, tuple)
        and len(dims) == 2
        and all(isinstance(x, (int, float)) for x in dims)
    ):
        meta_bits.append(f"{int(dims[0])}x{int(dims[1])} px")

    encoded_bytes = entry.get("encoded_bytes")
    if isinstance(encoded_bytes, (int, float)):
        meta_bits.append(f"{float(encoded_bytes) / 1024:.1f} KB")
    return meta_bits


def _render_result_heading(entry: Dict[str, Any]) -> None:
    label, class_name = get_status_badge(str(entry.get("status", "done")))
    filename = str(entry.get("filename") or "Untitled file")
    st.markdown(
        (
            '<div class="ocr-result-heading">'
            f'<span class="ocr-status-badge {class_name}" aria-label="Status: {escape(label)}">'
            f"{escape(label)}</span>"
            f'<span class="ocr-result-title">{escape(filename)}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_results(
    placeholder,
    entries: List[Dict[str, Any]],
    show_images: bool,
    compact_view: bool,
) -> None:
    """Render stored OCR results inside the provided placeholder."""
    with placeholder.container():
        st.subheader("Results")
        if not entries:
            st.info("No results yet. Upload files and run a scan.")
            return

        for entry in entries:
            with st.container(border=True):
                _render_result_heading(entry)
                col1, col2 = st.columns([1, 2])
                image_bytes = entry.get("image_bytes")
                if show_images and image_bytes:
                    col1.image(image_bytes, width=(180 if compact_view else 250))
                else:
                    col1.empty()

                with col2:
                    content = str(entry.get("content") or "")
                    if content:
                        st.markdown(content)
                    elif not entry.get("error"):
                        st.caption("No text returned.")

                    meta_bits = format_result_metadata(entry)
                    if meta_bits:
                        st.markdown(
                            f'<p class="ocr-result-meta">{escape(" | ".join(meta_bits))}</p>',
                            unsafe_allow_html=True,
                        )

                page_note = entry.get("page_note")
                if isinstance(page_note, str) and page_note.strip():
                    st.info(page_note.strip())
                structured = entry.get("structured_data")
                if isinstance(structured, dict) and len(structured) > 1:
                    with st.expander("Structured JSON"):
                        st.json(structured)
                err = entry.get("error")
                if err:
                    st.error(err)
