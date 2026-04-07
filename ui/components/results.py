"""Results rendering helpers for the Streamlit UI."""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


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
            status = entry.get("status", "done")
            chip = {"queued": "[queued]", "running": "[running]", "done": "[done]", "error": "[error]"}.get(status, "")
            st.markdown(f"#### {chip} {entry['filename']}")
            col1, col2 = st.columns([1, 2])
            image_bytes = entry.get("image_bytes")
            if show_images and image_bytes:
                col1.image(image_bytes, width=(180 if compact_view else 250))
            else:
                col1.empty()

            with col2:
                st.markdown(entry.get("content", ""))
                meta_bits: List[str] = []
                duration = entry.get("duration_sec")
                if isinstance(duration, (int, float)):
                    meta_bits.append(f"{float(duration):.2f} s")
                dims = entry.get("dimensions")
                if isinstance(dims, tuple) and len(dims) == 2 and all(isinstance(x, (int, float)) for x in dims):
                    meta_bits.append(f"{int(dims[0])}x{int(dims[1])} px")
                encoded_bytes = entry.get("encoded_bytes")
                if isinstance(encoded_bytes, (int, float)):
                    meta_bits.append(f"{float(encoded_bytes) / 1024:.1f} KB")
                if meta_bits:
                    st.caption(" | ".join(meta_bits))
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
            st.divider()
