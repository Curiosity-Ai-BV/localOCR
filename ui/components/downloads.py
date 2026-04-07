"""Download-button component wrapping core.export."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

import streamlit as st

from core.export import render_results
from core.models import Result


def render_downloads(results: Iterable[Result]) -> None:
    items: List[Result] = [r for r in results if isinstance(r, Result)]
    if not items:
        return
    st.subheader("Export")
    col1, col2, col3 = st.columns(3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        st.download_button(
            "Results (CSV)",
            data=render_results(items, format="csv"),
            file_name=f"image_analysis_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    struct = [r for r in items if r.fields]
    if struct:
        with col2:
            st.download_button(
                "Structured (CSV)",
                data=render_results(struct, format="structured_csv"),
                file_name=f"structured_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col3:
        st.download_button(
            "Results (JSONL)",
            data=render_results(items, format="jsonl"),
            file_name=f"image_analysis_{timestamp}.jsonl",
            mime="application/json",
            use_container_width=True,
        )
