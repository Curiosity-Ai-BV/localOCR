"""Streamlit download buttons that delegate to :mod:`core.export`."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, List, Sequence

import streamlit as st

from core.export import render_results
from core.models import Result


def _coerce_to_results(entries: Sequence[Any]) -> List[Result]:
    out: List[Result] = []
    for e in entries:
        if isinstance(e, Result):
            out.append(e)
            continue
        if isinstance(e, dict):
            # Legacy dict shape from older call sites.
            text = e.get("description") or e.get("extraction") or ""
            mode = "describe" if "description" in e and "extraction" not in e else "extract"
            out.append(
                Result(
                    source=str(e.get("filename", "")),
                    mode=mode,  # type: ignore[arg-type]
                    text=str(text),
                    raw=str(text),
                    fields={k: v for k, v in e.items() if k not in {"filename", "description", "extraction", "duration_sec", "input_width", "input_height", "encoded_bytes", "error"}},
                )
            )
    return out


def _coerce_structured(entries: Sequence[Any]) -> List[Result]:
    out: List[Result] = []
    for e in entries:
        if isinstance(e, Result):
            if e.fields:
                out.append(e)
            continue
        if isinstance(e, dict):
            fname = str(e.get("filename", ""))
            fields = {k: v for k, v in e.items() if k != "filename"}
            out.append(Result(source=fname, mode="extract", fields=fields))
    return out


def create_download_buttons(
    results: Iterable[Any],
    structured_results: Iterable[Any],
    extraction_mode: str,
) -> None:
    st.subheader("Export")
    col1, col2 = st.columns(2)

    res_list = _coerce_to_results(list(results))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_payload = render_results(res_list, format="csv")
    with col1:
        st.download_button(
            label="Download Results (CSV)",
            data=csv_payload,
            file_name=f"image_analysis_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    struct_list = _coerce_structured(list(structured_results))
    if struct_list:
        payload = render_results(struct_list, format="structured_csv")
        with col2:
            st.download_button(
                label="Download Structured (CSV)",
                data=payload,
                file_name=f"structured_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
