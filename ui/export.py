from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import List

import streamlit as st

from utils.typing import JSONDict


def create_download_buttons(results: List[JSONDict], structured_results: List[JSONDict], extraction_mode: str) -> None:
    st.subheader("Export")

    col1, col2 = st.columns(2)

    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(["Filename", "Description"])
    for result in results:
        csv_writer.writerow([result["filename"], result.get("description", result.get("extraction", ""))])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"image_analysis_{timestamp}.csv"

    with col1:
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True,
        )

    if structured_results:
        all_fields = set(["filename"])
        for result in structured_results:
            all_fields.update(result.keys())

        field_list = sorted(list(all_fields))
        structured_csv = io.StringIO()
        structured_writer = csv.writer(structured_csv)
        structured_writer.writerow(field_list)

        for result in structured_results:
            row = [result.get(field, "") for field in field_list]
            structured_writer.writerow(row)

        structured_filename = f"structured_data_{timestamp}.csv"

        with col2:
            st.download_button(
                label="📥 Download Structured (CSV)",
                data=structured_csv.getvalue(),
                file_name=structured_filename,
                mime="text/csv",
                use_container_width=True,
            )

