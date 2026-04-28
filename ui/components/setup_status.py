"""Compact setup status helpers for the Streamlit UI."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable, List

import streamlit as st

PDF_PER_PAGE_MODE = "Process each page separately"
PDF_FIRST_PAGE_MODE = "Use first page only"


@dataclass(frozen=True)
class ReadinessItem:
    label: str
    status: str
    detail: str


def get_pdf_mode_options() -> List[str]:
    return [PDF_PER_PAGE_MODE, PDF_FIRST_PAGE_MODE]


def get_pdf_mode_help(selected_mode: str | None = None) -> str:
    if selected_mode == PDF_FIRST_PAGE_MODE:
        return "Runs OCR on the first page only. Use per-page mode when every page matters."
    return "Per-page runs OCR once per page. First-page mode scans only the first page."


def build_readiness_items(
    *,
    ollama_available: bool,
    model_names: Iterable[str],
    selected_model: str,
    pdf_supported: bool,
) -> List[ReadinessItem]:
    local_models = [name for name in model_names if str(name).strip()]
    model_count = len(local_models)
    if ollama_available:
        ollama = ReadinessItem("Ollama", "ready", "Reachable")
    else:
        ollama = ReadinessItem("Ollama", "action", "Not reachable")

    if model_count == 0:
        models = ReadinessItem("Vision models", "action", "No local models found")
    else:
        suffix = "model" if model_count == 1 else "models"
        models = ReadinessItem("Vision models", "ready", f"{model_count} local {suffix}")

    pdf = ReadinessItem(
        "PDF support",
        "ready" if pdf_supported else "action",
        "Available" if pdf_supported else "Install PyMuPDF",
    )
    return [ollama, models, pdf]


def render_setup_status(items: Iterable[ReadinessItem]) -> None:
    labels = {"ready": "Ready", "check": "Check", "action": "Action"}
    rows = ['<div class="ocr-setup-status" aria-label="Setup check">']
    rows.append('<p class="ocr-setup-heading">Setup check</p>')
    for item in items:
        badge = labels.get(item.status, "Check")
        status_class = f"ocr-status-{escape(item.status)}"
        rows.append(
            (
                '<div class="ocr-setup-row">'
                f'<span class="ocr-setup-label">{escape(item.label)}</span>'
                f'<span class="ocr-setup-badge {status_class}">{escape(badge)}</span>'
                f'<span class="ocr-setup-detail">{escape(item.detail)}</span>'
                "</div>"
            )
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
