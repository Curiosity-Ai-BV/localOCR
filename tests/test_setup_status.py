from types import SimpleNamespace

from core.settings import Settings
import ui.components.sidebar as sidebar
from ui.components.setup_status import (
    PDF_FIRST_PAGE_MODE,
    PDF_PER_PAGE_MODE,
    build_readiness_items,
    get_pdf_mode_help,
    get_pdf_mode_options,
)


def test_pdf_first_page_mode_copy_is_truthful():
    options = get_pdf_mode_options()

    assert options == [PDF_PER_PAGE_MODE, PDF_FIRST_PAGE_MODE]
    assert "entire" not in PDF_FIRST_PAGE_MODE.lower()
    assert "first page" in PDF_FIRST_PAGE_MODE.lower()
    assert "first page" in get_pdf_mode_help().lower()
    assert "first page" in get_pdf_mode_help(PDF_FIRST_PAGE_MODE).lower()
    assert "whole" not in get_pdf_mode_help(PDF_FIRST_PAGE_MODE).lower()


def test_readiness_items_summarize_runtime_without_long_instructions():
    items = build_readiness_items(
        ollama_available=False,
        model_names=["gemma3:12b", "deepseek-ocr"],
        selected_model="gemma4",
        pdf_supported=True,
    )

    assert [item.label for item in items] == ["Ollama", "Vision models", "PDF support"]
    assert items[0].status == "action"
    assert items[0].detail == "Not reachable"
    assert items[1].status == "ready"
    assert items[1].detail == "2 local models"
    assert items[2].status == "ready"
    assert items[2].detail == "Available"


def test_sidebar_model_inventory_keeps_empty_success_reachable(monkeypatch):
    monkeypatch.setattr(
        sidebar,
        "list_models_with_status",
        lambda *, settings: SimpleNamespace(models=[], reachable=True),
    )

    model_names, ollama_available = sidebar._load_local_model_inventory(Settings())

    assert model_names == []
    assert ollama_available is True
