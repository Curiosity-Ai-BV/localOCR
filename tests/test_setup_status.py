import builtins
import importlib
import sys
from types import SimpleNamespace

from core.settings import Settings
import ui.components.sidebar as sidebar
from ui.components.sidebar import (
    BACKEND_OPTIONS,
    PREPROCESS_OPTIONS,
    PREPROCESS_PROFILE_DEFAULT,
    PROFILE_OPTIONS,
    SidebarState,
)
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
    monkeypatch.setitem(
        sys.modules,
        "adapters.ollama_adapter",
        SimpleNamespace(
            list_models_with_status=lambda *, settings: SimpleNamespace(
                models=[],
                reachable=True,
            )
        ),
    )

    model_names, ollama_available = sidebar._load_local_model_inventory(Settings())

    assert model_names == []
    assert ollama_available is True


class _SidebarContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _FakeSidebarStreamlit:
    sidebar = _SidebarContext()

    def __init__(self):
        self.captions = []

    def subheader(self, _label):
        return None

    def file_uploader(self, *_args, **_kwargs):
        return []

    def selectbox(self, label, options, **_kwargs):
        if label == "Backend mode":
            return "docling"
        return list(options)[0]

    def caption(self, value):
        self.captions.append(value)

    def expander(self, *_args, **_kwargs):
        return _SidebarContext()

    def text_area(self, *_args, value="", **_kwargs):
        return value

    def slider(self, *_args, **_kwargs):
        return _args[3]

    def checkbox(self, *_args, value=False, **_kwargs):
        return value

    def info(self, _value):
        return None


def test_docling_sidebar_render_skips_ollama_adapter_import_and_inventory(monkeypatch):
    fake_st = _FakeSidebarStreamlit()
    original_import = builtins.__import__

    def fail_ollama_import(name, *args, **kwargs):
        if name == "adapters.ollama_adapter":
            raise AssertionError("docling sidebar should not import Ollama adapter")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "adapters.ollama_adapter", raising=False)
    monkeypatch.delitem(sys.modules, "ui.components.sidebar", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_ollama_import)

    fresh_sidebar = importlib.import_module("ui.components.sidebar")
    assert "adapters.ollama_adapter" not in sys.modules

    monkeypatch.setattr(
        fresh_sidebar,
        "_load_local_model_inventory",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("docling sidebar should not load Ollama inventory")
        ),
    )
    monkeypatch.setattr(
        fresh_sidebar,
        "_available_model_options",
        lambda _models, _settings: (_ for _ in ()).throw(
            AssertionError("docling sidebar should not resolve Ollama model options")
        ),
    )
    monkeypatch.setattr(fresh_sidebar, "st", fake_st)

    state = fresh_sidebar.render_sidebar(Settings())

    assert state.ocr_backend == "docling"
    assert state.selected_model
    assert "Docling mode skips Ollama model checks." in fake_st.captions
    assert "adapters.ollama_adapter" not in sys.modules


def test_sidebar_processing_state_defaults_and_options():
    state = SidebarState()

    assert state.profile_id == "generic"
    assert state.ocr_backend == "ollama"
    assert state.preprocess is None
    assert PROFILE_OPTIONS == ("generic", "invoice", "receipt", "table")
    assert BACKEND_OPTIONS == ("ollama", "docling", "hybrid", "auto")
    assert PREPROCESS_OPTIONS == (
        PREPROCESS_PROFILE_DEFAULT,
        "none",
        "document-clean",
        "high-accuracy-scan",
    )


class _InvoiceProfileDefaultStreamlit(_FakeSidebarStreamlit):
    def selectbox(self, label, options, **_kwargs):
        if label == "Profile":
            return "invoice"
        if label == "Backend mode":
            return "docling"
        if label == "Preprocessing":
            return PREPROCESS_PROFILE_DEFAULT
        return list(options)[0]


def test_sidebar_profile_default_preprocess_routes_as_omitted(monkeypatch):
    fake_st = _InvoiceProfileDefaultStreamlit()
    monkeypatch.setattr(sidebar, "st", fake_st)

    state = sidebar.render_sidebar(Settings())

    assert state.profile_id == "invoice"
    assert state.preprocess is None


class _InvoiceCustomExtractionStreamlit(_InvoiceProfileDefaultStreamlit):
    def __init__(self):
        super().__init__()
        self.text_area_values = {}

    def file_uploader(self, *_args, **_kwargs):
        return [SimpleNamespace(name="invoice.png")]

    def text_area(self, label, *args, value="", **kwargs):
        self.text_area_values[label] = value
        return value

    def radio(self, label, options, **_kwargs):
        if label == "Extraction mode":
            return "Custom field extraction"
        return list(options)[0]

    def button(self, *_args, **_kwargs):
        return False

    def success(self, _value):
        return None

    def error(self, _value):
        return None

    def code(self, _value):
        return None


def test_sidebar_custom_extraction_defaults_to_selected_profile_fields(monkeypatch):
    fake_st = _InvoiceCustomExtractionStreamlit()
    monkeypatch.setattr(sidebar, "st", fake_st)

    state = sidebar.render_sidebar(Settings())

    assert state.profile_id == "invoice"
    assert state.extraction_mode == "Custom field extraction"
    assert "invoice_number" in state.fields
    assert "total" in state.fields
    assert fake_st.text_area_values["Enter fields to extract (comma separated):"].startswith(
        "invoice_number, invoice_date"
    )
