import importlib
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image


def _fresh_adapter():
    sys.modules.pop("adapters.docling_adapter", None)
    return importlib.import_module("adapters.docling_adapter")


def test_docling_adapter_import_does_not_import_docling_modules():
    sys.modules.pop("docling", None)
    sys.modules.pop("docling.document_converter", None)
    sys.modules.pop("docling.datamodel.base_models", None)

    _fresh_adapter()

    assert "docling" not in sys.modules
    assert "docling.document_converter" not in sys.modules


def test_missing_docling_reports_unavailable_install_message(monkeypatch):
    adapter = _fresh_adapter()

    def fake_import(name):
        if name.startswith("docling"):
            raise ModuleNotFoundError("No module named 'docling'")
        return importlib.import_module(name)

    monkeypatch.setattr(adapter.importlib, "import_module", fake_import)

    assert adapter.is_docling_available() is False
    status = adapter.get_docling_status()
    assert status.available is False
    assert "pip install -r requirements-docling.txt" in status.message

    try:
        adapter.convert_document_bytes("invoice.pdf", b"%PDF", "pdf")
    except adapter.DoclingUnavailableError as exc:
        assert "pip install -r requirements-docling.txt" in str(exc)
    else:
        raise AssertionError("Expected DoclingUnavailableError")


def test_convert_requires_local_artifacts_path_when_docling_is_importable(monkeypatch):
    adapter = _fresh_adapter()

    class FakeDocumentStream:
        pass

    class FakeDocumentConverter:
        pass

    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
    monkeypatch.setattr(
        adapter,
        "_load_docling_symbols",
        lambda: adapter.DoclingSymbols(
            DocumentConverter=FakeDocumentConverter,
            DocumentStream=FakeDocumentStream,
            PdfPipelineOptions=None,
            PdfFormatOption=None,
            ImageFormatOption=None,
            InputFormat=None,
        ),
    )

    try:
        adapter.convert_document_bytes("invoice.pdf", b"%PDF", "pdf")
    except adapter.DoclingUnavailableError as exc:
        message = str(exc)
        assert "DOCLING_ARTIFACTS_PATH" in message
        assert "pre-download Docling models" in message
    else:
        raise AssertionError("Expected DoclingUnavailableError")


def test_convert_requires_local_artifacts_path_directory(monkeypatch, tmp_path):
    adapter = _fresh_adapter()

    class FakeDocumentStream:
        pass

    class FakeDocumentConverter:
        pass

    artifact_file = tmp_path / "not-a-directory"
    artifact_file.write_text("nope", encoding="utf-8")
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(artifact_file))
    monkeypatch.setattr(
        adapter,
        "_load_docling_symbols",
        lambda: adapter.DoclingSymbols(
            DocumentConverter=FakeDocumentConverter,
            DocumentStream=FakeDocumentStream,
            PdfPipelineOptions=None,
            PdfFormatOption=None,
            ImageFormatOption=None,
            InputFormat=None,
        ),
    )

    try:
        adapter.convert_document_bytes("invoice.pdf", b"%PDF", "pdf")
    except adapter.DoclingUnavailableError as exc:
        message = str(exc)
        assert "DOCLING_ARTIFACTS_PATH" in message
        assert "directory" in message
        assert str(artifact_file) in message
    else:
        raise AssertionError("Expected DoclingUnavailableError")


def test_missing_local_only_config_symbols_fail_closed(monkeypatch, tmp_path):
    adapter = _fresh_adapter()
    artifacts_path = tmp_path / "docling-artifacts"
    artifacts_path.mkdir()
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(artifacts_path))

    class FakeDocumentStream:
        pass

    class FakeDocumentConverter:
        instantiated = False

        def __init__(self, **kwargs):
            type(self).instantiated = True

    class FakeInputFormat:
        PDF = "pdf-format"
        IMAGE = "image-format"

    monkeypatch.setattr(
        adapter,
        "_load_docling_symbols",
        lambda: adapter.DoclingSymbols(
            DocumentConverter=FakeDocumentConverter,
            DocumentStream=FakeDocumentStream,
            PdfPipelineOptions=None,
            PdfFormatOption=None,
            ImageFormatOption=None,
            InputFormat=FakeInputFormat,
        ),
    )

    try:
        adapter.convert_document_bytes("invoice.pdf", b"%PDF", "pdf")
    except adapter.DoclingUnavailableError as exc:
        message = str(exc)
        assert "local-only configuration symbols" in message
        assert "unsupported by this adapter" in message
    else:
        raise AssertionError("Expected DoclingUnavailableError")

    assert FakeDocumentConverter.instantiated is False


def test_convert_pdf_bytes_configures_local_artifacts_and_stream_name(
    monkeypatch,
    tmp_path,
):
    adapter = _fresh_adapter()
    artifacts_path = tmp_path / "docling-artifacts"
    artifacts_path.mkdir()
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(artifacts_path))

    class FakeInputFormat:
        PDF = "pdf-format"
        IMAGE = "image-format"

    class FakePipelineOptions:
        def __init__(self, *, artifacts_path):
            self.artifacts_path = artifacts_path
            self.enable_remote_services = True
            self.allow_external_plugins = True

    class FakePdfFormatOption:
        def __init__(self, *, pipeline_options):
            self.pipeline_options = pipeline_options

    class FakeImageFormatOption:
        def __init__(self, *, pipeline_options):
            self.pipeline_options = pipeline_options

    class FakeDocumentStream:
        def __init__(self, *, name, stream):
            self.name = name
            self.stream = stream

    class FakeDocument:
        def export_to_markdown(self):
            return "# Invoice\n\nTotal: 42"

    class FakeConverted:
        document = FakeDocument()

    class FakeDocumentConverter:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

        def convert(self, source):
            assert isinstance(source, FakeDocumentStream)
            assert source.name == "invoice.pdf"
            assert source.stream.read() == b"%PDF-1.7"
            return FakeConverted()

    monkeypatch.setattr(
        adapter,
        "_load_docling_symbols",
        lambda: adapter.DoclingSymbols(
            DocumentConverter=FakeDocumentConverter,
            DocumentStream=FakeDocumentStream,
            PdfPipelineOptions=FakePipelineOptions,
            PdfFormatOption=FakePdfFormatOption,
            ImageFormatOption=FakeImageFormatOption,
            InputFormat=FakeInputFormat,
        ),
    )

    result = adapter.convert_document_bytes("invoice", b"%PDF-1.7", "pdf")

    assert result.source == "invoice.pdf"
    assert result.engine == "docling"
    assert result.mode == "describe"
    assert result.text == "# Invoice\n\nTotal: 42"
    assert result.ocr_text == "# Invoice\n\nTotal: 42"
    assert len(result.ocr_blocks) == 1
    assert result.ocr_blocks[0].text == "# Invoice\n\nTotal: 42"
    assert FakeDocumentConverter.last_kwargs is not None
    assert FakeDocumentConverter.last_kwargs["allowed_formats"] == ["pdf-format"]
    format_options = FakeDocumentConverter.last_kwargs["format_options"]
    pdf_pipeline = format_options["pdf-format"].pipeline_options
    assert Path(pdf_pipeline.artifacts_path) == artifacts_path
    assert pdf_pipeline.enable_remote_services is False
    assert pdf_pipeline.allow_external_plugins is False


def test_convert_pil_image_uses_docling_document_text(monkeypatch, tmp_path):
    adapter = _fresh_adapter()
    artifacts_path = tmp_path / "docling-artifacts"
    artifacts_path.mkdir()
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(artifacts_path))

    class FakeInputFormat:
        PDF = "pdf-format"
        IMAGE = "image-format"

    class FakePipelineOptions:
        def __init__(self, *, artifacts_path):
            self.artifacts_path = artifacts_path

    class FakePdfFormatOption:
        def __init__(self, *, pipeline_options):
            self.pipeline_options = pipeline_options

    class FakeImageFormatOption:
        def __init__(self, *, pipeline_options):
            self.pipeline_options = pipeline_options

    class FakeDocumentStream:
        def __init__(self, *, name, stream):
            self.name = name
            self.stream = stream

    class FakeDocument:
        def export_to_text(self):
            return "plain OCR text"

    class FakeConverted:
        document = FakeDocument()

    class FakeDocumentConverter:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

        def convert(self, source):
            assert isinstance(source, FakeDocumentStream)
            assert source.name == "scan.png"
            assert isinstance(source.stream, BytesIO)
            assert source.stream.getvalue().startswith(b"\x89PNG")
            return FakeConverted()

    monkeypatch.setattr(
        adapter,
        "_load_docling_symbols",
        lambda: adapter.DoclingSymbols(
            DocumentConverter=FakeDocumentConverter,
            DocumentStream=FakeDocumentStream,
            PdfPipelineOptions=FakePipelineOptions,
            PdfFormatOption=FakePdfFormatOption,
            ImageFormatOption=FakeImageFormatOption,
            InputFormat=FakeInputFormat,
        ),
    )

    image = Image.new("RGB", (8, 8), "white")
    result = adapter.convert_image("scan", image)

    assert result.text == "plain OCR text"
    assert result.dimensions == (8, 8)
    assert result.engine == "docling"
    assert result.source == "scan.png"
    assert FakeDocumentConverter.last_kwargs is not None
    assert FakeDocumentConverter.last_kwargs["allowed_formats"] == ["image-format"]
    image_pipeline = (
        FakeDocumentConverter.last_kwargs["format_options"]["image-format"]
        .pipeline_options
    )
    assert Path(image_pipeline.artifacts_path) == artifacts_path


def test_rendered_pdf_page_image_source_gets_image_stream_suffix():
    adapter = _fresh_adapter()

    assert adapter._source_name("sample.pdf (Page 1)", "image") == "sample.pdf (Page 1).png"
    assert adapter._source_name("scan.png", "image") == "scan.png"


def test_requirements_docling_contains_only_optional_docling_dependency():
    with open("requirements-docling.txt", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    assert lines == ["docling>=2.88,<3"]


def test_base_requirements_do_not_include_docling():
    for path in ("requirements.txt", "constraints.txt"):
        with open(path, encoding="utf-8") as f:
            content = f.read().lower()

        assert "docling" not in content
