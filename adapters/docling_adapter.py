"""Optional Docling OCR adapter.

Docling is intentionally not a base dependency. Keep imports lazy so the
default Ollama route and base install never load Docling modules.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from PIL import Image

from core.models import OCRBlock, Result

DOCLING_INSTALL_MESSAGE = (
    "Docling is not installed. Install the optional local Docling backend with: "
    "pip install -r requirements-docling.txt"
)
DOCLING_ARTIFACTS_MESSAGE = (
    "Docling local artifacts are required before conversion. pre-download Docling "
    "models and set DOCLING_ARTIFACTS_PATH to that local artifacts directory."
)
DOCLING_UNSUPPORTED_LOCAL_ONLY_MESSAGE = (
    "The installed Docling version does not expose the required local-only "
    "configuration symbols and is unsupported by this adapter. Install a "
    "supported Docling v2 release with: pip install -r requirements-docling.txt"
)


class DoclingUnavailableError(RuntimeError):
    """Raised when the optional Docling backend is requested but unavailable."""


@dataclass(frozen=True)
class DoclingStatus:
    available: bool
    message: str


@dataclass(frozen=True)
class DoclingSymbols:
    DocumentConverter: type[Any]
    DocumentStream: type[Any]
    PdfPipelineOptions: type[Any] | None
    PdfFormatOption: type[Any] | None
    ImageFormatOption: type[Any] | None
    InputFormat: Any | None


def _optional_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    return getattr(module, attr_name, None)


def _load_docling_symbols() -> DoclingSymbols:
    try:
        converter_module = importlib.import_module("docling.document_converter")
        base_models_module = importlib.import_module("docling.datamodel.base_models")
    except ModuleNotFoundError as exc:
        missing_name = exc.name or str(exc)
        if "docling" in missing_name:
            raise DoclingUnavailableError(DOCLING_INSTALL_MESSAGE) from exc
        raise

    return DoclingSymbols(
        DocumentConverter=converter_module.DocumentConverter,
        DocumentStream=base_models_module.DocumentStream,
        PdfPipelineOptions=_optional_attr(
            "docling.datamodel.pipeline_options",
            "PdfPipelineOptions",
        ),
        PdfFormatOption=getattr(converter_module, "PdfFormatOption", None),
        ImageFormatOption=getattr(converter_module, "ImageFormatOption", None),
        InputFormat=getattr(base_models_module, "InputFormat", None),
    )


def is_docling_available() -> bool:
    try:
        _load_docling_symbols()
    except DoclingUnavailableError:
        return False
    return True


def get_docling_status() -> DoclingStatus:
    if is_docling_available():
        return DoclingStatus(True, "Docling backend is available.")
    return DoclingStatus(False, DOCLING_INSTALL_MESSAGE)


def _document_text(converted: Any) -> str:
    document = getattr(converted, "document", converted)

    export_to_markdown = getattr(document, "export_to_markdown", None)
    if callable(export_to_markdown):
        text = export_to_markdown()
        if text is not None:
            return str(text)

    export_to_text = getattr(document, "export_to_text", None)
    if callable(export_to_text):
        text = export_to_text()
        if text is not None:
            return str(text)

    text_attr = getattr(document, "text", None)
    if text_attr is not None:
        return str(text_attr)

    return str(document)


def _local_artifacts_path() -> Path:
    raw_path = os.environ.get("DOCLING_ARTIFACTS_PATH", "").strip()
    if not raw_path:
        raise DoclingUnavailableError(DOCLING_ARTIFACTS_MESSAGE)

    artifacts_path = Path(raw_path).expanduser()
    if not artifacts_path.is_dir():
        raise DoclingUnavailableError(
            f"{DOCLING_ARTIFACTS_MESSAGE} Current value is not a directory: "
            f"{artifacts_path}"
        )
    return artifacts_path


def _source_name(source: str, kind: str) -> str:
    suffix = ".pdf" if kind == "pdf" else ".png"
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    source_path = Path(source)
    source_suffix = source_path.suffix.lower()
    if kind == "image" and source_suffix in image_suffixes:
        return source
    if source_suffix == suffix:
        return source
    if kind == "pdf" and source_path.suffix:
        return source
    return f"{source}{suffix}"


def _disable_remote_features(pipeline_options: Any) -> Any:
    if hasattr(pipeline_options, "enable_remote_services"):
        pipeline_options.enable_remote_services = False
    if hasattr(pipeline_options, "allow_external_plugins"):
        pipeline_options.allow_external_plugins = False
    return pipeline_options


def _make_pipeline_options(symbols: DoclingSymbols, artifacts_path: Path) -> Any:
    pipeline_options_cls = cast(type[Any], symbols.PdfPipelineOptions)
    pipeline_options = pipeline_options_cls(artifacts_path=str(artifacts_path))
    return _disable_remote_features(pipeline_options)


def _converter_kwargs(
    symbols: DoclingSymbols,
    *,
    kind: str,
    artifacts_path: Path,
) -> dict[str, Any]:
    if symbols.InputFormat is None or symbols.PdfPipelineOptions is None:
        raise DoclingUnavailableError(DOCLING_UNSUPPORTED_LOCAL_ONLY_MESSAGE)

    input_format = (
        getattr(symbols.InputFormat, "PDF", None)
        if kind == "pdf"
        else getattr(symbols.InputFormat, "IMAGE", None)
    )
    format_option_cls = (
        symbols.PdfFormatOption if kind == "pdf" else symbols.ImageFormatOption
    )
    if input_format is None or format_option_cls is None:
        raise DoclingUnavailableError(DOCLING_UNSUPPORTED_LOCAL_ONLY_MESSAGE)

    pipeline_options = _make_pipeline_options(symbols, artifacts_path)
    return {
        "allowed_formats": [input_format],
        "format_options": {
            input_format: format_option_cls(pipeline_options=pipeline_options),
        },
    }


def _result(
    *,
    source: str,
    text: str,
    dimensions: tuple[int, int] | None = None,
) -> Result:
    blocks = [OCRBlock(text=text, kind="document")] if text else []
    return Result(
        source=source,
        mode="describe",
        text=text,
        raw=text,
        ocr_text=text,
        ocr_blocks=blocks,
        engine="docling",
        dimensions=dimensions,
    )


def convert_document_bytes(source: str, data: bytes, kind: str) -> Result:
    """Convert local PDF or image bytes with Docling and return OCR text metadata."""
    if kind not in ("pdf", "image"):
        raise ValueError("Docling input kind must be 'pdf' or 'image'.")

    symbols = _load_docling_symbols()
    artifacts_path = _local_artifacts_path()
    stream_name = _source_name(source, kind)
    converter_kwargs = _converter_kwargs(
        symbols,
        kind=kind,
        artifacts_path=artifacts_path,
    )
    stream = BytesIO(data)
    stream.seek(0)
    document_stream = symbols.DocumentStream(name=stream_name, stream=stream)
    converted = symbols.DocumentConverter(**converter_kwargs).convert(document_stream)
    text = _document_text(converted)
    return _result(source=stream_name, text=text)


def convert_image(source: str, image: Image.Image) -> Result:
    """Convert a PIL image through the optional Docling backend."""
    stream = BytesIO()
    image.save(stream, format="PNG")
    result = convert_document_bytes(source, stream.getvalue(), "image")
    result.dimensions = image.size
    return result
