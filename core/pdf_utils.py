from __future__ import annotations

from typing import Generator, Tuple

from PIL import Image

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - handled by consumer
    fitz = None  # type: ignore


class PDFNotSupportedError(RuntimeError):
    """Raised when PDF processing is requested without the necessary deps."""


def is_pdf_supported() -> bool:
    """Return True when PyMuPDF is available on the runtime."""
    return fitz is not None


def ensure_pdf_support() -> None:
    """Guard utility to raise a helpful error when PDF support is missing."""
    if not is_pdf_supported():
        raise PDFNotSupportedError("PyMuPDF is required. Install with `pip install pymupdf`.")


def get_pdf_page_count(file_bytes: bytes) -> int:
    """Return number of pages for the provided PDF bytes."""
    ensure_pdf_support()
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # type: ignore[attr-defined]
    try:
        return len(doc)
    finally:
        doc.close()


def iter_pdf_pages(file_bytes: bytes, *, scale: float = 1.0) -> Generator[Tuple[int, int, Image.Image], None, None]:
    """Yield (page_index, total_pages, PIL.Image) tuples for the given PDF bytes."""
    ensure_pdf_support()
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # type: ignore[attr-defined]
    try:
        total_pages = len(doc)
        for page_index in range(total_pages):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))  # type: ignore[attr-defined]
            if pix.alpha:  # drop alpha channel to keep downstream JPEG encoding simple
                pix = fitz.Pixmap(fitz.csRGB, pix)  # type: ignore[attr-defined]
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield page_index, total_pages, image
    finally:
        doc.close()
