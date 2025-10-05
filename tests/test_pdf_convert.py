import io

import pytest
from PIL import Image

from core.pipeline import process_pdf

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - test skipped if PyMuPDF not present
    fitz = None


pytestmark = pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")


def make_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF")
    data = doc.write()
    doc.close()
    return data


def dummy_infer(prompt: str, img_b64: str, model: str) -> str:
    # Return a well-formed JSON for extraction tests
    return '{"Invoice number": "X-1", "Date": "2025-01-01"}'


def test_process_pdf_first_page_only():
    pdf_bytes = make_pdf_bytes()
    gen = process_pdf(
        pdf_bytes,
        "sample.pdf",
        fields=None,
        process_pages_separately=False,
        model="dummy",
        system_prompt=None,
        options={},
        max_image_size=512,
        jpeg_quality=80,
        pdf_scale=1.0,
        inference=dummy_infer,
    )
    page_num, page_count, image, page_filename, content, structured_data, elapsed_sec, dims, size_bytes = next(gen)
    assert page_num == 0
    assert isinstance(image, Image.Image)
    assert isinstance(page_filename, str)
    assert isinstance(content, str)


def test_process_pdf_per_page_with_extraction():
    pdf_bytes = make_pdf_bytes()
    gen = process_pdf(
        pdf_bytes,
        "sample.pdf",
        fields=["Invoice number", "Date"],
        process_pages_separately=True,
        model="dummy",
        system_prompt=None,
        options={},
        max_image_size=512,
        jpeg_quality=80,
        pdf_scale=1.0,
        inference=dummy_infer,
    )
    page_num, page_count, image, page_filename, content, structured_data, elapsed_sec, dims, size_bytes = next(gen)
    assert structured_data is not None
    assert structured_data.get("Invoice number") == "X-1"

