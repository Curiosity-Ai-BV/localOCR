from __future__ import annotations

import time
from typing import Callable, Dict, Generator, List, Optional, Tuple

from PIL import Image

from adapters.ollama_adapter import query_ollama
from core.image_utils import DEFAULT_JPEG_QUALITY, DEFAULT_MAX_IMAGE_SIZE, image_to_base64, resize_image
from core.json_extract import extract_structured_data
from core.templates import build_description_prompt, build_extraction_prompt

try:
    import fitz  # type: ignore
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

JSONDict = Dict[str, object]


def process_image(
    image: Image.Image,
    filename: str,
    fields: Optional[List[str]] = None,
    *,
    model: str,
    system_prompt: Optional[str],
    options: Optional[JSONDict],
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    inference: Optional[Callable[[str, str, str], str]] = None,
) -> Tuple[JSONDict, str, Optional[JSONDict]]:
    """Process a single image with optional field extraction.

    Returns: (result_dict, raw_content, structured_dict_or_none)
    """
    t0 = time.perf_counter()
    prepared = resize_image(image, max_image_size)
    img_base64, encoded_size = image_to_base64(prepared, jpeg_quality)

    run_infer = inference
    if run_infer is None:
        def run_infer(prompt: str, img_b64: str, _model: str = model) -> str:  # type: ignore
            return query_ollama(prompt, img_b64, _model, options=options or {}, system_prompt=system_prompt)

    if not fields:
        prompt = build_description_prompt()
        content = run_infer(prompt, img_base64, model)
        elapsed = time.perf_counter() - t0
        return {
            "filename": filename,
            "description": content,
            "duration_sec": round(elapsed, 3),
            "input_width": prepared.size[0],
            "input_height": prepared.size[1],
            "encoded_bytes": int(encoded_size),
        }, content, None
    else:
        prompt = build_extraction_prompt(fields)
        content = run_infer(prompt, img_base64, model)
        structured_data: JSONDict = {"filename": filename}
        parsed = extract_structured_data(content, fields)
        structured_data.update(parsed)
        elapsed = time.perf_counter() - t0
        return {
            "filename": filename,
            "extraction": content,
            "duration_sec": round(elapsed, 3),
            "input_width": prepared.size[0],
            "input_height": prepared.size[1],
            "encoded_bytes": int(encoded_size),
        }, content, structured_data


def process_pdf(
    file_bytes: bytes,
    filename: str,
    fields: Optional[List[str]] = None,
    process_pages_separately: bool = True,
    *,
    model: str,
    system_prompt: Optional[str],
    options: Optional[JSONDict],
    max_image_size: int,
    jpeg_quality: int,
    pdf_scale: float,
    inference: Optional[Callable[[str, str, str], str]] = None,
) -> Generator[Tuple[Optional[int], Optional[int], Optional[Image.Image], str, str, Optional[JSONDict], Optional[float], Optional[Tuple[int,int]], Optional[int]], None, None]:
    """Process a PDF file using PyMuPDF, yielding page-level results."""
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(pdf_document)

        if process_pages_separately:
            for page_num in range(page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(pdf_scale, pdf_scale))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_filename = f"{filename} (Page {page_num + 1})"

                result, content, structured_data = process_image(
                    img,
                    page_filename,
                    fields,
                    model=model,
                    system_prompt=system_prompt,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                    inference=inference,
                )
                elapsed = result.get("duration_sec") if isinstance(result, dict) else None
                dims = (result.get("input_width"), result.get("input_height")) if isinstance(result, dict) else None
                size_bytes = result.get("encoded_bytes") if isinstance(result, dict) else None
                yield page_num, page_count, img, page_filename, content, structured_data, elapsed, dims, size_bytes
        else:
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(pdf_scale, pdf_scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            result, content, structured_data = process_image(
                img,
                filename,
                fields,
                model=model,
                system_prompt=system_prompt,
                options=options,
                max_image_size=max_image_size,
                jpeg_quality=jpeg_quality,
                inference=inference,
            )
            elapsed = result.get("duration_sec") if isinstance(result, dict) else None
            dims = (result.get("input_width"), result.get("input_height")) if isinstance(result, dict) else None
            size_bytes = result.get("encoded_bytes") if isinstance(result, dict) else None
            yield 0, page_count, img, filename, content, structured_data, elapsed, dims, size_bytes
    except Exception as e:
        yield None, None, None, filename, f"Error processing PDF: {str(e)}", None, None, None, None

