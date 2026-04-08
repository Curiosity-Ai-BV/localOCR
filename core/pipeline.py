from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

from PIL import Image

from adapters.ollama_adapter import query_ollama
from core.errors import PDFError
from core.image_utils import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_IMAGE_SIZE,
    image_to_base64,
    image_to_png_bytes,
    resize_image,
)
from core.json_extract import extract_structured_data
from core.logging import get_logger
from core.models import Result
from core.pdf_utils import PDFNotSupportedError, ensure_pdf_support, iter_pdf_pages
from core.prompts import PromptConfig
from core.settings import Settings

JSONDict = Dict[str, object]

_log = get_logger("pipeline")


def _resolve_prompts(prompts: Optional[PromptConfig]) -> PromptConfig:
    if prompts is not None:
        return prompts
    # Honour legacy module-global templates for back-compat.
    from core.templates import current_prompt_config
    return current_prompt_config()


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
    prompts: Optional[PromptConfig] = None,
) -> Tuple[JSONDict, str, Optional[JSONDict]]:
    """Process a single image with optional field extraction.

    Returns: (result_dict, raw_content, structured_dict_or_none)
    """
    pcfg = _resolve_prompts(prompts)
    t0 = time.perf_counter()
    prepared = resize_image(image, max_image_size)
    img_base64, encoded_size = image_to_base64(prepared, jpeg_quality)

    run_infer = inference
    if run_infer is None:
        def run_infer(prompt: str, img_b64: str, _model: str = model, **kwargs) -> str:  # type: ignore
            return query_ollama(prompt, img_b64, _model, options=options or {}, system_prompt=system_prompt, **kwargs)

    if not fields:
        prompt = pcfg.build_description()
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
        prompt = pcfg.build_extraction(fields)
        content = run_infer(prompt, img_base64, model, format="json")
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
    prompts: Optional[PromptConfig] = None,
) -> Generator[
    Tuple[
        Optional[int],
        Optional[int],
        Optional[Image.Image],
        str,
        str,
        Optional[JSONDict],
        Optional[float],
        Optional[Tuple[int, int]],
        Optional[int],
    ],
    None,
    None,
]:
    """Process a PDF file using PyMuPDF, yielding page-level results."""
    try:
        ensure_pdf_support()
        if process_pages_separately:
            for page_num, page_count, img in iter_pdf_pages(file_bytes, scale=pdf_scale):
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
                    prompts=prompts,
                )
                elapsed = result.get("duration_sec") if isinstance(result, dict) else None
                dims = (result.get("input_width"), result.get("input_height")) if isinstance(result, dict) else None
                size_bytes = result.get("encoded_bytes") if isinstance(result, dict) else None
                yield page_num, page_count, img, page_filename, content, structured_data, elapsed, dims, size_bytes
        else:
            first = next(iter_pdf_pages(file_bytes, scale=pdf_scale), None)
            if first is None:
                raise PDFError("PDF contains no renderable pages.")
            page_num, page_count, img = first
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
                prompts=prompts,
            )
            elapsed = result.get("duration_sec") if isinstance(result, dict) else None
            dims = (result.get("input_width"), result.get("input_height")) if isinstance(result, dict) else None
            size_bytes = result.get("encoded_bytes") if isinstance(result, dict) else None
            yield 0, page_count, img, filename, content, structured_data, elapsed, dims, size_bytes
    except PDFNotSupportedError as e:
        yield None, None, None, filename, str(e), None, None, None, None
    except Exception as e:
        _log.exception("pdf_processing_failed", extra={"filename": filename})
        yield None, None, None, filename, f"Error processing PDF: {str(e)}", None, None, None, None


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


@dataclass
class BatchJob:
    """One unit of work for :func:`run_batch`."""

    source: str  # display name / path
    data: Union[bytes, Image.Image]
    kind: str  # "image" or "pdf"


@dataclass
class BatchConfig:
    model: str
    fields: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    settings: Settings = None  # type: ignore[assignment]
    prompts: PromptConfig = None  # type: ignore[assignment]
    pdf_pages_separately: bool = True
    inference: Optional[Callable[[str, str, str], str]] = None

    def __post_init__(self) -> None:
        if self.settings is None:
            self.settings = Settings.from_env()
        if self.prompts is None:
            self.prompts = PromptConfig()
        if self.options is None:
            self.options = {}


def _job_from_path(path: str) -> BatchJob:
    name = os.path.basename(path)
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            return BatchJob(source=name, data=f.read(), kind="pdf")
    img = Image.open(path)
    img.load()
    return BatchJob(source=name, data=img, kind="image")


def run_batch(
    jobs: Iterable[Union[BatchJob, str]],
    cfg: BatchConfig,
) -> Iterator[Result]:
    """Yield :class:`Result` instances for each job.

    ``jobs`` accepts either :class:`BatchJob` instances or file path strings.
    This is the canonical entry point used by both the CLI and the Streamlit
    UI so they share a single execution path.
    """
    mode = "extract" if cfg.fields else "describe"
    for raw_job in jobs:
        job = _job_from_path(raw_job) if isinstance(raw_job, str) else raw_job
        try:
            if job.kind == "pdf":
                assert isinstance(job.data, (bytes, bytearray))
                for page_info in process_pdf(
                    bytes(job.data),
                    job.source,
                    cfg.fields,
                    cfg.pdf_pages_separately,
                    model=cfg.model,
                    system_prompt=cfg.system_prompt,
                    options=cfg.options,
                    max_image_size=cfg.settings.max_image_size,
                    jpeg_quality=cfg.settings.jpeg_quality,
                    pdf_scale=cfg.settings.pdf_scale,
                    inference=cfg.inference,
                    prompts=cfg.prompts,
                ):
                    page_num, page_count, image, page_filename, content, structured, elapsed, dims, size_bytes = page_info
                    if page_num is None:
                        yield Result(
                            source=job.source,
                            mode=mode,
                            text="",
                            error=content,
                        )
                        continue
                    clean_fields = {}
                    if structured:
                        clean_fields = {k: v for k, v in structured.items() if k != "filename"}
                    yield Result(
                        source=page_filename,
                        mode=mode,
                        text=content,
                        raw=content,
                        fields=clean_fields,
                        page=page_num,
                        page_count=page_count,
                        latency_ms=int((elapsed or 0.0) * 1000),
                        dimensions=tuple(dims) if dims and len(dims) == 2 else None,  # type: ignore[arg-type]
                        encoded_bytes=size_bytes,
                        preview_image_bytes=image_to_png_bytes(image) if image is not None else None,
                    )
            else:
                assert isinstance(job.data, Image.Image)
                result, content, structured = process_image(
                    job.data,
                    job.source,
                    cfg.fields,
                    model=cfg.model,
                    system_prompt=cfg.system_prompt,
                    options=cfg.options,
                    max_image_size=cfg.settings.max_image_size,
                    jpeg_quality=cfg.settings.jpeg_quality,
                    inference=cfg.inference,
                    prompts=cfg.prompts,
                )
                clean_fields = {}
                if structured:
                    clean_fields = {k: v for k, v in structured.items() if k != "filename"}
                elapsed_sec = float(result.get("duration_sec", 0.0)) if isinstance(result, dict) else 0.0
                dims_t: Optional[Tuple[int, int]] = None
                if isinstance(result, dict):
                    w = result.get("input_width")
                    h = result.get("input_height")
                    if isinstance(w, int) and isinstance(h, int):
                        dims_t = (w, h)
                yield Result(
                    source=job.source,
                    mode=mode,
                    text=content,
                    raw=content,
                    fields=clean_fields,
                    latency_ms=int(elapsed_sec * 1000),
                    dimensions=dims_t,
                    encoded_bytes=result.get("encoded_bytes") if isinstance(result, dict) else None,
                    preview_image_bytes=image_to_png_bytes(job.data),
                )
        except Exception as e:
            _log.exception("job_failed", extra={"source": job.source})
            yield Result(source=job.source, mode=mode, text="", error=str(e))
