from __future__ import annotations

import os
import time
from dataclasses import dataclass
from inspect import Parameter, signature
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
from core.models import Mode, Result
from core.pdf_utils import PDFNotSupportedError, ensure_pdf_support, iter_pdf_pages
from core.prompts import PromptConfig
from core.settings import Settings

JSONDict = Dict[str, object]
PDFPageResult = Tuple[
    Optional[int],
    Optional[int],
    Optional[Image.Image],
    str,
    str,
    Optional[JSONDict],
    Optional[float],
    Optional[Tuple[int, int]],
    Optional[int],
]

_log = get_logger("pipeline")


def _accepts_kwarg(func: Callable[..., str], name: str) -> bool:
    try:
        sig = signature(func)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind is Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            return True
    return False


def _call_inference(
    func: Callable[..., str],
    prompt: str,
    image_base64: str,
    model: str,
    **kwargs: Any,
) -> str:
    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if _accepts_kwarg(func, name)
    }
    return func(prompt, image_base64, model, **supported_kwargs)


def _resolve_prompts(prompts: Optional[PromptConfig]) -> PromptConfig:
    if prompts is not None:
        return prompts
    # Honour legacy module-global templates for back-compat.
    from core.templates import current_prompt_config
    return current_prompt_config()


def _number_as_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_value(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    return None


def _dimensions_from_result(result: JSONDict) -> Optional[Tuple[int, int]]:
    width = result.get("input_width")
    height = result.get("input_height")
    if isinstance(width, int) and isinstance(height, int):
        return (width, height)
    return None


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
    settings: Optional[Settings] = None,
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
            return query_ollama(
                prompt,
                img_b64,
                _model,
                options=options or {},
                system_prompt=system_prompt,
                **kwargs,
            )

    if not fields:
        prompt = pcfg.build_description()
        content = _call_inference(run_infer, prompt, img_base64, model, settings=settings)
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
        content = _call_inference(run_infer, prompt, img_base64, model, format="json", settings=settings)
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
    settings: Optional[Settings] = None,
) -> Generator[PDFPageResult, None, None]:
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
                    settings=settings,
                )
                elapsed = _number_as_float(result.get("duration_sec"))
                dims = _dimensions_from_result(result)
                size_bytes = _int_value(result.get("encoded_bytes"))
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
                settings=settings,
            )
            elapsed = _number_as_float(result.get("duration_sec"))
            dims = _dimensions_from_result(result)
            size_bytes = _int_value(result.get("encoded_bytes"))
            yield 0, page_count, img, filename, content, structured_data, elapsed, dims, size_bytes
    except PDFNotSupportedError as e:
        yield None, None, None, filename, str(e), None, None, None, None
    except Exception as e:
        _log.exception("pdf_processing_failed", extra={"source": filename})
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
    mode: Mode = "extract" if cfg.fields else "describe"
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
                    settings=cfg.settings,
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
                        dimensions=dims,
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
                    settings=cfg.settings,
                )
                clean_fields = {}
                if structured:
                    clean_fields = {k: v for k, v in structured.items() if k != "filename"}
                elapsed_sec = _number_as_float(result.get("duration_sec")) or 0.0
                dims_t = _dimensions_from_result(result)
                encoded_bytes = _int_value(result.get("encoded_bytes"))
                yield Result(
                    source=job.source,
                    mode=mode,
                    text=content,
                    raw=content,
                    fields=clean_fields,
                    latency_ms=int(elapsed_sec * 1000),
                    dimensions=dims_t,
                    encoded_bytes=encoded_bytes,
                    preview_image_bytes=image_to_png_bytes(job.data),
                )
        except Exception as e:
            _log.exception("job_failed", extra={"source": job.source})
            yield Result(source=job.source, mode=mode, text="", error=str(e))
