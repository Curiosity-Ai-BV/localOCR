"""Backend routing primitives for OCR execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OCRBackend = Literal["ollama", "docling", "hybrid", "auto"]

BACKEND_OLLAMA: OCRBackend = "ollama"
BACKEND_DOCLING: OCRBackend = "docling"
BACKEND_HYBRID: OCRBackend = "hybrid"
BACKEND_AUTO: OCRBackend = "auto"
SUPPORTED_BACKENDS: tuple[OCRBackend, ...] = (
    BACKEND_OLLAMA,
    BACKEND_DOCLING,
    BACKEND_HYBRID,
    BACKEND_AUTO,
)
IMPLEMENTED_BACKENDS: tuple[OCRBackend, ...] = (BACKEND_OLLAMA,)


@dataclass(frozen=True)
class BackendSelection:
    backend: OCRBackend
    profile_id: str
    preprocess: str


def normalize_backend(value: str) -> OCRBackend:
    """Return a known backend id or raise a clear configuration error."""
    normalized = value.strip().lower()
    if normalized in SUPPORTED_BACKENDS:
        return normalized  # type: ignore[return-value]
    valid = ", ".join(SUPPORTED_BACKENDS)
    raise ValueError(f"Unknown OCR backend '{value}'. Expected one of: {valid}.")


def resolve_backend(
    backend: str = BACKEND_OLLAMA,
    *,
    profile_id: str = "generic",
    preprocess: str = "none",
) -> BackendSelection:
    """Normalize backend routing fields without importing pipeline code."""
    normalized_profile = profile_id.strip().lower() or "generic"
    normalized_preprocess = preprocess.strip().lower() or "none"
    return BackendSelection(
        backend=normalize_backend(backend),
        profile_id=normalized_profile,
        preprocess=normalized_preprocess,
    )


def is_backend_implemented(backend: OCRBackend) -> bool:
    return backend in IMPLEMENTED_BACKENDS


def unsupported_backend_error(backend: OCRBackend) -> str:
    return (
        f"OCR backend '{backend}' is recognized but not implemented in this slice. "
        "Use ocr_backend='ollama' for the current vision path."
    )
