"""Central settings for localOCR.

Frozen dataclass preferred over pydantic-settings to avoid a new runtime dep.
Loaded from environment variables with sane defaults; CLI / UI may override
per-invocation by constructing a new Settings instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


@dataclass(frozen=True)
class Settings:
    max_image_size: int = 1920
    jpeg_quality: int = 90
    pdf_scale: float = 1.5
    default_model: str = "gemma4:latest"
    request_timeout: float = 120.0
    max_concurrency: int = 1
    model_list_ttl: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_image_size=_env_int("LOCALOCR_MAX_IMAGE_SIZE", 1920),
            jpeg_quality=_env_int("LOCALOCR_JPEG_QUALITY", 90),
            pdf_scale=_env_float("LOCALOCR_PDF_SCALE", 1.5),
            default_model=_env_str("LOCALOCR_DEFAULT_MODEL", "gemma4:latest"),
            request_timeout=_env_float("LOCALOCR_REQUEST_TIMEOUT", 120.0),
            max_concurrency=_env_int("LOCALOCR_MAX_CONCURRENCY", 1),
            model_list_ttl=_env_float("LOCALOCR_MODEL_LIST_TTL", 30.0),
        )

    def merged(self, **overrides) -> "Settings":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)
