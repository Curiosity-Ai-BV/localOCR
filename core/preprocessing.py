"""Pillow-only image preprocessing presets for OCR preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from PIL import Image, ImageFilter, ImageOps


@dataclass
class PreprocessResult:
    image: Image.Image
    preset: str
    steps: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


PREPROCESSING_PRESETS = ("none", "document-clean", "high-accuracy-scan")
_PREPROCESSING_STEPS = {
    "none": [],
    "document-clean": ["grayscale", "autocontrast", "sharpen"],
    "high-accuracy-scan": ["grayscale", "autocontrast-cutoff-1", "unsharp-mask"],
}


def normalize_preprocess_preset(preset: str) -> str:
    """Return a known preprocessing preset id or raise a clear error."""
    normalized_preset = preset.strip().lower()
    if normalized_preset not in PREPROCESSING_PRESETS:
        valid = ", ".join(PREPROCESSING_PRESETS)
        raise ValueError(
            f"Unknown preprocessing preset '{preset}'. Expected one of: {valid}."
        )
    return normalized_preset


def expected_preprocess_steps(preset: str) -> list[str]:
    """Return the deterministic step names for a preprocessing preset."""
    return list(_PREPROCESSING_STEPS[normalize_preprocess_preset(preset)])


def preprocess_image(image: Image.Image, preset: str = "none") -> PreprocessResult:
    """Apply a deterministic preprocessing preset without mutating input."""
    normalized_preset = normalize_preprocess_preset(preset)

    prepared = image.copy()
    steps: list[str] = []
    if normalized_preset == "document-clean":
        prepared, steps = _document_clean(ImageOps.exif_transpose(prepared))
    elif normalized_preset == "high-accuracy-scan":
        prepared, steps = _high_accuracy_scan(ImageOps.exif_transpose(prepared))

    return PreprocessResult(
        image=prepared,
        preset=normalized_preset,
        steps=steps,
        metadata={
            "original_mode": image.mode,
            "original_size": image.size,
            "prepared_mode": prepared.mode,
            "prepared_size": prepared.size,
        },
    )


def _document_clean(image: Image.Image) -> tuple[Image.Image, list[str]]:
    prepared = ImageOps.grayscale(image)
    prepared = ImageOps.autocontrast(prepared)
    prepared = prepared.filter(ImageFilter.SHARPEN)
    return prepared, expected_preprocess_steps("document-clean")


def _high_accuracy_scan(image: Image.Image) -> tuple[Image.Image, list[str]]:
    prepared = ImageOps.grayscale(image)
    prepared = ImageOps.autocontrast(prepared, cutoff=1)
    prepared = prepared.filter(
        ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
    )
    return prepared, expected_preprocess_steps("high-accuracy-scan")
