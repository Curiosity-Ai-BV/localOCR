"""Typed exception hierarchy for localOCR."""
from __future__ import annotations


class OCRError(Exception):
    """Base class for all localOCR domain errors."""


class ModelUnavailable(OCRError):
    """Raised when the requested model cannot be resolved or reached."""


class ParseError(OCRError):
    """Raised when a model response cannot be parsed into structured data."""


class PDFError(OCRError):
    """Raised when PDF decoding or rendering fails."""
