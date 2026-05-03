"""Typed result model used across pipeline, CLI, UI, and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

Mode = Literal["describe", "extract"]
BBox = Tuple[float, float, float, float]


@dataclass
class OCRBlock:
    text: str
    kind: str
    page: Optional[int] = None
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None


@dataclass
class FieldEvidence:
    value: Any
    confidence: Optional[float] = None
    page: Optional[int] = None
    bbox: Optional[BBox] = None
    evidence_text: str = ""
    engine: Optional[str] = None
    validation_status: Optional[str] = None


@dataclass
class DocumentProfile:
    id: str
    label: str
    fields: list[str] = field(default_factory=list)
    default_backend: Optional[str] = None
    default_preprocess: str = "none"


@dataclass
class Result:
    source: str
    mode: Mode
    text: str = ""
    raw: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    page: Optional[int] = None
    page_count: Optional[int] = None
    error: Optional[str] = None
    latency_ms: int = 0
    dimensions: Optional[Tuple[int, int]] = None
    encoded_bytes: Optional[int] = None
    preview_image_bytes: Optional[bytes] = None
    ocr_text: Optional[str] = None
    ocr_blocks: list[OCRBlock] = field(default_factory=list)
    field_evidence: dict[str, FieldEvidence] = field(default_factory=dict)
    engine: Optional[str] = None
    profile_id: Optional[str] = None
    preprocess_steps: list[str] = field(default_factory=list)
    backend_note: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.source

    @property
    def description(self) -> str:
        return self.text if self.mode == "describe" else ""

    @property
    def extraction(self) -> str:
        return self.text if self.mode == "extract" else ""

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Back-compat shape for older code paths that expected a dict."""
        d: Dict[str, Any] = {
            "filename": self.source,
            "duration_sec": round(self.latency_ms / 1000.0, 3),
        }
        if self.dimensions:
            d["input_width"], d["input_height"] = self.dimensions
        if self.encoded_bytes is not None:
            d["encoded_bytes"] = self.encoded_bytes
        if self.mode == "describe":
            d["description"] = self.text
        else:
            d["extraction"] = self.text
        if self.error:
            d["error"] = self.error
        return d
