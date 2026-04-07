"""Typed result model used across pipeline, CLI, UI, and export."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

Mode = Literal["describe", "extract"]


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
