"""Unified writers for Result objects.

This is the single source of truth for CSV/JSONL export. Both the CLI and
the Streamlit UI should go through :func:`write_results`.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any, Iterable, List, Literal, Sequence, TextIO, Union

from core.models import Result

Format = Literal["csv", "jsonl", "structured_csv"]
Target = Union[str, Path, TextIO]


def _open(target: Target):
    if isinstance(target, (str, Path)):
        return open(target, "w", encoding="utf-8", newline="")
    return _NullCtx(target)


class _NullCtx:
    def __init__(self, f: TextIO) -> None:
        self.f = f

    def __enter__(self) -> TextIO:
        return self.f

    def __exit__(self, *exc: Any) -> None:
        return None


def _results_csv(results: Sequence[Result]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Filename", "Description"])
    for r in results:
        w.writerow([r.source, r.text or r.error or ""])
    return buf.getvalue()


def _results_jsonl(results: Sequence[Result]) -> str:
    lines: List[str] = []
    for r in results:
        lines.append(
            json.dumps(
                {
                    "source": r.source,
                    "page": r.page,
                    "mode": r.mode,
                    "text": r.text,
                    "fields": r.fields,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _structured_csv(results: Sequence[Result]) -> str:
    rows = [r for r in results if r.fields]
    if not rows:
        return ""
    all_fields = {"filename"}
    for r in rows:
        all_fields.update(r.fields.keys())
    field_list = sorted(all_fields)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(field_list)
    for r in rows:
        row = []
        for f in field_list:
            if f == "filename":
                row.append(r.source)
            else:
                row.append(r.fields.get(f, ""))
        w.writerow(row)
    return buf.getvalue()


def _plain_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _field_evidence_dict(result: Result) -> dict[str, Any]:
    return {
        field: _plain_dict(result.field_evidence[field])
        for field in sorted(result.field_evidence)
    }


def _evidence_result(result: Result) -> dict[str, Any]:
    return {
        "source": result.source,
        "mode": result.mode,
        "text": result.text,
        "fields": result.fields,
        "ocr_text": result.ocr_text,
        "ocr_blocks": [_plain_dict(block) for block in result.ocr_blocks],
        "field_evidence": _field_evidence_dict(result),
        "engine": result.engine,
        "profile_id": result.profile_id,
        "preprocess_steps": list(result.preprocess_steps),
        "backend_note": result.backend_note,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "page": result.page,
        "page_count": result.page_count,
        "dimensions": list(result.dimensions) if result.dimensions else None,
        "encoded_bytes": result.encoded_bytes,
    }


def render_evidence(results: Iterable[Result]) -> str:
    """Serialize detailed OCR evidence without embedding binary previews."""
    items = list(results)
    payload = {
        "schema_version": "localocr.evidence.v1",
        "summary": {
            "ok": not any(result.error for result in items),
            "total_results": len(items),
            "error_count": sum(1 for result in items if result.error),
        },
        "results": [_evidence_result(result) for result in items],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_results(results: Iterable[Result], format: Format = "csv") -> str:
    items = list(results)
    if format == "csv":
        return _results_csv(items)
    if format == "jsonl":
        return _results_jsonl(items)
    if format == "structured_csv":
        return _structured_csv(items)
    raise ValueError(f"Unknown format: {format}")


def write_results(
    results: Iterable[Result],
    target: Target,
    format: Format = "csv",
) -> str:
    """Write results to ``target`` and return the serialized payload."""
    payload = render_results(results, format=format)
    with _open(target) as f:
        f.write(payload)
    return payload


def write_evidence(results: Iterable[Result], target: Target) -> str:
    """Write detailed OCR evidence JSON to ``target``."""
    payload = render_evidence(results)
    with _open(target) as f:
        f.write(payload)
    return payload


def read_results_csv(target: Target) -> List[dict]:
    """Round-trip helper used by tests."""
    if isinstance(target, (str, Path)):
        with open(target, "r", encoding="utf-8") as f:
            data = f.read()
    else:
        data = target.read()
    reader = csv.DictReader(io.StringIO(data))
    return list(reader)
