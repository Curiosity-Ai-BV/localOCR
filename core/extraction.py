"""Schema and evidence helpers for structured field extraction."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.models import FieldEvidence

JSONDict = Dict[str, Any]


def normalize_extraction_fields(fields: Optional[Iterable[str]]) -> List[str]:
    """Return non-empty field names with stable de-duplication."""
    normalized: List[str] = []
    seen: set[str] = set()
    for field in fields or []:
        name = str(field).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def build_extraction_schema(fields: Iterable[str]) -> JSONDict:
    """Build an Ollama-compatible JSON Schema for requested fields."""
    field_names = normalize_extraction_fields(fields)
    value_schema: JSONDict = {
        "type": ["string", "number", "integer", "boolean", "array", "object", "null"],
    }
    return {
        "type": "object",
        "properties": {
            field: {
                **value_schema,
                "description": f"Extracted value for {field}.",
            }
            for field in field_names
        },
        "required": field_names,
        "additionalProperties": True,
    }


def is_missing_extracted_value(value: Any) -> bool:
    """Return whether an extracted requested value should count as missing."""
    return value is None or (isinstance(value, str) and not value.strip())


def clean_result_fields(
    *,
    requested_fields: Optional[Iterable[str]],
    parsed_fields: JSONDict,
) -> JSONDict:
    """Remove missing values for requested fields while preserving other keys."""
    requested = set(normalize_extraction_fields(requested_fields))
    return {
        field: value
        for field, value in parsed_fields.items()
        if field not in requested or not is_missing_extracted_value(value)
    }


def build_field_evidence(
    *,
    requested_fields: Iterable[str],
    parsed_fields: JSONDict,
    raw_content: str,
    engine: str,
    page: Optional[int] = None,
) -> dict[str, FieldEvidence]:
    """Create simple per-field evidence without changing extracted values."""
    evidence: dict[str, FieldEvidence] = {}
    for field in normalize_extraction_fields(requested_fields):
        value = parsed_fields.get(field)
        is_present = field in parsed_fields and not is_missing_extracted_value(value)
        evidence[field] = FieldEvidence(
            value=value if is_present else None,
            confidence=None,
            page=page,
            evidence_text=raw_content,
            engine=engine,
            validation_status="present" if is_present else "missing",
        )
    return evidence
