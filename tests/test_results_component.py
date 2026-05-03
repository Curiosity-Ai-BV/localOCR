from core.models import FieldEvidence
from ui.components.results import (
    _plain_json,
    format_evidence_metadata,
    format_result_metadata,
    get_status_badge,
)


def test_format_result_metadata_keeps_existing_units():
    entry = {
        "duration_sec": 1.234,
        "dimensions": (1200, 800),
        "encoded_bytes": 1536,
    }

    assert format_result_metadata(entry) == ["1.23 s", "1200x800 px", "1.5 KB"]


def test_format_result_metadata_ignores_partial_values():
    entry = {
        "duration_sec": None,
        "dimensions": (1200,),
        "encoded_bytes": "1536",
    }

    assert format_result_metadata(entry) == []


def test_get_status_badge_returns_accessible_label_and_css_class():
    assert get_status_badge("done") == ("Done", "ocr-status-done")
    assert get_status_badge("error") == ("Error", "ocr-status-error")
    assert get_status_badge("unknown") == ("Check", "ocr-status-check")


def test_format_evidence_metadata_is_compact():
    entry = {
        "engine": "hybrid",
        "profile_id": "invoice",
        "backend_note": "auto fallback from docling",
    }

    assert format_evidence_metadata(entry) == [
        "engine: hybrid",
        "profile: invoice",
        "note: auto fallback from docling",
    ]


def test_field_evidence_is_converted_to_plain_json():
    evidence = {
        "total": FieldEvidence(
            value="42.00",
            confidence=0.9,
            evidence_text="Total 42.00",
            engine="hybrid",
        )
    }

    assert _plain_json(evidence) == {
        "total": {
            "value": "42.00",
            "confidence": 0.9,
            "page": None,
            "bbox": None,
            "evidence_text": "Total 42.00",
            "engine": "hybrid",
            "validation_status": None,
        }
    }
