from __future__ import annotations

import json
from dataclasses import asdict

from core.models import DocumentProfile, FieldEvidence, OCRBlock, Result


def test_result_metadata_defaults_preserve_old_constructor():
    result = Result("sample.png", "extract", text="done", fields={"total": "10"})

    assert result.source == "sample.png"
    assert result.mode == "extract"
    assert result.fields == {"total": "10"}
    assert result.ocr_text is None
    assert result.ocr_blocks == []
    assert result.field_evidence == {}
    assert result.engine is None
    assert result.profile_id is None
    assert result.preprocess_steps == []


def test_result_evidence_is_separate_from_fields():
    evidence = FieldEvidence(
        value="10",
        confidence=0.95,
        page=1,
        bbox=(1.0, 2.0, 3.0, 4.0),
        evidence_text="Total 10",
        engine="test-engine",
        validation_status="valid",
    )
    result = Result(
        source="invoice.png",
        mode="extract",
        fields={"total": "10"},
        field_evidence={"total": evidence},
    )

    assert result.fields == {"total": "10"}
    assert result.field_evidence["total"] == evidence


def test_core_dataclasses_serialize_to_plain_json():
    block = OCRBlock(
        text="Invoice 123",
        kind="line",
        page=1,
        bbox=(10.0, 20.0, 30.0, 40.0),
        confidence=0.9,
    )
    evidence = FieldEvidence(
        value="123",
        confidence=0.9,
        page=1,
        bbox=(10.0, 20.0, 30.0, 40.0),
        evidence_text="Invoice 123",
        engine="test-engine",
        validation_status="valid",
    )
    profile = DocumentProfile(
        id="invoice",
        label="Invoice",
        fields=["invoice_number"],
        default_backend="ollama",
        default_preprocess="document-clean",
    )

    payload = {
        "block": asdict(block),
        "evidence": asdict(evidence),
        "profile": asdict(profile),
    }

    assert json.loads(json.dumps(payload))["profile"]["id"] == "invoice"
