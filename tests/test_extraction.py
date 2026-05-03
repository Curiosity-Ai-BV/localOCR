from core.extraction import build_extraction_schema, build_field_evidence


def test_build_extraction_schema_creates_object_schema_with_requested_fields():
    schema = build_extraction_schema(["Invoice number", "Total", "Total", " "])

    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["Invoice number", "Total"]
    assert schema["required"] == ["Invoice number", "Total"]
    assert schema["properties"]["Invoice number"]["description"]


def test_build_field_evidence_marks_present_and_missing_fields():
    evidence = build_field_evidence(
        requested_fields=["Invoice number", "Total"],
        parsed_fields={"Invoice number": "INV-42"},
        raw_content='{"Invoice number": "INV-42"}',
        engine="ollama",
        page=2,
    )

    assert evidence["Invoice number"].value == "INV-42"
    assert evidence["Invoice number"].validation_status == "present"
    assert evidence["Invoice number"].confidence is None
    assert evidence["Invoice number"].page == 2
    assert evidence["Total"].value is None
    assert evidence["Total"].validation_status == "missing"


def test_build_field_evidence_marks_whitespace_only_string_missing():
    evidence = build_field_evidence(
        requested_fields=["Total"],
        parsed_fields={"Total": "   \n\t"},
        raw_content='{"Total": "   "}',
        engine="ollama",
    )

    assert evidence["Total"].value is None
    assert evidence["Total"].validation_status == "missing"
