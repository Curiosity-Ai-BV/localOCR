from ui.components.results import format_result_metadata, get_status_badge


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
