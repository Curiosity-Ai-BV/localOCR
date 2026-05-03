import io
import json

from core.export import read_results_csv, render_evidence, render_results, write_results
from core.models import FieldEvidence, OCRBlock, Result


def sample_results():
    return [
        Result(source="a.png", mode="describe", text="hello", raw="hello", latency_ms=120),
        Result(
            source="b.pdf (Page 1)",
            mode="extract",
            text='{"Invoice number":"X-1"}',
            raw='{"Invoice number":"X-1"}',
            fields={"Invoice number": "X-1"},
            page=0,
            page_count=1,
            latency_ms=250,
        ),
    ]


def test_csv_round_trip(tmp_path):
    path = tmp_path / "out.csv"
    write_results(sample_results(), path, format="csv")
    rows = read_results_csv(path)
    assert len(rows) == 2
    assert rows[0]["Filename"] == "a.png"
    assert rows[0]["Description"] == "hello"
    assert rows[1]["Filename"] == "b.pdf (Page 1)"


def test_jsonl_render():
    payload = render_results(sample_results(), format="jsonl")
    lines = [line for line in payload.splitlines() if line]
    assert len(lines) == 2
    assert '"mode": "describe"' in lines[0]
    assert '"Invoice number"' in lines[1]


def test_structured_csv_skips_empty():
    payload = render_results(sample_results(), format="structured_csv")
    # Only the extraction row has fields
    lines = payload.strip().splitlines()
    assert len(lines) == 2  # header + 1 row
    assert "Invoice number" in lines[0]
    assert "X-1" in lines[1]


def test_write_to_stringio_buffer():
    buf = io.StringIO()
    write_results(sample_results(), buf, format="csv")
    assert "a.png" in buf.getvalue()


def test_csv_uses_error_text_when_result_failed():
    payload = render_results(
        [
            Result(
                source="failed.pdf",
                mode="describe",
                text="",
                error="Error: model request failed",
            )
        ],
        format="csv",
    )
    assert "failed.pdf" in payload
    assert "Error: model request failed" in payload


def test_evidence_render_excludes_preview_bytes_and_includes_meta():
    payload = render_evidence(
        [
            Result(
                source="invoice.png",
                mode="extract",
                text='{"total": "42.00"}',
                fields={"total": "42.00"},
                ocr_text="Total 42.00",
                ocr_blocks=[OCRBlock(text="Total 42.00", kind="line", page=0)],
                field_evidence={
                    "total": FieldEvidence(
                        value="42.00",
                        evidence_text="Total 42.00",
                        engine="hybrid",
                        validation_status="present",
                    )
                },
                engine="hybrid",
                profile_id="invoice",
                preprocess_steps=["grayscale"],
                backend_note="docling text plus ollama extraction",
                latency_ms=25,
                page=0,
                page_count=1,
                dimensions=(100, 200),
                encoded_bytes=1234,
                preview_image_bytes=b"binary-preview",
            )
        ]
    )

    data = json.loads(payload)

    assert data["schema_version"] == "localocr.evidence.v1"
    assert data["summary"] == {
        "ok": True,
        "total_results": 1,
        "error_count": 0,
    }
    result = data["results"][0]
    assert "preview_image_bytes" not in result
    assert result["ocr_text"] == "Total 42.00"
    assert result["ocr_blocks"][0]["text"] == "Total 42.00"
    assert result["field_evidence"]["total"]["validation_status"] == "present"
    assert result["engine"] == "hybrid"
    assert result["profile_id"] == "invoice"
    assert result["preprocess_steps"] == ["grayscale"]
    assert result["backend_note"] == "docling text plus ollama extraction"
    assert result["latency_ms"] == 25
    assert result["page"] == 0
    assert result["page_count"] == 1
    assert result["dimensions"] == [100, 200]
    assert result["encoded_bytes"] == 1234
