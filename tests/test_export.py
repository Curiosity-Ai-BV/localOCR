import io

from core.export import read_results_csv, render_results, write_results
from core.models import Result


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
