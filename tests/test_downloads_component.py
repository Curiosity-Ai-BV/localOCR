from core.models import Result
from ui.components import downloads


class _Column:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _FakeStreamlit:
    def __init__(self):
        self.downloads = []

    def subheader(self, _label):
        return None

    def columns(self, count):
        return [_Column() for _ in range(count)]

    def download_button(self, label, *, data, file_name, mime, use_container_width):
        self.downloads.append(
            {
                "label": label,
                "data": data,
                "file_name": file_name,
                "mime": mime,
                "use_container_width": use_container_width,
            }
        )


def test_downloads_include_evidence_json_and_existing_exports(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(downloads, "st", fake_st)

    downloads.render_downloads(
        [
            Result(source="a.png", mode="describe", text="hello"),
            Result(
                source="b.png",
                mode="extract",
                text='{"total": "42"}',
                fields={"total": "42"},
            ),
        ]
    )

    labels = [item["label"] for item in fake_st.downloads]
    assert "Results (CSV)" in labels
    assert "Structured (CSV)" in labels
    assert "Results (JSONL)" in labels
    assert "Evidence (JSON)" in labels

    evidence = next(item for item in fake_st.downloads if item["label"] == "Evidence (JSON)")
    assert evidence["mime"] == "application/json"
    assert '"schema_version": "localocr.evidence.v1"' in evidence["data"]
