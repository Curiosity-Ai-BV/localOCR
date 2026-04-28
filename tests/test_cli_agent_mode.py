from __future__ import annotations

import json
from pathlib import Path

import cli
from core.export import read_results_csv
from core.models import Result


def test_json_quiet_prints_single_machine_summary(tmp_path, monkeypatch, capsys):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    structured_out = tmp_path / "structured.csv"

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fake_run_batch(files, cfg):
        assert [Path(f).name for f in files] == ["invoice.png"]
        yield Result(
            source="invoice.png",
            mode="extract",
            text='{"total": "42.00"}',
            fields={"total": "42.00"},
            latency_ms=12,
        )

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--mode",
            "extract",
            "--fields",
            "total",
            "--json",
            "--quiet",
            "--out-results",
            str(out),
            "--out-structured",
            str(structured_out),
            str(image),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert payload == {
        "ok": True,
        "processed_files": 1,
        "total_results": 1,
        "results": [
            {
                "source": "invoice.png",
                "mode": "extract",
                "text": '{"total": "42.00"}',
                "fields": {"total": "42.00"},
                "error": None,
                "latency_ms": 12,
            }
        ],
    }


def test_json_without_quiet_prints_single_machine_summary(tmp_path, monkeypatch, capsys):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"

    monkeypatch.setattr(
        cli,
        "resolve_model_name",
        lambda model, **kwargs: (True, model, "Using closest local tag."),
    )

    def fake_run_batch(files, cfg):
        yield Result(
            source="invoice.png",
            mode="describe",
            text="Invoice summary",
            latency_ms=8,
        )

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--json",
            "--out-results",
            str(out),
            str(image),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert captured.out.count("\n") == 1
    assert payload["ok"] is True
    assert payload["processed_files"] == 1
    assert payload["results"][0]["text"] == "Invoice summary"


def test_json_quiet_schema_warning_does_not_break_stdout_json(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing.png"
    missing_schema = tmp_path / "missing-schema.json"
    out = tmp_path / "results.csv"

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fail_run_batch(files, cfg):
        raise AssertionError("missing-only runs should not invoke OCR")

    monkeypatch.setattr(cli, "run_batch", fail_run_batch)

    code = cli.main(
        [
            "--mode",
            "extract",
            "--schema",
            str(missing_schema),
            "--json",
            "--quiet",
            "--out-results",
            str(out),
            str(missing),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 1
    assert captured.out.count("\n") == 1
    assert payload["ok"] is False
    assert "Failed to load schema" in captured.err


def test_cli_model_resolution_uses_runtime_settings(tmp_path, monkeypatch):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    seen = {}

    monkeypatch.setenv("LOCALOCR_REQUEST_TIMEOUT", "6.5")

    def fake_resolve_model_name(model, **kwargs):
        seen["settings"] = kwargs.get("settings")
        return True, model, None

    def fake_run_batch(files, cfg):
        yield Result(source="invoice.png", mode="describe", text="done")

    monkeypatch.setattr(cli, "resolve_model_name", fake_resolve_model_name)
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(["--quiet", "--out-results", str(out), str(image)])

    assert code == 0
    assert seen["settings"] is not None
    assert seen["settings"].request_timeout == 6.5


def test_json_quiet_reports_all_missing_inputs_as_result_rows(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing.png"
    out = tmp_path / "results.csv"

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fail_run_batch(files, cfg):
        raise AssertionError("missing-only runs should not invoke OCR")

    monkeypatch.setattr(cli, "run_batch", fail_run_batch)

    code = cli.main(
        [
            "--json",
            "--quiet",
            "--out-results",
            str(out),
            str(missing),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 1
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["processed_files"] == 1
    assert payload["total_results"] == 1
    assert payload["results"] == [
        {
            "source": "missing.png",
            "mode": "describe",
            "text": "",
            "fields": {},
            "error": f"Input file not found: {missing}",
            "latency_ms": 0,
        }
    ]
    assert read_results_csv(out) == [
        {
            "Filename": "missing.png",
            "Description": f"Input file not found: {missing}",
        }
    ]


def test_human_cli_returns_nonzero_when_all_inputs_are_missing(tmp_path, monkeypatch):
    missing = tmp_path / "missing.png"
    out = tmp_path / "results.csv"

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fail_run_batch(files, cfg):
        raise AssertionError("missing-only runs should not invoke OCR")

    monkeypatch.setattr(cli, "run_batch", fail_run_batch)

    code = cli.main(["--quiet", "--out-results", str(out), str(missing)])

    assert code == 1
    assert read_results_csv(out) == [
        {
            "Filename": "missing.png",
            "Description": f"Input file not found: {missing}",
        }
    ]
