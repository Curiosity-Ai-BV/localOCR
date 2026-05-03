from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cli
from core.export import read_results_csv
from core.models import FieldEvidence, OCRBlock, Result


def test_cli_import_does_not_import_ollama_adapter():
    code = (
        "import sys; "
        "sys.modules.pop('adapters.ollama_adapter', None); "
        "import cli; "
        "raise SystemExit(1 if 'adapters.ollama_adapter' in sys.modules else 0)"
    )

    completed = subprocess.run([sys.executable, "-c", code], check=False)

    assert completed.returncode == 0


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


def test_cli_new_flags_route_to_batch_config(tmp_path, monkeypatch):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    seen = {}

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fake_run_batch(files, cfg):
        seen["files"] = [Path(f).name for f in files]
        seen["backend"] = cfg.ocr_backend
        seen["profile"] = cfg.profile_id
        seen["preprocess"] = cfg.preprocess
        yield Result(source="invoice.png", mode="describe", text="done")

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--quiet",
            "--ocr-backend",
            "hybrid",
            "--profile",
            "invoice",
            "--preprocess",
            "document-clean",
            "--out-results",
            str(out),
            str(image),
        ]
    )

    assert code == 0
    assert seen == {
        "files": ["invoice.png"],
        "backend": "hybrid",
        "profile": "invoice",
        "preprocess": "document-clean",
    }


def test_cli_docling_backend_skips_ollama_model_resolution(tmp_path, monkeypatch):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"

    def fail_resolve_model_name(model, **kwargs):
        raise AssertionError("pure docling must not preflight Ollama")

    def fake_run_batch(files, cfg):
        assert cfg.ocr_backend == "docling"
        assert cfg.model
        yield Result(
            source="invoice.png",
            mode="describe",
            text="Docling text",
            engine="docling",
        )

    monkeypatch.setattr(cli, "resolve_model_name", fail_resolve_model_name)
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--quiet",
            "--ocr-backend",
            "docling",
            "--out-results",
            str(out),
            str(image),
        ]
    )

    assert code == 0
    assert read_results_csv(out)[0]["Description"] == "Docling text"


def test_cli_description_profile_does_not_enable_profile_extraction(tmp_path, monkeypatch):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    seen = {}

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fake_run_batch(files, cfg):
        seen["fields"] = cfg.fields
        seen["use_profile_fields"] = cfg.use_profile_fields
        yield Result(source="invoice.png", mode="describe", text="Invoice summary")

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--quiet",
            "--mode",
            "description",
            "--profile",
            "invoice",
            "--out-results",
            str(out),
            str(image),
        ]
    )

    assert code == 0
    assert seen == {"fields": None, "use_profile_fields": False}


def test_cli_hybrid_description_skips_ollama_model_resolution(tmp_path, monkeypatch):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    seen = {}

    def fail_resolve_model_name(model, **kwargs):
        raise AssertionError("hybrid description without fields should not preflight Ollama")

    def fake_run_batch(files, cfg):
        seen["fields"] = cfg.fields
        seen["backend"] = cfg.ocr_backend
        yield Result(
            source="invoice.png",
            mode="describe",
            text="Docling text",
            engine="hybrid",
        )

    monkeypatch.setattr(cli, "resolve_model_name", fail_resolve_model_name)
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--quiet",
            "--ocr-backend",
            "hybrid",
            "--profile",
            "invoice",
            "--out-results",
            str(out),
            str(image),
        ]
    )

    assert code == 0
    assert seen == {"fields": None, "backend": "hybrid"}


def test_cli_hybrid_profile_extraction_preflights_profile_fields(tmp_path, monkeypatch):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    structured_out = tmp_path / "structured.csv"
    seen = {}

    def fake_resolve_model_name(model, **kwargs):
        seen["resolved_model"] = model
        return True, model, None

    def fake_run_batch(files, cfg):
        seen["fields"] = cfg.fields
        seen["use_profile_fields"] = cfg.use_profile_fields
        yield Result(
            source="invoice.png",
            mode="extract",
            text='{"invoice_number": "INV-42"}',
            fields={"invoice_number": "INV-42"},
            engine="hybrid",
        )

    monkeypatch.setattr(cli, "resolve_model_name", fake_resolve_model_name)
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--quiet",
            "--mode",
            "extract",
            "--ocr-backend",
            "hybrid",
            "--profile",
            "invoice",
            "--out-results",
            str(out),
            "--out-structured",
            str(structured_out),
            str(image),
        ]
    )

    assert code == 0
    assert seen["resolved_model"]
    assert "invoice_number" in seen["fields"]
    assert seen["use_profile_fields"] is True


def test_cli_docling_backend_does_not_import_ollama_adapter(tmp_path, monkeypatch):
    existing_ollama = sys.modules.get("adapters.ollama_adapter")
    sys.modules.pop("adapters.ollama_adapter", None)
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"

    def fail_resolve_model_name(model, **kwargs):
        raise AssertionError("pure docling must not resolve Ollama models")

    def fake_run_batch(files, cfg):
        yield Result(
            source="invoice.png",
            mode="describe",
            text="Docling text",
            engine="docling",
        )

    monkeypatch.setattr(cli, "resolve_model_name", fail_resolve_model_name)
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    try:
        code = cli.main(
            [
                "--quiet",
                "--ocr-backend",
                "docling",
                "--out-results",
                str(out),
                str(image),
            ]
        )

        assert code == 0
        assert "adapters.ollama_adapter" not in sys.modules
    finally:
        if existing_ollama is not None:
            sys.modules["adapters.ollama_adapter"] = existing_ollama
        else:
            sys.modules.pop("adapters.ollama_adapter", None)


def test_cli_out_evidence_does_not_pollute_json_stdout(tmp_path, monkeypatch, capsys):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not-real-image")
    out = tmp_path / "results.csv"
    structured_out = tmp_path / "structured.csv"
    evidence = tmp_path / "evidence.json"

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fake_run_batch(files, cfg):
        yield Result(
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
                    engine="ollama",
                    validation_status="present",
                )
            },
            engine="ollama",
            profile_id="invoice",
            preprocess_steps=["grayscale"],
            backend_note="note",
            latency_ms=12,
            preview_image_bytes=b"preview",
        )

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--json",
            "--quiet",
            "--out-results",
            str(out),
            "--out-structured",
            str(structured_out),
            "--out-evidence",
            str(evidence),
            str(image),
        ]
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))

    assert code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert stdout_payload == {
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
    assert evidence_payload["schema_version"] == "localocr.evidence.v1"
    assert evidence_payload["results"][0]["ocr_text"] == "Total 42.00"
    assert evidence_payload["results"][0]["field_evidence"]["total"]["value"] == "42.00"
    assert "preview_image_bytes" not in evidence_payload["results"][0]
