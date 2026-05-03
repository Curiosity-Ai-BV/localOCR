from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from PIL import Image

import evaluate
from core.models import FieldEvidence, OCRBlock, Result
from core.pipeline import BatchConfig


def _write_dataset(tmp_path: Path, ground_truth: dict[str, dict[str, str]]) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for filename in ground_truth:
        Image.new("RGB", (24, 24), "white").save(dataset / filename)
    (dataset / "ground_truth.json").write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )
    return dataset


def _write_chart(metrics: dict[str, float], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")


def test_evaluate_routes_through_batch_config_and_writes_metrics_and_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "invoice.png": {
                "invoice_number": "INV-42",
                "total": "99.00",
            }
        },
    )
    evidence_path = tmp_path / "evidence.json"
    metrics_path = tmp_path / "metrics.json"
    detailed_csv = tmp_path / "details.csv"
    chart_path = tmp_path / "chart.png"
    seen: dict[str, object] = {}

    def fake_run_batch(jobs, cfg):
        seen["jobs"] = list(jobs)
        seen["cfg"] = cfg
        yield Result(
            source="invoice.png",
            mode="extract",
            text='{"invoice_number": "INV-42", "total": "99.00"}',
            raw='{"invoice_number": "INV-42", "total": "99.00"}',
            fields={"invoice_number": "INV-42", "total": "99.00"},
            ocr_text="Invoice INV-42 total 99.00",
            ocr_blocks=[
                OCRBlock(text="Invoice INV-42 total 99.00", kind="line", page=0)
            ],
            field_evidence={
                "invoice_number": FieldEvidence(
                    value="INV-42",
                    evidence_text="Invoice INV-42",
                    engine="hybrid",
                    validation_status="present",
                )
            },
            engine="hybrid",
            profile_id="invoice",
            preprocess_steps=["grayscale", "autocontrast", "sharpen"],
            backend_note="docling text plus ollama extraction",
        )

    readme_called = False

    def fake_update_readme(metrics, chart) -> None:
        nonlocal readme_called
        readme_called = True

    monkeypatch.setattr(evaluate, "is_ollama_running", lambda: True)
    monkeypatch.setattr(evaluate, "run_batch", fake_run_batch)
    monkeypatch.setattr(evaluate, "create_chart", _write_chart)
    monkeypatch.setattr(evaluate, "update_readme", fake_update_readme)

    rc = evaluate.main(
        [
            "--dataset",
            str(dataset),
            "--model",
            "fake-model",
            "--ocr-backend",
            "hybrid",
            "--profile",
            "invoice",
            "--preprocess",
            "document-clean",
            "--out-evidence",
            str(evidence_path),
            "--metrics-json",
            str(metrics_path),
            "--detailed-csv",
            str(detailed_csv),
            "--chart-output",
            str(chart_path),
        ]
    )

    assert rc == 0
    assert seen["jobs"] == [str(dataset / "invoice.png")]
    cfg = cast(BatchConfig, seen["cfg"])
    assert cfg.model == "fake-model"
    assert cfg.ocr_backend == "hybrid"
    assert cfg.profile_id == "invoice"
    assert cfg.preprocess == "document-clean"
    assert cfg.fields == ["invoice_number", "total"]
    assert readme_called is False

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "localocr.eval.metrics.v1"
    assert metrics["run"]["model"] == "fake-model"
    assert metrics["run"]["ocr_backend"] == "hybrid"
    assert metrics["run"]["profile_id"] == "invoice"
    assert metrics["run"]["preprocess"] == "document-clean"
    assert metrics["run"]["mock"] is False
    assert metrics["run"]["evidence_path"] == str(evidence_path)
    assert metrics["field_accuracy"]["invoice_number"] == {
        "correct": 1,
        "total": 1,
        "accuracy": 1.0,
    }
    assert metrics["aggregate"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    assert metrics["group_metrics"]["backend"]["hybrid"]["accuracy"] == 1.0
    assert metrics["group_metrics"]["profile"]["invoice"]["total"] == 2
    assert metrics["group_metrics"]["preprocess"]["document-clean"]["correct"] == 2
    assert metrics["detailed_rows"][0]["engine"] == "hybrid"
    assert metrics["detailed_rows"][0]["preprocess_steps"] == [
        "grayscale",
        "autocontrast",
        "sharpen",
    ]

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    result = evidence["results"][0]
    assert result["engine"] == "hybrid"
    assert result["profile_id"] == "invoice"
    assert result["ocr_text"] == "Invoice INV-42 total 99.00"
    assert result["field_evidence"]["invoice_number"]["validation_status"] == "present"


def test_mock_evaluation_is_deterministic_and_never_updates_readme(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "ground_truth.json").write_text(
        json.dumps(
            {
                "receipt-a.png": {"merchant_name": "Shop A", "total": "12.50"},
                "receipt-b.png": {"merchant_name": "Shop B", "total": "8.00"},
            }
        ),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"
    evidence_path = tmp_path / "evidence.json"

    def fail_update_readme(metrics, chart) -> None:
        raise AssertionError("mock evaluation must not update README")

    monkeypatch.setattr(evaluate, "is_ollama_running", lambda: False)
    monkeypatch.setattr(evaluate, "create_chart", _write_chart)
    monkeypatch.setattr(evaluate, "update_readme", fail_update_readme)

    argv = [
        "--dataset",
        str(dataset),
        "--model",
        "fake-model",
        "--ocr-backend",
        "ollama",
        "--profile",
        "receipt",
        "--preprocess",
        "none",
        "--allow-mock",
        "--update-readme",
        "--metrics-json",
        str(metrics_path),
        "--out-evidence",
        str(evidence_path),
        "--detailed-csv",
        str(tmp_path / "details.csv"),
        "--chart-output",
        str(tmp_path / "chart.png"),
    ]

    assert evaluate.main(argv) == 0
    first_metrics = metrics_path.read_text(encoding="utf-8")
    first_evidence = evidence_path.read_text(encoding="utf-8")

    assert evaluate.main(argv) == 0

    assert metrics_path.read_text(encoding="utf-8") == first_metrics
    assert evidence_path.read_text(encoding="utf-8") == first_evidence

    metrics = json.loads(first_metrics)
    assert metrics["run"]["mock"] is True
    assert metrics["run"]["ocr_backend"] == "ollama"
    assert metrics["run"]["profile_id"] == "receipt"
    assert metrics["run"]["preprocess"] == "none"
    assert metrics["run"]["evidence_path"] == str(evidence_path)
    assert set(metrics["group_metrics"]["backend"]) == {"ollama"}
    assert len(metrics["detailed_rows"]) == 4

    evidence = json.loads(first_evidence)
    assert evidence["schema_version"] == "localocr.evidence.v1"
    assert evidence["summary"]["total_results"] == 2
    assert evidence["results"][0]["field_evidence"]["merchant_name"]["engine"] == "ollama"


def test_empty_ground_truth_values_are_ignored_from_metric_totals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "invoice.png": {
                "invoice_number": "INV-42",
                "date": "",
                "total": "   ",
            }
        },
    )
    metrics_path = tmp_path / "metrics.json"
    chart_path = tmp_path / "chart.json"

    def fake_run_batch(jobs, cfg):
        yield Result(
            source="invoice.png",
            mode="extract",
            text='{"invoice_number": "INV-42", "date": "", "total": ""}',
            fields={
                "invoice_number": "INV-42",
                "date": "",
                "total": "",
            },
            engine="ollama",
            profile_id="generic",
            preprocess_steps=[],
        )

    monkeypatch.setattr(evaluate, "is_ollama_running", lambda: True)
    monkeypatch.setattr(evaluate, "run_batch", fake_run_batch)
    monkeypatch.setattr(evaluate, "create_chart", _write_chart)

    assert evaluate.main(
        [
            "--dataset",
            str(dataset),
            "--model",
            "fake-model",
            "--metrics-json",
            str(metrics_path),
            "--detailed-csv",
            str(tmp_path / "details.csv"),
            "--chart-output",
            str(chart_path),
        ]
    ) == 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    chart_metrics = json.loads(chart_path.read_text(encoding="utf-8"))

    assert metrics["field_accuracy"]["invoice_number"] == {
        "correct": 1,
        "total": 1,
        "accuracy": 1.0,
    }
    assert metrics["field_accuracy"]["date"] == {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
    }
    assert metrics["field_accuracy"]["total"] == {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
    }
    assert metrics["aggregate"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert metrics["group_metrics"]["backend"]["ollama"] == {
        "correct": 1,
        "total": 1,
        "accuracy": 1.0,
    }
    assert metrics["ignored"] == {
        "total": 2,
        "by_reason": {"empty_ground_truth": 2},
        "by_field": {"date": 1, "total": 1},
    }
    assert chart_metrics == {"invoice_number": 1.0}

    ignored_rows = [
        row
        for row in metrics["detailed_rows"]
        if row["ignored"]
    ]
    assert len(ignored_rows) == 2
    assert {row["field"] for row in ignored_rows} == {"date", "total"}
    assert {row["ignore_reason"] for row in ignored_rows} == {"empty_ground_truth"}
    assert all(row["match"] is None for row in ignored_rows)
