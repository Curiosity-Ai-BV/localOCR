import subprocess
import sys

from PIL import Image

from core.models import Result
from core.pipeline import BatchConfig, BatchJob, run_batch
from core.prompts import PromptConfig
from core.settings import Settings


def _img(color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (64, 64), color)


def fake_infer_describe(prompt, img_b64, model):
    assert "Describe" in prompt or "describe" in prompt.lower()
    return "a red square"


def fake_infer_extract(prompt, img_b64, model):
    assert "Invoice" in prompt or "invoice" in prompt.lower()
    return '{"Invoice number": "INV-42", "Total": "99.00"}'


def test_pipeline_import_does_not_import_ollama_adapter():
    code = (
        "import sys; "
        "sys.modules.pop('adapters.ollama_adapter', None); "
        "import core.pipeline; "
        "raise SystemExit(1 if 'adapters.ollama_adapter' in sys.modules else 0)"
    )

    completed = subprocess.run([sys.executable, "-c", code], check=False)

    assert completed.returncode == 0


def test_run_batch_describe_image():
    jobs = [BatchJob(source="one.png", data=_img(), kind="image")]
    cfg = BatchConfig(model="fake", inference=fake_infer_describe)
    results = list(run_batch(jobs, cfg))
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, Result)
    assert r.mode == "describe"
    assert r.text == "a red square"
    assert r.error is None
    assert r.dimensions == (64, 64)
    assert r.preview_image_bytes is not None


def test_batch_config_defaults_to_ollama_generic_no_preprocess():
    cfg = BatchConfig()

    assert cfg.ocr_backend == "ollama"
    assert cfg.profile_id == "generic"
    assert cfg.preprocess == "none"
    assert cfg.model == cfg.settings.default_model
    assert cfg.model == "deepseek-ocr:latest"


def test_batch_config_profile_default_preprocess_applies_when_omitted():
    cfg = BatchConfig(profile_id="invoice")

    assert cfg.profile_id == "invoice"
    assert cfg.preprocess == "high-accuracy-scan"


def test_batch_config_explicit_none_preprocess_overrides_profile_default():
    cfg = BatchConfig(profile_id="invoice", preprocess="none")

    assert cfg.profile_id == "invoice"
    assert cfg.preprocess == "none"


def test_run_batch_default_backend_preserves_ollama_result_with_metadata():
    jobs = [BatchJob(source="one.png", data=_img(), kind="image")]
    cfg = BatchConfig(model="fake", inference=fake_infer_describe)

    results = list(run_batch(jobs, cfg))

    assert len(results) == 1
    assert results[0].text == "a red square"
    assert results[0].error is None
    assert results[0].engine == "ollama"
    assert results[0].profile_id == "generic"
    assert results[0].preprocess_steps == []


def test_run_batch_docling_backend_errors_without_running_ollama(monkeypatch):
    called = False

    def fake_infer(prompt, img_b64, model):
        nonlocal called
        called = True
        return "should not run"

    def fake_docling(job):
        raise RuntimeError("Docling is not configured")

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    jobs = [BatchJob(source="one.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake",
        ocr_backend="docling",
        inference=fake_infer,
        pdf_pages_separately=False,
    )

    results = list(run_batch(jobs, cfg))

    assert called is False
    assert len(results) == 1
    assert results[0].text == ""
    assert results[0].engine == "docling"
    assert results[0].profile_id == "generic"
    assert results[0].error is not None
    assert "Docling is not configured" in results[0].error


def test_run_batch_docling_pdf_uses_adapter_without_ollama(monkeypatch):
    called = False

    def fake_infer(prompt, img_b64, model):
        nonlocal called
        called = True
        return "should not run"

    def fake_docling(job):
        assert job.kind == "pdf"
        return Result(
            source=job.source,
            mode="describe",
            text="Docling OCR text",
            raw="Docling OCR text",
            ocr_text="Docling OCR text",
            engine="docling",
        )

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    jobs = [BatchJob(source="sample.pdf", data=b"%PDF-1.4", kind="pdf")]
    cfg = BatchConfig(
        model="fake",
        ocr_backend="docling",
        inference=fake_infer,
        pdf_pages_separately=False,
    )

    result = list(run_batch(jobs, cfg))[0]

    assert called is False
    assert result.mode == "describe"
    assert result.text == "Docling OCR text"
    assert result.raw == "Docling OCR text"
    assert result.ocr_text == "Docling OCR text"
    assert result.engine == "docling"
    assert result.profile_id == "generic"
    assert result.preprocess_steps == []
    assert result.error is None


def test_run_batch_docling_pdf_pages_fan_out_through_docling_images(monkeypatch):
    seen_sources = []

    def fake_iter_pdf_pages(file_bytes, *, scale):
        assert file_bytes == b"%PDF"
        yield 0, 2, _img((255, 255, 255))
        yield 1, 2, _img((0, 0, 0))

    def fake_docling(job):
        assert job.kind == "image"
        seen_sources.append(job.source)
        return Result(
            source=job.source,
            mode="describe",
            text=f"Docling OCR text for {job.source}",
            raw=f"Docling OCR text for {job.source}",
            ocr_text=f"Docling OCR text for {job.source}",
            engine="docling",
        )

    monkeypatch.setattr("core.pipeline.iter_pdf_pages", fake_iter_pdf_pages)
    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    cfg = BatchConfig(model="fake", ocr_backend="docling", pdf_pages_separately=True)
    results = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )

    assert seen_sources == ["sample.pdf (Page 1)", "sample.pdf (Page 2)"]
    assert [result.source for result in results] == seen_sources
    assert [result.page for result in results] == [0, 1]
    assert [result.page_count for result in results] == [2, 2]
    assert all(result.engine == "docling" for result in results)


def test_run_batch_docling_pdf_does_not_import_ollama(monkeypatch):
    existing_ollama = sys.modules.get("adapters.ollama_adapter")
    sys.modules.pop("adapters.ollama_adapter", None)

    def fake_docling(job):
        return Result(
            source=job.source,
            mode="describe",
            text="Docling only",
            raw="Docling only",
            ocr_text="Docling only",
            engine="docling",
        )

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    try:
        cfg = BatchConfig(
            model="fake",
            ocr_backend="docling",
            pdf_pages_separately=False,
        )
        result = list(
            run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
        )[0]

        assert result.text == "Docling only"
        assert result.engine == "docling"
        assert "adapters.ollama_adapter" not in sys.modules
    finally:
        if existing_ollama is not None:
            sys.modules["adapters.ollama_adapter"] = existing_ollama
        else:
            sys.modules.pop("adapters.ollama_adapter", None)


def test_run_batch_docling_with_fields_returns_clear_error(monkeypatch):
    called_ollama = False
    called_docling = False

    def fake_infer(prompt, img_b64, model):
        nonlocal called_ollama
        called_ollama = True
        return "{}"

    def fake_docling(job):
        nonlocal called_docling
        called_docling = True
        return Result(source=job.source, mode="describe", text="Docling text")

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="docling",
        fields=["Invoice number"],
        inference=fake_infer,
    )
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert called_ollama is False
    assert called_docling is False
    assert result.mode == "extract"
    assert result.engine == "docling"
    assert result.fields == {}
    assert result.error is not None
    assert "Use ocr_backend='hybrid' for field extraction" in result.error


def test_run_batch_extract_image_parses_fields():
    jobs = [BatchJob(source="inv.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake",
        fields=["Invoice number", "Total"],
        inference=fake_infer_extract,
    )
    results = list(run_batch(jobs, cfg))
    assert len(results) == 1
    r = results[0]
    assert r.mode == "extract"
    assert r.fields.get("Invoice number") == "INV-42"
    assert r.fields.get("Total") == "99.00"


def test_run_batch_error_captured():
    def boom(prompt, img_b64, model):
        raise RuntimeError("boom")

    jobs = [BatchJob(source="bad.png", data=_img(), kind="image")]
    cfg = BatchConfig(model="fake", inference=boom)
    results = list(run_batch(jobs, cfg))
    assert len(results) == 1
    assert results[0].error is not None
    assert "boom" in results[0].error


def test_run_batch_uses_prompt_config():
    seen = {}

    def capture(prompt, img_b64, model):
        seen["prompt"] = prompt
        return "ok"

    prompts = PromptConfig(description="CUSTOM_PROMPT_MARKER")
    cfg = BatchConfig(model="fake", inference=capture, prompts=prompts)
    jobs = [BatchJob(source="x.png", data=_img(), kind="image")]
    list(run_batch(jobs, cfg))
    assert seen["prompt"] == "CUSTOM_PROMPT_MARKER"


def test_run_batch_passes_settings_to_default_inference(monkeypatch):
    seen = {}

    def fake_query(prompt, img_b64, model, **kwargs):
        seen["settings"] = kwargs.get("settings")
        return "ok"

    monkeypatch.setattr("core.pipeline.query_ollama", fake_query)

    settings = Settings(request_timeout=7.0)
    cfg = BatchConfig(model="fake", settings=settings)
    jobs = [BatchJob(source="x.png", data=_img(), kind="image")]

    results = list(run_batch(jobs, cfg))

    assert results[0].text == "ok"
    assert seen["settings"] is settings


def test_run_batch_extraction_passes_schema_format_to_ollama(monkeypatch):
    seen = {}

    def fake_query(prompt, img_b64, model, **kwargs):
        seen["format"] = kwargs.get("format")
        return '{"Invoice number": "INV-42"}'

    monkeypatch.setattr("core.pipeline.query_ollama", fake_query)

    cfg = BatchConfig(model="fake", fields=["Invoice number"])
    jobs = [BatchJob(source="x.png", data=_img(), kind="image")]

    results = list(run_batch(jobs, cfg))

    assert results[0].fields == {"Invoice number": "INV-42"}
    assert seen["format"]["type"] == "object"
    assert "Invoice number" in seen["format"]["properties"]


def test_run_batch_with_fields_populates_fields_and_evidence_separately():
    jobs = [BatchJob(source="inv.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake",
        fields=["Invoice number", "Total"],
        inference=fake_infer_extract,
    )

    result = list(run_batch(jobs, cfg))[0]

    assert result.fields == {"Invoice number": "INV-42", "Total": "99.00"}
    assert result.field_evidence["Invoice number"].value == "INV-42"
    assert result.field_evidence["Invoice number"].engine == "ollama"
    assert result.field_evidence["Invoice number"].validation_status == "present"
    assert result.field_evidence["Invoice number"].confidence is None


def test_run_batch_missing_requested_field_gets_evidence_only():
    def missing_total(prompt, img_b64, model):
        return '{"Invoice number": "INV-42"}'

    jobs = [BatchJob(source="inv.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake",
        fields=["Invoice number", "Total"],
        inference=missing_total,
    )

    result = list(run_batch(jobs, cfg))[0]

    assert result.fields == {"Invoice number": "INV-42"}
    assert "Total" not in result.fields
    assert result.field_evidence["Total"].value is None
    assert result.field_evidence["Total"].validation_status == "missing"


def test_run_batch_removes_missing_requested_values_from_fields_only():
    def missing_values(prompt, img_b64, model):
        return (
            '{"Present": "ok", "None value": null, "Empty": "", '
            '"Blank": "   ", "Non-requested blank": "   "}'
        )

    jobs = [BatchJob(source="inv.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake",
        fields=["Present", "None value", "Empty", "Blank"],
        inference=missing_values,
    )

    result = list(run_batch(jobs, cfg))[0]

    assert result.fields == {"Present": "ok", "Non-requested blank": "   "}
    for field in ["None value", "Empty", "Blank"]:
        assert field not in result.fields
        assert result.field_evidence[field].value is None
        assert result.field_evidence[field].validation_status == "missing"
    assert result.field_evidence["Present"].value == "ok"
    assert result.field_evidence["Present"].validation_status == "present"


def test_invoice_profile_without_fields_uses_profile_fields_for_extraction():
    def invoice_profile_extract(prompt, img_b64, model):
        assert "invoice_number" in prompt
        return '{"invoice_number": "INV-42", "total": "99.00"}'

    jobs = [BatchJob(source="inv.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake", profile_id="invoice", inference=invoice_profile_extract
    )

    result = list(run_batch(jobs, cfg))[0]

    assert cfg.fields is not None
    assert "invoice_number" in cfg.fields
    assert result.mode == "extract"
    assert result.fields["invoice_number"] == "INV-42"
    assert result.field_evidence["invoice_number"].validation_status == "present"


def test_profile_fields_can_be_disabled_for_explicit_description_mode():
    jobs = [BatchJob(source="one.png", data=_img(), kind="image")]
    cfg = BatchConfig(
        model="fake",
        profile_id="invoice",
        inference=fake_infer_describe,
        use_profile_fields=False,
    )

    result = list(run_batch(jobs, cfg))[0]

    assert cfg.fields is None
    assert result.mode == "describe"
    assert result.text == "a red square"
    assert result.fields == {}


def test_generic_profile_without_fields_remains_describe():
    jobs = [BatchJob(source="one.png", data=_img(), kind="image")]
    cfg = BatchConfig(model="fake", profile_id="generic", inference=fake_infer_describe)

    result = list(run_batch(jobs, cfg))[0]

    assert cfg.fields is None
    assert result.mode == "describe"
    assert result.fields == {}
    assert result.field_evidence == {}


def test_run_batch_keeps_pdf_preview_bytes(monkeypatch):
    def fake_process_pdf(*args, **kwargs):
        yield (
            0,
            1,
            _img((0, 255, 0)),
            "sample.pdf (Page 1)",
            "page text",
            None,
            0.05,
            (64, 64),
            123,
        )

    monkeypatch.setattr("core.pipeline.process_pdf", fake_process_pdf)

    jobs = [BatchJob(source="sample.pdf", data=b"%PDF-1.4", kind="pdf")]
    cfg = BatchConfig(model="fake", inference=fake_infer_describe)
    results = list(run_batch(jobs, cfg))

    assert len(results) == 1
    assert results[0].source == "sample.pdf (Page 1)"
    assert results[0].preview_image_bytes is not None


def test_run_batch_pdf_reports_preprocess_steps_without_pdf_tuple_change(monkeypatch):
    def fake_process_pdf(*args, **kwargs):
        assert kwargs["preprocess"] == "document-clean"
        page_tuple = (
            0,
            1,
            _img((0, 255, 0)),
            "sample.pdf (Page 1)",
            "page text",
            None,
            0.05,
            (64, 64),
            123,
        )
        assert len(page_tuple) == 9
        yield page_tuple

    monkeypatch.setattr("core.pipeline.process_pdf", fake_process_pdf)

    cfg = BatchConfig(
        model="fake",
        preprocess="document-clean",
        inference=fake_infer_describe,
    )
    image_result = list(
        run_batch([BatchJob(source="sample.png", data=_img(), kind="image")], cfg)
    )[0]
    pdf_result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF-1.4", kind="pdf")], cfg)
    )[0]

    assert image_result.preprocess_steps == ["grayscale", "autocontrast", "sharpen"]
    assert pdf_result.preprocess_steps == image_result.preprocess_steps


def test_hybrid_extract_uses_docling_text_and_ollama_fields(monkeypatch):
    seen = {}

    def fake_docling(job):
        return Result(
            source=job.source,
            mode="describe",
            text="Invoice INV-42 total 99.00",
            raw="Invoice INV-42 total 99.00",
            ocr_text="Invoice INV-42 total 99.00",
            engine="docling",
        )

    def fake_text_query(prompt, model, **kwargs):
        seen["prompt"] = prompt
        seen["format"] = kwargs.get("format")
        return '{"Invoice number": "INV-42", "Total": "99.00"}'

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.query_ollama_text", fake_text_query)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="hybrid",
        fields=["Invoice number", "Total"],
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert "Invoice INV-42 total 99.00" in seen["prompt"]
    assert seen["format"]["type"] == "object"
    assert result.mode == "extract"
    assert result.engine == "hybrid"
    assert result.ocr_text == "Invoice INV-42 total 99.00"
    assert result.preprocess_steps == []
    assert result.fields == {"Invoice number": "INV-42", "Total": "99.00"}
    assert result.field_evidence["Invoice number"].engine == "hybrid"


def test_hybrid_pdf_pages_fan_out_before_text_extraction(monkeypatch):
    seen_docling_sources = []
    seen_prompts = []

    def fake_iter_pdf_pages(file_bytes, *, scale):
        assert file_bytes == b"%PDF"
        yield 0, 2, _img((255, 255, 255))
        yield 1, 2, _img((0, 0, 0))

    def fake_docling(job):
        assert job.kind == "image"
        seen_docling_sources.append(job.source)
        return Result(
            source=job.source,
            mode="describe",
            text=f"OCR text for {job.source}",
            raw=f"OCR text for {job.source}",
            ocr_text=f"OCR text for {job.source}",
            engine="docling",
        )

    def fake_text_query(prompt, model, **kwargs):
        seen_prompts.append(prompt)
        return '{"Invoice number": "INV-42"}'

    monkeypatch.setattr("core.pipeline.iter_pdf_pages", fake_iter_pdf_pages)
    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.query_ollama_text", fake_text_query)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="hybrid",
        fields=["Invoice number"],
        pdf_pages_separately=True,
    )
    results = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )

    assert seen_docling_sources == ["sample.pdf (Page 1)", "sample.pdf (Page 2)"]
    assert len(seen_prompts) == 2
    assert [result.source for result in results] == seen_docling_sources
    assert [result.page for result in results] == [0, 1]
    assert [result.page_count for result in results] == [2, 2]
    assert all(result.mode == "extract" for result in results)
    assert all(result.engine == "hybrid" for result in results)
    assert all(result.fields == {"Invoice number": "INV-42"} for result in results)


def test_hybrid_with_image_inference_hook_uses_text_query(monkeypatch):
    called_image_inference = False

    def fake_docling(job):
        return Result(
            source=job.source,
            mode="describe",
            text="Invoice INV-42",
            raw="Invoice INV-42",
            ocr_text="Invoice INV-42",
            engine="docling",
        )

    def fake_image_inference(prompt, img_b64, model):
        nonlocal called_image_inference
        called_image_inference = True
        return '{"Invoice number": "wrong"}'

    def fake_text_query(prompt, model, **kwargs):
        return '{"Invoice number": "INV-42"}'

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.query_ollama_text", fake_text_query)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="hybrid",
        fields=["Invoice number"],
        inference=fake_image_inference,
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert called_image_inference is False
    assert result.fields == {"Invoice number": "INV-42"}


def test_hybrid_describe_without_fields_returns_docling_without_ollama(monkeypatch):
    called = False

    def fake_docling(job):
        return Result(
            source=job.source,
            mode="describe",
            text="Docling description text",
            raw="Docling description text",
            ocr_text="Docling description text",
            engine="docling",
        )

    def fake_text_query(prompt, model, **kwargs):
        nonlocal called
        called = True
        return "{}"

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.query_ollama_text", fake_text_query)

    cfg = BatchConfig(model="fake", ocr_backend="hybrid", pdf_pages_separately=False)
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert called is False
    assert result.mode == "describe"
    assert result.text == "Docling description text"
    assert result.ocr_text == "Docling description text"
    assert result.preprocess_steps == []
    assert result.fields == {}


def test_auto_pdf_with_fields_uses_hybrid_when_docling_succeeds(monkeypatch):
    def fake_docling(job):
        return Result(
            source=job.source,
            mode="describe",
            text="Invoice INV-42 total 99.00",
            raw="Invoice INV-42 total 99.00",
            ocr_text="Invoice INV-42 total 99.00",
            engine="docling",
        )

    def fake_text_query(prompt, model, **kwargs):
        return '{"Invoice number": "INV-42", "Total": "99.00"}'

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.query_ollama_text", fake_text_query)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="auto",
        fields=["Invoice number", "Total"],
        inference=fake_infer_extract,
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert result.mode == "extract"
    assert result.engine == "hybrid"
    assert result.fields == {"Invoice number": "INV-42", "Total": "99.00"}
    assert result.backend_note is None
    assert result.preprocess_steps == []


def test_auto_pdf_falls_back_to_ollama_when_docling_unavailable(monkeypatch):
    def fake_docling(job):
        raise RuntimeError("Docling unavailable")

    def fake_process_pdf(*args, **kwargs):
        yield (
            0,
            1,
            _img(),
            "sample.pdf (Page 1)",
            "ollama pdf text",
            None,
            0.01,
            (64, 64),
            456,
        )

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.process_pdf", fake_process_pdf)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="auto",
        inference=fake_infer_describe,
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert result.text == "ollama pdf text"
    assert result.engine == "ollama"
    assert result.backend_note == "auto fallback from docling: Docling unavailable"
    assert result.error is None


def test_auto_pdf_page_docling_failure_falls_back_without_duplicate_partial_results(
    monkeypatch,
):
    def fake_iter_pdf_pages(file_bytes, *, scale):
        assert file_bytes == b"%PDF"
        yield 0, 2, _img((255, 255, 255))
        yield 1, 2, _img((0, 0, 0))

    def fake_docling(job):
        if "Page 2" in job.source:
            raise RuntimeError("Docling page failure")
        return Result(
            source=job.source,
            mode="describe",
            text=f"Docling text for {job.source}",
            raw=f"Docling text for {job.source}",
            ocr_text=f"Docling text for {job.source}",
            engine="docling",
        )

    def fake_process_pdf(*args, **kwargs):
        yield (
            0,
            2,
            _img(),
            "sample.pdf (Page 1)",
            "ollama page 1",
            None,
            0.01,
            (64, 64),
            456,
        )
        yield (
            1,
            2,
            _img(),
            "sample.pdf (Page 2)",
            "ollama page 2",
            None,
            0.01,
            (64, 64),
            456,
        )

    monkeypatch.setattr("core.pipeline.iter_pdf_pages", fake_iter_pdf_pages)
    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.process_pdf", fake_process_pdf)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="auto",
        inference=fake_infer_describe,
        pdf_pages_separately=True,
    )
    results = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )

    assert [result.source for result in results] == [
        "sample.pdf (Page 1)",
        "sample.pdf (Page 2)",
    ]
    assert [result.text for result in results] == ["ollama page 1", "ollama page 2"]
    assert all(result.engine == "ollama" for result in results)
    assert all(
        result.backend_note == "auto fallback from docling: Docling page failure"
        for result in results
    )


def test_auto_pdf_with_fields_falls_back_to_ollama_extraction(monkeypatch):
    def fake_docling(job):
        raise RuntimeError("Docling unavailable")

    def fake_process_pdf(*args, **kwargs):
        assert kwargs["preprocess"] == "document-clean"
        yield (
            0,
            1,
            _img(),
            "sample.pdf (Page 1)",
            '{"Invoice number": "INV-42"}',
            {"filename": "sample.pdf (Page 1)", "Invoice number": "INV-42"},
            0.01,
            (64, 64),
            456,
        )

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.process_pdf", fake_process_pdf)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="auto",
        fields=["Invoice number"],
        preprocess="document-clean",
        inference=fake_infer_extract,
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="sample.pdf", data=b"%PDF", kind="pdf")], cfg)
    )[0]

    assert result.mode == "extract"
    assert result.engine == "ollama"
    assert result.fields == {"Invoice number": "INV-42"}
    assert result.backend_note == "auto fallback from docling: Docling unavailable"
    assert result.preprocess_steps == ["grayscale", "autocontrast", "sharpen"]


def test_auto_image_uses_ollama_without_docling(monkeypatch):
    called = False

    def fake_docling(job):
        nonlocal called
        called = True
        raise AssertionError("Docling should not be used for auto image")

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    cfg = BatchConfig(model="fake", ocr_backend="auto", inference=fake_infer_describe)
    result = list(
        run_batch([BatchJob(source="sample.png", data=_img(), kind="image")], cfg)
    )[0]

    assert called is False
    assert result.text == "a red square"
    assert result.engine == "ollama"


def test_default_ollama_path_does_not_import_docling_adapter(monkeypatch):
    sys.modules.pop("adapters.docling_adapter", None)

    cfg = BatchConfig(model="fake", inference=fake_infer_describe)
    result = list(
        run_batch([BatchJob(source="sample.png", data=_img(), kind="image")], cfg)
    )[0]

    assert result.engine == "ollama"
    assert "adapters.docling_adapter" not in sys.modules


def test_docling_image_route_applies_selected_preprocessing(monkeypatch):
    seen = {}

    def fake_docling(job):
        seen["mode"] = job.data.mode
        seen["source"] = job.source
        return Result(
            source=job.source,
            mode="describe",
            text="Docling OCR text",
            raw="Docling OCR text",
            ocr_text="Docling OCR text",
            engine="docling",
            dimensions=job.data.size,
        )

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="docling",
        preprocess="document-clean",
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="scan.png", data=_img(), kind="image")], cfg)
    )[0]

    assert seen == {"mode": "L", "source": "scan.png"}
    assert result.text == "Docling OCR text"
    assert result.preprocess_steps == ["grayscale", "autocontrast", "sharpen"]


def test_hybrid_image_route_applies_selected_preprocessing(monkeypatch):
    seen = {}

    def fake_docling(job):
        seen["mode"] = job.data.mode
        return Result(
            source=job.source,
            mode="describe",
            text="Invoice INV-42",
            raw="Invoice INV-42",
            ocr_text="Invoice INV-42",
            engine="docling",
            dimensions=job.data.size,
        )

    def fake_text_query(prompt, model, **kwargs):
        return '{"Invoice number": "INV-42"}'

    monkeypatch.setattr("core.pipeline._convert_with_docling", fake_docling)
    monkeypatch.setattr("core.pipeline.query_ollama_text", fake_text_query)

    cfg = BatchConfig(
        model="fake",
        ocr_backend="hybrid",
        fields=["Invoice number"],
        preprocess="high-accuracy-scan",
        pdf_pages_separately=False,
    )
    result = list(
        run_batch([BatchJob(source="scan.png", data=_img(), kind="image")], cfg)
    )[0]

    assert seen == {"mode": "L"}
    assert result.fields == {"Invoice number": "INV-42"}
    assert result.preprocess_steps == [
        "grayscale",
        "autocontrast-cutoff-1",
        "unsharp-mask",
    ]
