from PIL import Image

from core.models import Result
from core.pipeline import BatchConfig, BatchJob, run_batch
from core.prompts import PromptConfig


def _img(color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (64, 64), color)


def fake_infer_describe(prompt, img_b64, model):
    assert "Describe" in prompt or "describe" in prompt.lower()
    return "a red square"


def fake_infer_extract(prompt, img_b64, model):
    assert "Invoice" in prompt or "invoice" in prompt.lower()
    return '{"Invoice number": "INV-42", "Total": "99.00"}'


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


def test_run_batch_keeps_pdf_preview_bytes(monkeypatch):
    def fake_process_pdf(*args, **kwargs):
        yield 0, 1, _img((0, 255, 0)), "sample.pdf (Page 1)", "page text", None, 0.05, (64, 64), 123

    monkeypatch.setattr("core.pipeline.process_pdf", fake_process_pdf)

    jobs = [BatchJob(source="sample.pdf", data=b"%PDF-1.4", kind="pdf")]
    cfg = BatchConfig(model="fake", inference=fake_infer_describe)
    results = list(run_batch(jobs, cfg))

    assert len(results) == 1
    assert results[0].source == "sample.pdf (Page 1)"
    assert results[0].preview_image_bytes is not None
