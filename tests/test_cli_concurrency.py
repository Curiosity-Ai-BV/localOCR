"""Regression test proving parallel CLI runs do not cross-contaminate prompts.

Before the PromptConfig refactor, ``core.templates`` kept prompts as module
globals mutated via ``set_templates``. Running two CLI invocations in
parallel threads with different templates would interleave writes and
produce wrong prompts per request. This test hammers ``run_batch`` with two
distinct ``PromptConfig`` instances concurrently and asserts every recorded
prompt matches the config that originated the call.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import cli
from core.export import read_results_csv
from core.models import Result
from core.pipeline import BatchConfig, BatchJob, run_batch
from core.prompts import PromptConfig


def _img() -> Image.Image:
    return Image.new("RGB", (32, 32), (0, 255, 0))


def test_parallel_prompts_do_not_cross_contaminate():
    recorded: list[tuple[str, str]] = []
    lock = threading.Lock()

    def make_infer(tag: str):
        def _infer(prompt: str, img_b64: str, model: str) -> str:
            # simulate some work so threads interleave
            with lock:
                recorded.append((tag, prompt))
            return f"ok:{tag}"
        return _infer

    cfg_a = BatchConfig(
        model="fake",
        prompts=PromptConfig(description="PROMPT_A"),
        inference=make_infer("A"),
    )
    cfg_b = BatchConfig(
        model="fake",
        prompts=PromptConfig(description="PROMPT_B"),
        inference=make_infer("B"),
    )

    def run(cfg, count):
        jobs = [BatchJob(source=f"{cfg.prompts.description}-{i}.png", data=_img(), kind="image") for i in range(count)]
        return list(run_batch(jobs, cfg))

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        for _ in range(4):
            futs.append(ex.submit(run, cfg_a, 5))
            futs.append(ex.submit(run, cfg_b, 5))
        for f in futs:
            f.result()

    assert recorded, "no prompts recorded"
    for tag, prompt in recorded:
        expected = "PROMPT_A" if tag == "A" else "PROMPT_B"
        assert prompt == expected, f"cross-contamination: tag={tag} prompt={prompt!r}"


def test_cli_parallel_results_preserve_input_order(tmp_path, monkeypatch):
    slow = tmp_path / "slow.png"
    fast = tmp_path / "fast.png"
    slow.write_bytes(b"slow")
    fast.write_bytes(b"fast")
    out = tmp_path / "results.csv"

    monkeypatch.setattr(cli, "resolve_model_name", lambda model, **kwargs: (True, model, None))

    def fake_run_batch(files, cfg):
        path = Path(files[0])
        if path.name == "slow.png":
            time.sleep(0.05)
        yield Result(source=path.name, mode="describe", text=f"done:{path.name}")

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)

    code = cli.main(
        [
            "--max-concurrency",
            "2",
            "--out-results",
            str(out),
            str(slow),
            str(fast),
        ]
    )

    assert code == 0
    rows = read_results_csv(out)
    assert [row["Filename"] for row in rows] == ["slow.png", "fast.png"]
