from __future__ import annotations

import pytest
from PIL import Image

from core.preprocessing import PREPROCESSING_PRESETS, preprocess_image


def _sample_image() -> Image.Image:
    image = Image.new("RGB", (4, 4), "white")
    pixels = image.load()
    assert pixels is not None
    for x in range(4):
        for y in range(4):
            pixels[x, y] = (x * 30, y * 30, (x + y) * 20)
    return image


def test_preprocessing_presets_do_not_mutate_original_image():
    for preset in PREPROCESSING_PRESETS:
        original = _sample_image()
        before = original.tobytes()

        result = preprocess_image(original, preset)

        assert original.tobytes() == before
        assert result.image is not original
        assert result.metadata["original_size"] == (4, 4)
        assert result.metadata["prepared_size"] == (4, 4)


def test_preprocessing_presets_are_deterministic():
    for preset in PREPROCESSING_PRESETS:
        first = preprocess_image(_sample_image(), preset)
        second = preprocess_image(_sample_image(), preset)

        assert first.steps == second.steps
        assert first.metadata == second.metadata
        assert first.image.mode == second.image.mode
        assert first.image.tobytes() == second.image.tobytes()


def test_none_preset_is_true_no_op_copy():
    original = _sample_image()

    result = preprocess_image(original, "none")

    assert result.image is not original
    assert result.image.mode == original.mode
    assert result.image.size == original.size
    assert result.image.tobytes() == original.tobytes()


def test_preprocessing_presets_report_expected_steps():
    assert preprocess_image(_sample_image(), "none").steps == []
    assert preprocess_image(_sample_image(), "document-clean").steps == [
        "grayscale",
        "autocontrast",
        "sharpen",
    ]
    assert preprocess_image(_sample_image(), "high-accuracy-scan").steps == [
        "grayscale",
        "autocontrast-cutoff-1",
        "unsharp-mask",
    ]


def test_invalid_preprocessing_preset_fails_clearly():
    with pytest.raises(ValueError, match="Unknown preprocessing preset 'blur'"):
        preprocess_image(_sample_image(), "blur")
