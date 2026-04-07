from unittest.mock import patch

import pytest

import adapters.ollama_adapter as adapter


@pytest.fixture(autouse=True)
def _clear_cache():
    adapter.clear_model_cache()
    yield
    adapter.clear_model_cache()


def _fake_listing(names):
    return {"models": [{"name": n} for n in names]}


def test_exact_match_via_show():
    with patch.object(adapter.ollama, "show", return_value={"details": {}}):
        ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is True
    assert note is None


def test_exact_match_via_list():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:latest", "llama3.2-vision"]),
        ):
            ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is True
    assert note is None


def test_tag_fallback_warns():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:e4b"]),
        ):
            ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is True
    assert note is not None and "gemma4:e4b" in note


def test_resolve_model_name_returns_matching_tag():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:e4b"]),
        ):
            ok, resolved, note = adapter.resolve_model_name("gemma4:latest")
    assert ok is True
    assert resolved == "gemma4:e4b"
    assert note is not None and "gemma4:e4b" in note


def test_missing_model_reports_candidates():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["llama3.2-vision", "granite3.2-vision"]),
        ):
            ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is False
    assert note is not None
    assert "gemma4" in note
    assert "llama3.2-vision" in note


def test_list_models_cache_ttl():
    calls = {"n": 0}

    def _fake_list():
        calls["n"] += 1
        return _fake_listing(["m1"])

    with patch.object(adapter.ollama, "list", side_effect=_fake_list):
        adapter.list_models()
        adapter.list_models()
        adapter.list_models()
    assert calls["n"] == 1
    adapter.clear_model_cache()
    with patch.object(adapter.ollama, "list", side_effect=_fake_list):
        adapter.list_models()
    assert calls["n"] == 2


def test_substring_no_longer_matches():
    """Regression: 'gemma4' should NOT match 'gemma4:26b' without explicit tag."""
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:26b"]),
        ):
            # target 'gemma4:latest' has different base-tag than candidate,
            # but base() of 'gemma4' (no tag) should match 'gemma4:26b' base.
            ok, note = adapter.ensure_model_available("gemma4")
    assert ok is True  # base match
    assert note and "gemma4:26b" in note
