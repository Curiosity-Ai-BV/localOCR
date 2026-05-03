from __future__ import annotations

import pytest

from core.profiles import BUILTIN_PROFILES, get_profile, list_profiles


def test_builtin_profile_selection_works():
    profile = get_profile("invoice")

    assert profile.id == "invoice"
    assert "total" in profile.fields
    assert profile.default_preprocess == "high-accuracy-scan"


def test_generic_profile_preserves_no_preprocessing_default():
    profile = get_profile("generic")

    assert profile.default_preprocess == "none"


def test_profile_selection_normalizes_ids():
    profile = get_profile(" Receipt ")

    assert profile.id == "receipt"


def test_invalid_profile_fails_clearly():
    with pytest.raises(ValueError, match="Unknown document profile 'unknown'"):
        get_profile("unknown")


def test_list_profiles_contains_all_builtins_in_stable_order():
    profiles = list_profiles()

    assert [profile.id for profile in profiles] == sorted(BUILTIN_PROFILES)
    assert {profile.id for profile in profiles} == {
        "generic",
        "invoice",
        "receipt",
        "table",
    }


def test_profile_lookup_returns_mutation_safe_copies():
    profile = get_profile("invoice")
    profile.fields.append("leaked_field")

    fresh_profile = get_profile("invoice")
    listed_profile = next(item for item in list_profiles() if item.id == "invoice")

    assert "leaked_field" not in fresh_profile.fields
    assert "leaked_field" not in listed_profile.fields
