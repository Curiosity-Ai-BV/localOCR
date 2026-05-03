"""Built-in document profiles for field extraction defaults."""

from __future__ import annotations

from typing import Dict

from core.models import DocumentProfile


BUILTIN_PROFILES: Dict[str, DocumentProfile] = {
    "generic": DocumentProfile(
        id="generic",
        label="Generic document",
        fields=[],
        default_backend=None,
        default_preprocess="none",
    ),
    "invoice": DocumentProfile(
        id="invoice",
        label="Invoice",
        fields=[
            "invoice_number",
            "invoice_date",
            "vendor_name",
            "customer_name",
            "subtotal",
            "tax",
            "total",
        ],
        default_backend=None,
        default_preprocess="high-accuracy-scan",
    ),
    "receipt": DocumentProfile(
        id="receipt",
        label="Receipt",
        fields=[
            "merchant_name",
            "transaction_date",
            "items",
            "tax",
            "total",
            "payment_method",
        ],
        default_backend=None,
        default_preprocess="document-clean",
    ),
    "table": DocumentProfile(
        id="table",
        label="Table",
        fields=["rows", "columns", "headers"],
        default_backend=None,
        default_preprocess="high-accuracy-scan",
    ),
}


def list_profiles() -> list[DocumentProfile]:
    """Return built-in profiles in stable display order."""
    return [_copy_profile(BUILTIN_PROFILES[key]) for key in sorted(BUILTIN_PROFILES)]


def get_profile(profile_id: str) -> DocumentProfile:
    """Return a built-in profile or raise a clear error for unknown ids."""
    normalized_id = profile_id.strip().lower()
    try:
        return _copy_profile(BUILTIN_PROFILES[normalized_id])
    except KeyError as exc:
        valid = ", ".join(sorted(BUILTIN_PROFILES))
        raise ValueError(
            f"Unknown document profile '{profile_id}'. Expected one of: {valid}."
        ) from exc


def _copy_profile(profile: DocumentProfile) -> DocumentProfile:
    return DocumentProfile(
        id=profile.id,
        label=profile.label,
        fields=list(profile.fields),
        default_backend=profile.default_backend,
        default_preprocess=profile.default_preprocess,
    )
