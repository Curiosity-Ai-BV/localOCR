from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ollama

from core.errors import ModelUnavailable
from core.logging import get_logger

_log = get_logger("ollama")

# --- list_models cache -----------------------------------------------------

_CACHE_TTL_SECONDS = 30.0
_cache_lock = threading.Lock()
_cache_value: Optional[List[Dict[str, Any]]] = None
_cache_expires_at: float = 0.0


def _iter_model_names(raw_models: Iterable[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for m in raw_models:
        for key in ("name", "model"):
            val = m.get(key)
            if isinstance(val, str) and val not in seen:
                names.append(val)
                seen.add(val)
    return names


def list_models(*, ttl: float = _CACHE_TTL_SECONDS, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Return the raw ``ollama.list()['models']`` list, cached for ``ttl`` seconds."""
    global _cache_value, _cache_expires_at
    now = time.monotonic()
    with _cache_lock:
        if not force_refresh and _cache_value is not None and now < _cache_expires_at:
            return _cache_value
    try:
        listing = ollama.list()
        raw_models = list(listing.get("models", []))
    except Exception as e:
        _log.warning("list_models_failed", extra={"err": str(e)})
        raw_models = []
    with _cache_lock:
        _cache_value = raw_models
        _cache_expires_at = now + ttl
    return raw_models


def _norm(s: str) -> str:
    return s.strip().lower()


def _base(s: str) -> str:
    parts = s.rsplit(":", 1)
    return parts[0] if len(parts) == 2 else s


def resolve_model_name(model: str) -> Tuple[bool, str, Optional[str]]:
    """Resolve a usable local model name for callers that will execute requests."""
    try:
        ollama.show(model)
        return True, model, None
    except Exception:
        pass

    try:
        raw_models = list_models()
    except Exception as e:
        return True, model, f"Could not verify model availability: {e}"

    candidates = _iter_model_names(raw_models)
    normalized = {_norm(name): name for name in candidates}
    target = _norm(model)
    target_base = _base(target)

    exact = normalized.get(target)
    if exact is not None:
        return True, exact, None

    for candidate in candidates:
        if _base(_norm(candidate)) == target_base:
            return True, candidate, (
                f"Exact model '{model}' not found; using closest tag '{candidate}'."
            )

    sample = ", ".join(normalized.keys()) if normalized else "<none>"
    return False, model, (
        f"Model '{model}' not found in Ollama. Available: {sample}. "
        f"Run: ollama pull {model}"
    )


def ensure_model_available(model: str) -> Tuple[bool, Optional[str]]:
    """Check for model availability in Ollama.

    Resolution order:
      1. Exact ``ollama.show`` lookup (authoritative).
      2. Exact match against the cached ``list_models()`` output.
      3. Tag-aware fallback: ``base(candidate) == base(target)``.
      4. Otherwise, return ``(False, msg)`` with the list of candidates.

    Substring matching was removed because it silently picked the wrong
    quantization (e.g. ``gemma4:e4b`` vs ``gemma4:26b``).
    """
    ok, _resolved, note = resolve_model_name(model)
    return ok, note


def get_available_models(defaults: Optional[List[str]] = None) -> List[str]:
    """Return a list of available model names, preferring locally installed."""
    defaults = defaults or []
    ordered: List[str] = []
    seen: set[str] = set()
    try:
        raw = list_models()
        for name in _iter_model_names(raw):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    except Exception:
        pass
    for d in defaults:
        if d not in seen:
            ordered.append(d)
            seen.add(d)
    return ordered or defaults


def query_ollama(
    prompt: str,
    image_base64: str,
    model: str,
    *,
    options: Optional[dict] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Query Ollama chat with an image, returning content string."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt, "images": [image_base64]})

    try:
        response = ollama.chat(model=model, messages=messages, options=options or {})
        content = response.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise ModelUnavailable("Unexpected response content type from model")
        return content
    except ModelUnavailable:
        raise
    except Exception as e:
        raise ModelUnavailable(f"Ollama chat failed: {e}") from e


def query_ollama_stream(
    prompt: str,
    image_base64: str,
    model: str,
    *,
    options: Optional[dict] = None,
    system_prompt: Optional[str] = None,
):
    """Yield content chunks from Ollama's streaming chat API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt, "images": [image_base64]})

    try:
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            options=options or {},
            stream=True,
        ):
            piece = chunk.get("message", {}).get("content", "")
            if isinstance(piece, str) and piece:
                yield piece
    except Exception as e:
        raise ModelUnavailable(f"Ollama stream failed: {e}") from e


def clear_model_cache() -> None:
    """Reset the list_models cache (used by tests)."""
    global _cache_value, _cache_expires_at
    with _cache_lock:
        _cache_value = None
        _cache_expires_at = 0.0
