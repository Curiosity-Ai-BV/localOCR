from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import ollama

from core.errors import ModelUnavailable
from core.logging import get_logger
from core.settings import Settings

_log = get_logger("ollama")

# --- list_models cache -----------------------------------------------------

_CACHE_TTL_SECONDS = 30.0
_cache_lock = threading.Lock()
_cache_value: Optional[List[Dict[str, Any]]] = None
_cache_reachable: Optional[bool] = None
_cache_expires_at: float = 0.0


@dataclass(frozen=True)
class ModelListInventory:
    models: List[Dict[str, Any]]
    reachable: bool


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


def _request_timeout(
    *,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> Optional[float]:
    if timeout is not None:
        return timeout
    if settings is not None:
        return settings.request_timeout
    return None


def _ollama_api(timeout: Optional[float] = None) -> Any:
    if timeout is None:
        return ollama
    return ollama.Client(timeout=timeout)


def list_models_with_status(
    *,
    ttl: Optional[float] = None,
    force_refresh: bool = False,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> ModelListInventory:
    """Return cached Ollama model listing plus whether the list call succeeded."""
    global _cache_value, _cache_reachable, _cache_expires_at
    effective_ttl = ttl
    if effective_ttl is None:
        effective_ttl = settings.model_list_ttl if settings is not None else _CACHE_TTL_SECONDS
    now = time.monotonic()
    with _cache_lock:
        if not force_refresh and _cache_value is not None and now < _cache_expires_at:
            return ModelListInventory(_cache_value, bool(_cache_reachable))
    reachable = True
    try:
        api = _ollama_api(_request_timeout(settings=settings, timeout=timeout))
        listing = api.list()
        raw_models = list(listing.get("models", []))
    except Exception as e:
        _log.warning("list_models_failed", extra={"err": str(e)})
        raw_models = []
        reachable = False
    with _cache_lock:
        _cache_value = raw_models
        _cache_reachable = reachable
        _cache_expires_at = now + effective_ttl
    return ModelListInventory(raw_models, reachable)


def list_models(
    *,
    ttl: Optional[float] = None,
    force_refresh: bool = False,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return the raw ``ollama.list()['models']`` list, cached for ``ttl`` seconds."""
    return list_models_with_status(
        ttl=ttl,
        force_refresh=force_refresh,
        settings=settings,
        timeout=timeout,
    ).models


def _norm(s: str) -> str:
    return s.strip().lower()


def _base(s: str) -> str:
    parts = s.rsplit(":", 1)
    return parts[0] if len(parts) == 2 else s


def resolve_model_name(
    model: str,
    *,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Resolve a usable local model name for callers that will execute requests."""
    request_timeout = _request_timeout(settings=settings, timeout=timeout)
    try:
        _ollama_api(request_timeout).show(model)
        return True, model, None
    except Exception:
        pass

    try:
        raw_models = list_models(settings=settings, timeout=request_timeout)
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


def ensure_model_available(
    model: str,
    *,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """Check for model availability in Ollama.

    Resolution order:
      1. Exact ``ollama.show`` lookup (authoritative).
      2. Exact match against the cached ``list_models()`` output.
      3. Tag-aware fallback: ``base(candidate) == base(target)``.
      4. Otherwise, return ``(False, msg)`` with the list of candidates.

    Substring matching was removed because it silently picked the wrong
    quantization (e.g. ``gemma4:e4b`` vs ``gemma4:26b``).
    """
    ok, _resolved, note = resolve_model_name(model, settings=settings, timeout=timeout)
    return ok, note


def get_available_models(
    defaults: Optional[List[str]] = None,
    *,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> List[str]:
    """Return a list of available model names, preferring locally installed."""
    defaults = defaults or []
    ordered: List[str] = []
    seen: set[str] = set()
    try:
        raw = list_models(settings=settings, timeout=timeout)
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
    format: Optional[str] = None,
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> str:
    """Query Ollama chat with an image, returning content string."""
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt, "images": [image_base64]})

    try:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "options": options or {}}
        if format:
            kwargs["format"] = format
            
        response = _ollama_api(_request_timeout(settings=settings, timeout=timeout)).chat(**kwargs)
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
    settings: Optional[Settings] = None,
    timeout: Optional[float] = None,
) -> Iterator[str]:
    """Yield content chunks from Ollama's streaming chat API."""
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt, "images": [image_base64]})

    try:
        api = _ollama_api(_request_timeout(settings=settings, timeout=timeout))
        for chunk in api.chat(
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
    global _cache_value, _cache_reachable, _cache_expires_at
    with _cache_lock:
        _cache_value = None
        _cache_reachable = None
        _cache_expires_at = 0.0
