from __future__ import annotations

from typing import List, Optional, Tuple

import ollama


def ensure_model_available(model: str) -> Tuple[bool, Optional[str]]:
    """Best-effort check for model availability in Ollama.

    Returns: (available, optional_warning_message)
    """
    try:
        _ = ollama.show(model)
        return True, None
    except Exception:
        pass

    try:
        listing = ollama.list()
        raw_models = listing.get("models", [])

        def norm(s: str) -> str:
            return s.strip().lower()

        def base(s: str) -> str:
            parts = s.rsplit(":", 1)
            return parts[0] if len(parts) == 2 else s

        target = norm(model)
        target_base = base(target)

        candidates: List[str] = []
        for m in raw_models:
            for key in ("name", "model"):
                val = m.get(key)
                if isinstance(val, str):
                    candidates.append(norm(val))

        for c in candidates:
            if c == target:
                return True, None
            if base(c) == target_base:
                return True, None
            if c.startswith(target) or target.startswith(c):
                return True, None

        return False, "Model not detected from Ollama list; proceeding anyway."
    except Exception as e:
        return True, f"Could not verify model availability: {e}"


def get_available_models(defaults: Optional[List[str]] = None) -> List[str]:
    """Return a list of available model names, preferring locally installed."""
    defaults = defaults or []
    ordered: List[str] = []
    seen: set[str] = set()
    try:
        listing = ollama.list()
        for m in listing.get("models", []):
            name = m.get("name") or m.get("model")
            if isinstance(name, str) and name not in seen:
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
            raise RuntimeError("Unexpected response content type from model")
        return content
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}")

