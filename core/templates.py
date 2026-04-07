"""Back-compat shim for the old ``core.templates`` module.

The canonical API now lives in :mod:`core.prompts` as an immutable
``PromptConfig`` dataclass. These functions remain only to avoid breaking
external callers; new code should use ``PromptConfig`` directly.
"""
from __future__ import annotations

from typing import Optional

from core.prompts import (
    DEFAULT_DESCRIPTION_PROMPT as _DEFAULT_DESCRIPTION,
    DEFAULT_EXTRACTION_PROMPT as _DEFAULT_EXTRACTION,
    PromptConfig,
)

# Module-level mutable state is retained purely for back-compat with
# third-party callers. Internal code paths (pipeline, CLI, UI) now thread
# a ``PromptConfig`` instance explicitly and do NOT read these globals.
DESCRIPTION_PROMPT = _DEFAULT_DESCRIPTION
EXTRACTION_PROMPT = _DEFAULT_EXTRACTION


def build_description_prompt() -> str:
    return DESCRIPTION_PROMPT


def build_extraction_prompt(fields: list[str]) -> str:
    return EXTRACTION_PROMPT.format(fields=", ".join(fields))


def set_templates(description: Optional[str] = None, extraction: Optional[str] = None) -> None:
    global DESCRIPTION_PROMPT, EXTRACTION_PROMPT
    if description:
        DESCRIPTION_PROMPT = description
    if extraction:
        EXTRACTION_PROMPT = extraction


def load_templates_file(path: str) -> None:
    cfg = PromptConfig.from_file(path)
    set_templates(description=cfg.description, extraction=cfg.extraction)


def current_prompt_config() -> PromptConfig:
    """Return a ``PromptConfig`` snapshot of the current module globals."""
    return PromptConfig(description=DESCRIPTION_PROMPT, extraction=EXTRACTION_PROMPT)
