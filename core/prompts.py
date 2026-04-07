"""Immutable prompt configuration.

Replaces the mutable module globals in ``core.templates`` with a
``@dataclass(frozen=True) PromptConfig`` that must be explicitly threaded
through callers. This eliminates the CLI concurrency race where one thread
could mutate another thread's prompt template mid-batch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import List, Optional

DEFAULT_DESCRIPTION_PROMPT = "Describe what you see in this image in detail."
DEFAULT_EXTRACTION_PROMPT = (
    "Extract the following information from this image: {fields}. "
    "Return the results in JSON format with these exact field names."
)


@dataclass(frozen=True)
class PromptConfig:
    description: str = DEFAULT_DESCRIPTION_PROMPT
    extraction: str = DEFAULT_EXTRACTION_PROMPT

    def build_description(self) -> str:
        return self.description

    def build_extraction(self, fields: List[str]) -> str:
        return self.extraction.format(fields=", ".join(fields))

    def with_overrides(
        self,
        description: Optional[str] = None,
        extraction: Optional[str] = None,
    ) -> "PromptConfig":
        kwargs = {}
        if description:
            kwargs["description"] = description
        if extraction:
            kwargs["extraction"] = extraction
        return replace(self, **kwargs) if kwargs else self

    @classmethod
    def from_file(cls, path: str) -> "PromptConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls().with_overrides(
            description=data.get("description"),
            extraction=data.get("extraction"),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PromptConfig":
        return cls().with_overrides(
            description=data.get("description"),
            extraction=data.get("extraction"),
        )
