from __future__ import annotations

import json
from typing import Optional

DESCRIPTION_PROMPT = "Describe what you see in this image in detail."

EXTRACTION_PROMPT = (
    "Extract the following information from this image: {fields}. "
    "Return the results in JSON format with these exact field names."
)


def build_description_prompt() -> str:
    return DESCRIPTION_PROMPT


def build_extraction_prompt(fields: list[str]) -> str:
    fields_str = ", ".join(fields)
    return EXTRACTION_PROMPT.format(fields=fields_str)


def set_templates(description: Optional[str] = None, extraction: Optional[str] = None) -> None:
    global DESCRIPTION_PROMPT, EXTRACTION_PROMPT
    if description:
        DESCRIPTION_PROMPT = description
    if extraction:
        EXTRACTION_PROMPT = extraction


def load_templates_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    set_templates(
        description=data.get("description"),
        extraction=data.get("extraction"),
    )
