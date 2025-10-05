from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

JSONDict = Dict[str, object]


def _extract_json_from_fenced_blocks(text: str) -> List[JSONDict]:
    results: List[JSONDict] = []
    for match in re.finditer(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidate = match.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                results.append(obj)
        except Exception:
            continue
    return results


def _extract_json_by_brace_scanning(text: str) -> List[JSONDict]:
    results: List[JSONDict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{" and depth == 0:
            start = i
            depth = 1
        elif ch == "{" and depth > 0:
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                snippet = text[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict):
                        results.append(obj)
                except Exception:
                    pass
                start = -1
    return results


def _extract_field_heuristics(text: str, fields: Iterable[str]) -> JSONDict:
    data: JSONDict = {}
    for field in fields:
        f = field.strip()
        if not f:
            continue
        pattern = rf"(?i)\b{re.escape(f)}\b\s*[:=\-]\s*(\".*?\"|'.*?'|[\w\-./$%,]+)"
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            data[f] = val
    return data


def extract_structured_data(content: str, fields: List[str] | None) -> JSONDict:
    fields = fields or []
    for obj in _extract_json_from_fenced_blocks(content):
        return obj
    objs = _extract_json_by_brace_scanning(content)
    if objs:
        if fields:
            for o in objs:
                if any(f in o for f in fields):
                    return o
        return objs[0]
    if fields:
        return _extract_field_heuristics(content, fields)
    return {}

