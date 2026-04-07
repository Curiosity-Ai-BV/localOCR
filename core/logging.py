"""Lightweight structured logging for localOCR.

Uses the stdlib ``logging`` module with a JSON-ish formatter so operators
can pipe logs through ``jq`` without adding a runtime dep.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extras injected via ``logger.info(..., extra={"k": v})``.
        for k, v in record.__dict__.items():
            if k in payload or k.startswith("_"):
                continue
            if k in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message", "module",
                "msecs", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
            ):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = repr(v)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | int | None = None) -> None:
    """Idempotently configure the root logger.

    Honours ``LOCALOCR_LOG_LEVEL`` env var. Uses plain text output unless
    ``LOCALOCR_LOG_JSON=1`` is set, so interactive Streamlit / CLI usage
    stays readable.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = level or os.environ.get("LOCALOCR_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("LOCALOCR_LOG_JSON") == "1":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("localocr")
    root.handlers[:] = [handler]
    root.setLevel(lvl)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"localocr.{name}")
