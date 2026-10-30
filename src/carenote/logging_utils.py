"""Structured JSON-lines logging for the pipeline.

Every pipeline stage emits one JSON object per event, so retry and fallback
behaviour becomes an auditable artifact instead of a vague claim. Events go to
stderr by default; set CARENOTE_LOG_FILE to also append them to a file.
"""

from __future__ import annotations

import json
import logging
import os
import time


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
        }
        if isinstance(record.msg, dict):
            event.update(record.msg)
        else:
            event["message"] = record.getMessage()
        return json.dumps(event, ensure_ascii=False)


def get_logger(name: str = "carenote") -> logging.Logger:
    """Return the pipeline logger, configured once per process."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = _JsonLinesFormatter()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_file = os.getenv("CARENOTE_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_event(logger: logging.Logger, stage: str, event: str, **fields: object) -> None:
    """Emit one structured event, e.g. log_event(log, 'validate', 'repair_retry', ...)."""
    logger.info({"stage": stage, "event": event, **fields})
