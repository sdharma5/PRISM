"""Append-only JSONL run logging.

A structured line-per-event log is diffable and machine-readable, which matters
when the question months later is "did *that* run really use those folds?".
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_LOG_NAME = "training_log.jsonl"


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and Paths into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


class JsonlLogger:
    """Writes one JSON object per line, and optionally mirrors to the console."""

    def __init__(
        self,
        path: str | Path,
        *,
        echo: bool = True,
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.run_id = run_id
        self._console = _console_logger() if echo else None

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event. Returns the record written, for convenience in tests."""
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
        }
        if self.run_id:
            record["run_id"] = self.run_id
        record.update(_jsonable(fields))

        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        if self._console is not None:
            self._console.info("%s %s", event, _summarize(record))
        return record

    def read(self) -> list[dict[str, Any]]:
        """Read the log back — used by tests and by post-hoc run inspection."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def _summarize(record: dict[str, Any]) -> str:
    skip = {"timestamp", "event", "run_id"}
    parts = [f"{k}={v}" for k, v in record.items() if k not in skip]
    return " ".join(parts)[:400]


def _console_logger() -> logging.Logger:
    logger = logging.getLogger("prism.training")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def get_logger(name: str = "prism") -> logging.Logger:
    """A plain console logger for scripts that do not need structured events."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["DEFAULT_LOG_NAME", "JsonlLogger", "get_logger"]
