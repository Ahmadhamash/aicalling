"""Structured JSON logging tagged with the Vapi ``call_id``.

Every log line is emitted as a single JSON object so it can be shipped to
CloudWatch / Loki / Datadog without a parser. A ``call_id`` context variable is
threaded through the request so all logs for one phone call share the same tag,
and a dedicated helper records the time-to-first-token (TTFT) per turn — the
single most important latency metric for a live voice call.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# Per-request call id. Defaults to ``-`` outside a request scope.
_call_id_ctx: ContextVar[str] = ContextVar("call_id", default="-")


class JsonFormatter(logging.Formatter):
    """Render log records as compact single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "call_id": _call_id_ctx.get(),
            "msg": record.getMessage(),
        }
        # Merge any structured extras attached via ``logger.info(..., extra=...)``.
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Quiet down noisy third-party loggers on the hot path.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def set_call_id(call_id: str) -> None:
    """Bind ``call_id`` to the current context (for the duration of a request)."""

    _call_id_ctx.set(call_id or "-")


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper for symmetry / future hooks)."""

    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Emit a structured log line with arbitrary extra JSON fields."""

    logger.log(level, msg, extra={"extra_fields": fields})


class TurnTimer:
    """Measure TTFT and total duration for a single conversational turn.

    Usage::

        timer = TurnTimer(logger, call_id)
        ...  # on first streamed token:
        timer.mark_first_token()
        ...  # when the turn ends:
        timer.finish()
    """

    def __init__(self, logger: logging.Logger, call_id: str) -> None:
        self._logger = logger
        self._call_id = call_id
        self._start = time.perf_counter()
        self._first_token_ms: float | None = None

    def mark_first_token(self) -> None:
        """Record and log the time-to-first-token (once per turn)."""

        if self._first_token_ms is not None:
            return
        self._first_token_ms = (time.perf_counter() - self._start) * 1000.0
        log_event(
            self._logger,
            logging.INFO,
            "ttft",
            call_id=self._call_id,
            ttft_ms=round(self._first_token_ms, 1),
        )

    def finish(self, **fields: Any) -> None:
        """Log total turn latency plus any caller-supplied fields."""

        total_ms = (time.perf_counter() - self._start) * 1000.0
        log_event(
            self._logger,
            logging.INFO,
            "turn_complete",
            call_id=self._call_id,
            total_ms=round(total_ms, 1),
            ttft_ms=(
                round(self._first_token_ms, 1)
                if self._first_token_ms is not None
                else None
            ),
            **fields,
        )
