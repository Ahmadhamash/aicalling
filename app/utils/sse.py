"""Helpers to build OpenAI-format Server-Sent Events (SSE) chunks.

Vapi expects the Custom LLM to reply with the *exact* OpenAI streaming shape:
each event is a ``data: {json}\\n\\n`` line whose payload is a
``chat.completion.chunk`` object, and the stream terminates with a literal
``data: [DONE]\\n\\n``.

Arabic must survive the wire intact, so every payload is serialized with
``json.dumps(..., ensure_ascii=False)``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

# Sentinel line that ends every OpenAI-compatible SSE stream.
DONE = "data: [DONE]\n\n"


def new_completion_id() -> str:
    """Return a fresh ``chatcmpl-`` style id for a streamed completion."""

    return f"chatcmpl-{uuid.uuid4().hex}"


def _envelope(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
    created: int,
) -> str:
    """Serialize one ``chat.completion.chunk`` into an SSE ``data:`` line."""

    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    # ensure_ascii=False keeps Arabic as UTF-8 rather than \uXXXX escapes.
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def role_chunk(completion_id: str, model: str, role: str = "assistant") -> str:
    """First chunk of a stream: announces the assistant role."""

    return _envelope(completion_id, model, {"role": role}, None, int(time.time()))


def content_chunk(completion_id: str, model: str, text: str) -> str:
    """A content delta chunk carrying a piece of the reply text."""

    return _envelope(completion_id, model, {"content": text}, None, int(time.time()))


def finish_chunk(
    completion_id: str, model: str, finish_reason: str = "stop"
) -> str:
    """Terminal chunk: empty delta plus a ``finish_reason``."""

    return _envelope(completion_id, model, {}, finish_reason, int(time.time()))


def done() -> str:
    """The ``[DONE]`` sentinel that closes the SSE stream."""

    return DONE
