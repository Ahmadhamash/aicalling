"""Per-call conversation state stored in Redis, keyed by the Vapi call id.

Vapi sends the full message history on every request, but we keep our own
compact per-call record so we can (a) trim history to a fixed window
deterministically and (b) attach lightweight metadata (turn counts, timing)
without trusting the client. A single Redis ``GET`` returns the state well under
10ms, so it never threatens the TTFT budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import redis.asyncio as redis

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "call:"

_client: redis.Redis | None = None


def init_state(client: redis.Redis) -> None:
    """Bind the shared Redis client used for per-call state."""

    global _client
    _client = client


def _require_client() -> redis.Redis:
    if _client is None:  # pragma: no cover - defensive
        raise RuntimeError("State store not initialised; call init_state() first")
    return _client


def _key(call_id: str) -> str:
    return f"{_KEY_PREFIX}{call_id}"


@dataclass
class Turn:
    """One (role, content) exchange element in the conversation."""

    role: str
    content: str


@dataclass
class CallState:
    """Compact persisted state for a single phone call."""

    call_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def recent_history(self, limit: int) -> list[dict[str, str]]:
        """Return the last ``limit`` turns as chat-message dicts."""

        window = self.turns[-limit:] if limit > 0 else []
        return [{"role": t.role, "content": t.content} for t in window]


def _deserialize(raw: str, call_id: str) -> CallState:
    data = json.loads(raw)
    turns = [Turn(**t) for t in data.get("turns", [])]
    return CallState(
        call_id=data.get("call_id", call_id),
        turns=turns,
        created_at=data.get("created_at", time.time()),
        updated_at=data.get("updated_at", time.time()),
    )


async def get_state(call_id: str) -> CallState:
    """Load call state (single Redis GET). Returns a fresh state on miss/error."""

    try:
        raw = await _require_client().get(_key(call_id))
    except Exception:
        logger.warning("state read failed", exc_info=True)
        return CallState(call_id=call_id)
    if not raw:
        return CallState(call_id=call_id)
    try:
        return _deserialize(raw, call_id)
    except (ValueError, TypeError):
        logger.warning("state deserialize failed", exc_info=True)
        return CallState(call_id=call_id)


async def get_recent_history(call_id: str, limit: int | None = None) -> list[dict[str, str]]:
    """Convenience: load state and return its recent history window."""

    limit = settings.history_turns if limit is None else limit
    state = await get_state(call_id)
    return state.recent_history(limit)


async def append_turn(call_id: str, user_text: str, assistant_text: str) -> None:
    """Append a user+assistant exchange and persist with a refreshed TTL.

    Called *after* a turn completes, so it never sits on the TTFT path. History
    is trimmed to twice the configured window to bound memory.
    """

    state = await get_state(call_id)
    if user_text:
        state.turns.append(Turn(role="user", content=user_text))
    if assistant_text:
        state.turns.append(Turn(role="assistant", content=assistant_text))
    # Keep a little more than the prompt window so context survives trimming.
    max_keep = max(settings.history_turns * 2, 4)
    state.turns = state.turns[-max_keep:]
    state.updated_at = time.time()

    try:
        await _require_client().set(
            _key(call_id),
            json.dumps(asdict(state)),
            ex=settings.state_ttl_s,
        )
    except Exception:
        logger.warning("state write failed", exc_info=True)
