"""LLM access: one app-wide AsyncOpenAI client + guarded streaming.

This module owns the *single* :class:`openai.AsyncOpenAI` instance for the whole
process (connection pooling is what keeps TTFT low), exposes cached embeddings
for RAG, and implements the sentence-level **output guard** that runs *while*
the reply streams.

The guard buffers tokens only until an Arabic sentence boundary, runs the fast
regex check from :mod:`app.core.guards`, and then flushes. The added latency is
one sentence — never the whole reply — and no second LLM is ever called.
"""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.core import cache, guards
from app.utils.logging import get_logger, log_event
import logging

logger = get_logger(__name__)

_client: AsyncOpenAI | None = None


def init_client() -> AsyncOpenAI:
    """Create the shared AsyncOpenAI client (called from the app lifespan)."""

    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # retries on the live path would blow the latency budget
        )
    return _client


def get_client() -> AsyncOpenAI:
    if _client is None:  # pragma: no cover - defensive
        raise RuntimeError("LLM client not initialised; call init_client() first")
    return _client


async def close_client() -> None:
    """Close the shared client (called on shutdown)."""

    global _client
    if _client is not None:
        await _client.close()
        _client = None


# --------------------------------------------------------------------------- #
# Embeddings (cached)
# --------------------------------------------------------------------------- #

async def embed(text: str) -> list[float]:
    """Return the embedding for ``text``, using the Redis cache when possible."""

    cached = await cache.get_embedding(text)
    if cached is not None:
        return cached

    resp = await get_client().embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    vector = resp.data[0].embedding
    await cache.set_embedding(text, vector)
    return vector


# --------------------------------------------------------------------------- #
# Warming
# --------------------------------------------------------------------------- #

async def warm() -> None:
    """Issue tiny throwaway LLM + embedding calls to warm connection pools.

    Run once at startup so the first *real* caller doesn't pay TLS/handshake
    setup cost inside their TTFT budget. Failures are logged, not fatal.
    """

    try:
        stream = await get_client().chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
            stream=True,
        )
        async for _ in stream:  # drain
            break
    except Exception:
        logger.warning("LLM warm call failed", exc_info=True)

    try:
        await embed("تهيئة")
    except Exception:
        logger.warning("embedding warm call failed", exc_info=True)


async def llm_health_check() -> bool:
    """Cheap reachability probe for ``/health`` (lists models)."""

    try:
        await get_client().models.retrieve(settings.llm_model)
        return True
    except Exception:
        logger.warning("llm health check failed", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Raw streaming
# --------------------------------------------------------------------------- #

async def _raw_stream(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Yield raw content deltas from the model as they arrive."""

    stream = await get_client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# --------------------------------------------------------------------------- #
# Guarded streaming (sentence-level output guard)
# --------------------------------------------------------------------------- #

async def stream_guarded(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Stream the reply, flushing one *validated* sentence at a time.

    Yields user-facing text pieces (each already checked by the output guard).
    On a blocked sentence, a failed generation, or an empty reply, it yields the
    graceful Arabic fallback and stops — so the call never dies silently.
    """

    buffer = ""
    yielded_any = False
    try:
        async for delta in _raw_stream(messages):
            buffer += delta
            # Flush every completed sentence sitting in the buffer.
            while True:
                cut = guards.find_sentence_boundary(buffer)
                if cut == -1:
                    break
                sentence, buffer = buffer[:cut], buffer[cut:]
                verdict = guards.check_output_sentence(sentence)
                if not verdict.ok:
                    log_event(
                        logger,
                        logging.WARNING,
                        "output_blocked",
                        reason=verdict.reason,
                    )
                    yield settings.generation_fallback_reply
                    return
                yield sentence
                yielded_any = True

        # Flush any trailing partial sentence (no terminator seen).
        tail = buffer.strip()
        if tail:
            verdict = guards.check_output_sentence(tail)
            if verdict.ok:
                yield buffer
                yielded_any = True
            else:
                log_event(
                    logger,
                    logging.WARNING,
                    "output_blocked",
                    reason=verdict.reason,
                )
                yield settings.generation_fallback_reply
                return

        # Empty generation → never leave the caller with dead air.
        if not yielded_any:
            log_event(logger, logging.WARNING, "empty_generation")
            yield settings.generation_fallback_reply

    except Exception:
        logger.warning("generation failed", exc_info=True)
        # If nothing was spoken yet, the fallback becomes the whole reply.
        yield settings.generation_fallback_reply
