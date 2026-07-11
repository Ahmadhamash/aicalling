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

import asyncio
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
    if settings.llm_provider == "local_stub":
        return AsyncOpenAI(api_key="local-dev")

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

    if settings.llm_provider == "local_stub":
        # Deterministic lightweight vector for offline smoke tests. Production
        # RAG should use real embeddings with ``LLM_PROVIDER=openai``.
        seed = abs(hash(text)) % 997
        return [((seed + i) % 997) / 997.0 for i in range(settings.embedding_dim)]

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

    if settings.llm_provider == "local_stub":
        logger.info("local_stub warm-up complete")
        return

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

    if settings.llm_provider == "local_stub":
        return True

    try:
        await get_client().models.retrieve(settings.llm_model)
        return True
    except Exception:
        logger.warning("llm health check failed", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Raw streaming
# --------------------------------------------------------------------------- #

def _last_user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _local_stub_reply(messages: list[dict[str, str]]) -> str:
    user_text = _last_user_message(messages)
    if any(word in user_text for word in ("مرحبا", "هلا", "السلام", "اهلا", "أهلا")):
        return "هلا والله، أنا نوا من المطعم. كيف بقدر أساعدك اليوم؟"
    if any(word in user_text for word in ("دوام", "ساعات", "متى", "أوقات")):
        return "أكيد، للتجربة المحلية بقدر أحكيلك إن الدوام لازم يطلع من قاعدة معرفة المطعم. إذا بدك بعطيك رابط الحجز أو بحولك لموظف."
    if any(word in user_text for word in ("حجز", "طاولة", "احجز")):
        return "تمام، بقدر أساعدك بالحجز. احكيلي اليوم، الساعة، وعدد الأشخاص."
    if any(word in user_text for word in ("منيو", "قائمة", "سعر", "أسعار")):
        return "بالنسبة للمنيو والأسعار، لازم أعتمد على معلومات المطعم الرسمية. ممكن تحددلي الصنف اللي بتسأل عنه؟"
    return "تمام، وصلتني. أنا نسخة تجربة محلية ستريمنج للمطاعم، وبقدر أساعد بالحجز، المنيو، ساعات الدوام، أو أوصلك مع موظف."


async def _raw_stream(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Yield raw content deltas from the model as they arrive."""

    if settings.llm_provider == "local_stub":
        reply = _local_stub_reply(messages)
        for i in range(0, len(reply), 12):
            await asyncio.sleep(0.04)
            yield reply[i : i + 12]
        return

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
