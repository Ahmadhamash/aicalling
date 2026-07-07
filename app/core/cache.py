"""Redis-backed cache for query embeddings.

Embedding the same normalized query twice is wasteful and adds latency to the
RAG step. We cache embeddings keyed by a normalized form of the query so repeat
questions (very common on a support line) skip the OpenAI embeddings round-trip
entirely.

A single async Redis client is shared process-wide; :func:`init_cache` is called
from the app lifespan and :func:`close_cache` on shutdown.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import redis.asyncio as redis

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Namespaced key prefix so embedding keys never collide with call state.
_KEY_PREFIX = "emb:"

# Diacritics/tatweel we strip so trivially different spellings share a cache key.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_WS = re.compile(r"\s+")

_client: redis.Redis | None = None


def init_cache(client: redis.Redis) -> None:
    """Bind the shared Redis client used for the embedding cache."""

    global _client
    _client = client


def _require_client() -> redis.Redis:
    if _client is None:  # pragma: no cover - defensive
        raise RuntimeError("Embedding cache not initialised; call init_cache() first")
    return _client


def normalize_query(text: str) -> str:
    """Normalize an Arabic/Latin query for stable cache keys.

    Lower-cases, strips diacritics and tatweel, unifies alef/ya/ta-marbuta
    variants, and collapses whitespace. This deliberately errs toward more cache
    hits (semantically identical queries map together).
    """

    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    text = _WS.sub(" ", text)
    return text


def _cache_key(normalized: str) -> str:
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}{settings.embedding_model}:{digest}"


async def get_embedding(text: str) -> list[float] | None:
    """Return a cached embedding for ``text`` if present, else ``None``."""

    key = _cache_key(normalize_query(text))
    try:
        raw = await _require_client().get(key)
    except Exception:  # never let cache errors break the live path
        logger.warning("embedding cache read failed", exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


async def set_embedding(text: str, vector: list[float]) -> None:
    """Cache ``vector`` for ``text`` with the configured TTL (best effort)."""

    key = _cache_key(normalize_query(text))
    try:
        await _require_client().set(
            key,
            json.dumps(vector),
            ex=settings.embedding_cache_ttl_s,
        )
    except Exception:  # cache failures must never break RAG
        logger.warning("embedding cache write failed", exc_info=True)
