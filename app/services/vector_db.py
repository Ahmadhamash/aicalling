"""Async Qdrant client wrapper for the knowledge-base collection.

Wraps :class:`qdrant_client.AsyncQdrantClient` with the two operations the app
needs: ensuring the collection exists (used by the ingest script and startup)
and a metadata-prefiltered top-k similarity search (used by RAG). A single
client is shared process-wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    """Return the shared async Qdrant client, creating it on first use."""

    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            timeout=5.0,
        )
    return _client


async def close_client() -> None:
    """Close the shared client (called from the app lifespan on shutdown)."""

    global _client
    if _client is not None:
        await _client.close()
        _client = None


@dataclass
class RetrievedChunk:
    """A single retrieved knowledge-base chunk."""

    text: str
    score: float
    payload: dict[str, Any]


async def ensure_collection() -> None:
    """Create the KB collection with cosine distance if it does not exist."""

    client = get_client()
    exists = await client.collection_exists(settings.qdrant_collection)
    if exists:
        return
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("created Qdrant collection %s", settings.qdrant_collection)


def _build_filter(prefilter: dict[str, Any] | None) -> models.Filter | None:
    """Translate a simple ``{field: value}`` dict into a Qdrant filter."""

    if not prefilter:
        return None
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in prefilter.items()
    ]
    return models.Filter(must=conditions)


async def search(
    vector: list[float],
    top_k: int | None = None,
    prefilter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k most similar chunks, optionally metadata-prefiltered.

    Parameters
    ----------
    vector:
        The query embedding.
    top_k:
        Number of results (defaults to ``settings.rag_top_k``).
    prefilter:
        Optional exact-match metadata filter applied *before* scoring, e.g.
        ``{"lang": "ar", "category": "hours"}``.
    score_threshold:
        Optional minimum cosine score (defaults to ``settings.rag_score_threshold``).
    """

    client = get_client()
    top_k = settings.rag_top_k if top_k is None else top_k
    threshold = settings.rag_score_threshold if score_threshold is None else score_threshold

    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=top_k,
        query_filter=_build_filter(prefilter),
        score_threshold=threshold,
        with_payload=True,
    )

    chunks: list[RetrievedChunk] = []
    for point in results.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        if not text:
            continue
        chunks.append(RetrievedChunk(text=text, score=point.score, payload=payload))
    return chunks


async def health_check() -> bool:
    """Lightweight reachability probe used by the ``/health`` endpoint."""

    try:
        await get_client().get_collections()
        return True
    except Exception:
        logger.warning("qdrant health check failed", exc_info=True)
        return False
