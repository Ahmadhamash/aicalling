"""Retrieval-Augmented Generation for the knowledge base.

The RAG step runs *before* the SSE stream opens and is budgeted at ~80ms:

1. A cheap regex **intent gate** skips retrieval entirely for greetings and
   chitchat (no point embedding "مرحبا").
2. Otherwise the query is embedded (Redis-cached) and the top-k chunks are
   fetched from Qdrant with an optional metadata prefilter and score threshold.
3. Retrieved facts are joined into a single string that the prompt builder wraps
   in a dedicated system block instructing the model to answer ONLY from it.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.config import settings
from app.core import llm
from app.services import vector_db
from app.utils.logging import get_logger, log_event
import logging

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Intent gate — skip RAG for greetings / social chitchat.
# --------------------------------------------------------------------------- #
_CHITCHAT_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"^\s*(مرحبا|هلا|هلو|اهلين|أهلا|السلام|سلام|صباح|مسا|كيفك|شلونك|شو اخبارك)\b",
        r"^\s*(hi|hello|hey|good\s+(morning|evening)|how\s+are\s+you|thanks?|thank\s+you)\b",
        r"^\s*(شكرا|مشكور|يسلمو|تسلم|باي|مع السلامه|تمام|اوكي|اوكيه)\b",
    )
]

# Very short utterances are almost always social; embedding them wastes budget.
_MIN_RAG_CHARS = 6


def should_skip_rag(text: str) -> bool:
    """Return ``True`` when the utterance is a greeting/chitchat (skip retrieval)."""

    stripped = text.strip()
    if len(stripped) < _MIN_RAG_CHARS:
        return True
    return any(rx.search(stripped) for rx in _CHITCHAT_PATTERNS)


def _join_context(chunks: list[vector_db.RetrievedChunk]) -> str:
    """Join retrieved chunk texts into a single numbered context block."""

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"- {chunk.text.strip()}")
    return "\n".join(lines)


async def retrieve_context(
    query: str,
    prefilter: dict[str, Any] | None = None,
) -> str | None:
    """Return joined retrieved facts for ``query``, or ``None`` to skip RAG.

    Parameters
    ----------
    query:
        The user's latest utterance.
    prefilter:
        Optional exact-match metadata filter forwarded to Qdrant (e.g. language
        or category constraints).
    """

    if should_skip_rag(query):
        log_event(logger, logging.INFO, "rag_skipped", reason="chitchat")
        return None

    start = time.perf_counter()
    try:
        vector = await llm.embed(query)
        chunks = await vector_db.search(vector, top_k=settings.rag_top_k, prefilter=prefilter)
    except Exception:
        # RAG is best-effort: on failure, fall through to the base prompt rather
        # than blocking the reply. The prompt's "if unknown, offer a human" rule
        # keeps the answer safe.
        logger.warning("rag retrieval failed", exc_info=True)
        return None

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log_event(
        logger,
        logging.INFO,
        "rag_done",
        hits=len(chunks),
        elapsed_ms=round(elapsed_ms, 1),
    )

    if not chunks:
        return None
    return _join_context(chunks)
