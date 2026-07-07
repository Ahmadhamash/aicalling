"""FastAPI application entrypoint for Nawa AI.

The lifespan handler warms every pool that sits on the live-call latency path —
the OpenAI client, Qdrant, and Redis — with throwaway calls, so the first real
caller never pays cold-start cost inside their TTFT budget. On a live phone
call, a cold start is dead air, so this warming is load-bearing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.config import settings
from app.core import cache, llm, state
from app.services import vector_db
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and warm all shared clients, tear them down on shutdown."""

    configure_logging(settings.log_level)
    logger.info("starting %s (env=%s)", settings.app_name, settings.environment)

    # --- Redis (shared by embedding cache + per-call state) --------------
    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    cache.init_cache(redis_client)
    state.init_state(redis_client)

    # --- OpenAI + Qdrant -------------------------------------------------
    llm.init_client()
    try:
        await vector_db.ensure_collection()
    except Exception:
        # Missing/unreachable collection shouldn't stop the app from booting;
        # /health will report it and RAG degrades gracefully.
        logger.warning("could not ensure Qdrant collection at startup", exc_info=True)

    # --- Warm the hot path ----------------------------------------------
    try:
        await redis_client.ping()
    except Exception:
        logger.warning("redis ping failed at startup", exc_info=True)
    await llm.warm()
    logger.info("warm-up complete; ready to serve")

    try:
        yield
    finally:
        logger.info("shutting down")
        await llm.close_client()
        await vector_db.close_client()
        try:
            await redis_client.aclose()
        except Exception:
            logger.warning("error closing redis", exc_info=True)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Jordanian-Arabic voice AI backend — Vapi Custom LLM.",
    lifespan=lifespan,
)

app.include_router(chat_router)


@app.get("/health")
async def health() -> JSONResponse:
    """Readiness probe checking Redis, Qdrant, and LLM reachability.

    Returns 200 only when all dependencies are reachable; 503 otherwise. Used by
    the ALB/ECS target group and by ops dashboards.
    """

    redis_ok = False
    try:
        await cache._require_client().ping()  # type: ignore[attr-defined]
        redis_ok = True
    except Exception:
        logger.warning("redis health check failed", exc_info=True)

    qdrant_ok = await vector_db.health_check()
    llm_ok = await llm.llm_health_check()

    healthy = redis_ok and qdrant_ok and llm_ok
    body = {
        "status": "ok" if healthy else "degraded",
        "redis": redis_ok,
        "qdrant": qdrant_ok,
        "llm": llm_ok,
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@app.get("/")
async def root() -> dict[str, str]:
    """Simple liveness endpoint."""

    return {"service": settings.app_name, "status": "alive"}
