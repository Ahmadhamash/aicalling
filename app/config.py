"""Application configuration via pydantic-settings.

All tunables (model names, API keys, service URLs, latency budgets) are read
from environment variables so the deployment can be reconfigured without code
changes. In particular ``LLM_MODEL`` defaults to ``gpt-4o-mini`` but can be
swapped for any OpenAI chat model later.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service metadata -------------------------------------------------
    app_name: str = "Nawa AI"
    environment: str = Field("development", description="dev/staging/production")
    log_level: str = Field("INFO", description="Root log level")

    # --- OpenAI / LLM -----------------------------------------------------
    openai_api_key: str = Field("local-dev", description="OpenAI API key")
    llm_provider: str = Field(
        "openai",
        description="LLM provider: openai for API-compatible backends, local_stub for offline local testing",
    )
    openai_base_url: str | None = Field(
        None, description="Optional override for OpenAI-compatible gateways"
    )
    # Swap this env var to change the model without touching code.
    llm_model: str = Field("gpt-4o-mini", description="Chat completion model")
    llm_temperature: float = Field(0.3, description="Sampling temperature")
    llm_max_tokens: int = Field(250, description="Max tokens per streamed reply")
    llm_timeout_s: float = Field(15.0, description="Per-request LLM timeout")

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model for RAG"
    )
    embedding_dim: int = Field(1536, description="Embedding vector dimensionality")

    # --- Qdrant -----------------------------------------------------------
    qdrant_url: str = Field("http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: str | None = Field(None, description="Qdrant API key (cloud)")
    qdrant_collection: str = Field("nawa_kb", description="Knowledge-base collection")
    rag_top_k: int = Field(3, description="Number of context chunks to retrieve")
    rag_score_threshold: float = Field(
        0.30, description="Minimum cosine score to keep a retrieved chunk"
    )
    rag_enabled: bool = Field(True, description="Enable embedding + Qdrant retrieval")

    # --- Redis ------------------------------------------------------------
    redis_url: str = Field("redis://localhost:6379/0", description="Redis URL")
    embedding_cache_ttl_s: int = Field(
        60 * 60 * 24 * 7, description="TTL for cached query embeddings (7d)"
    )
    state_ttl_s: int = Field(
        60 * 60 * 2, description="TTL for per-call conversation state (2h)"
    )
    history_turns: int = Field(
        6, description="Number of recent turns to load into the prompt"
    )
    strict_dependency_health: bool = Field(
        True,
        description="Require Redis and Qdrant in /health; disable only for offline local LLM smoke tests",
    )

    # --- Guardrails / behaviour ------------------------------------------
    # Polite Jordanian-Arabic redirect used when the input guard blocks a turn.
    blocked_input_reply: str = (
        "بعتذر، بس هاد الشي مش من شغلي. أنا هون أساعدك بأمور خدمة العملاء بس. "
        "كيف بقدر أخدمك؟"
    )
    # Graceful spoken fallback used when generation fails or output is blocked.
    generation_fallback_reply: str = (
        "بعتذر منك، صار عندي خلل بسيط هلق. ممكن تعيد سؤالك أو بوصلك مع "
        "موظف يساعدك؟"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (read once per process)."""

    return Settings()  # type: ignore[call-arg]


# Convenience module-level singleton for imports that don't need DI.
settings = get_settings()
