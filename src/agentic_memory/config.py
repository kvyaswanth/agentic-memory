"""Configuration management via pydantic-settings."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryConfig(BaseSettings):
    """Central configuration for the Agentic Memory system.

    All settings can be overridden via environment variables prefixed with
    ``AGENTIC_MEMORY_`` (e.g. ``AGENTIC_MEMORY_BACKEND=postgres``).
    """

    # -- Storage ----------------------------------------------------------
    backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = "sqlite+aiosqlite:///./memory.db"
    redis_url: str = "redis://localhost:6379/0"
    pool_size: int = 10
    max_overflow: int = 20

    # -- Embeddings -------------------------------------------------------
    embedding_provider: Literal["openai", "sentence_transformers", "fake"] = "fake"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    openai_api_key: str | None = None

    # -- Graph (HydraDB) --------------------------------------------------
    auto_link_threshold: float = 0.85
    max_auto_links: int = 5
    default_traversal_depth: int = 3
    default_traversal_max_nodes: int = 50

    # -- Sleep-time compute -----------------------------------------------
    consolidation_interval_seconds: int = 300
    importance_decay_half_life_hours: float = 168.0
    pruning_threshold: float = 0.1
    max_cluster_size: int = 20
    summary_max_words: int = 100

    # -- Core memory ------------------------------------------------------
    default_block_char_limit: int = 2000

    model_config = SettingsConfigDict(env_prefix="AGENTIC_MEMORY_")
