"""Shared test fixtures."""

from __future__ import annotations

import pytest

from agentic_memory import AgenticMemoryClient, MemoryConfig


@pytest.fixture
async def client():
    """Provide an in-memory SQLite client with fake embeddings."""
    cfg = MemoryConfig(
        backend="sqlite",
        database_url="sqlite+aiosqlite:///:memory:",
        embedding_provider="fake",
        embedding_dimension=64,
    )
    c = AgenticMemoryClient(cfg)
    await c.initialize(agent_id="test-agent")
    yield c
    await c.shutdown()


@pytest.fixture
def config():
    """Provide a test configuration."""
    return MemoryConfig(
        backend="sqlite",
        database_url="sqlite+aiosqlite:///:memory:",
        embedding_provider="fake",
        embedding_dimension=64,
    )
