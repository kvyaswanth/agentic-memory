"""Recall memory tier: searchable history of agent interactions.

Stores conversation passages with embeddings for semantic retrieval.
"""

from __future__ import annotations

from datetime import datetime

from agentic_memory.embeddings.base import BaseEmbeddingProvider
from agentic_memory.schemas.passage import Passage, PassageCreate
from agentic_memory.schemas.search import PassageSearchResult
from agentic_memory.storage.base import PassageStore
from agentic_memory.utils.logging import get_logger

logger = get_logger("recall")


class RecallMemory:
    """Searchable history of agent conversations."""

    def __init__(
        self,
        agent_id: str,
        passage_store: PassageStore,
        embedder: BaseEmbeddingProvider,
    ) -> None:
        self._agent_id = agent_id
        self._store = passage_store
        self._embedder = embedder

    async def add(self, content: str, role: str, metadata: dict | None = None) -> Passage:
        """Store a conversation passage with its embedding."""
        embedding = await self._embedder.embed(content)
        passage = await self._store.put(
            PassageCreate(
                agent_id=self._agent_id,
                content=content,
                role=role,
                metadata=metadata or {},
            ),
            embedding=embedding,
        )
        logger.debug("added recall passage (%s, %d chars)", role, len(content))
        return passage

    async def search(self, query: str, limit: int = 10) -> list[PassageSearchResult]:
        """Search recall passages by semantic similarity."""
        query_embedding = await self._embedder.embed(query)
        raw = await self._store.search(query_embedding, agent_id=self._agent_id, limit=limit)
        return [PassageSearchResult(passage=p, score=s) for p, s in raw]

    async def search_by_date(
        self, start: datetime | None = None, end: datetime | None = None, limit: int = 100
    ) -> list[Passage]:
        """Retrieve passages within a date range (filtering done post-query)."""
        passages = await self._store.list_by_agent(self._agent_id, limit=limit * 5)
        results = passages
        if start:
            results = [p for p in results if p.created_at >= start]
        if end:
            results = [p for p in results if p.created_at <= end]
        return results[:limit]

    async def list_recent(self, limit: int = 20) -> list[Passage]:
        """Return the most recent passages."""
        return await self._store.list_by_agent(self._agent_id, limit=limit)
