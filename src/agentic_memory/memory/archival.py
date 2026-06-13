"""Archival memory tier: long-term persistent memory with vector + graph search.

Each inserted memory becomes a graph node with an embedding. Auto-linking
connects semantically similar nodes via graph edges for richer retrieval.
"""

from __future__ import annotations

from agentic_memory.embeddings.base import BaseEmbeddingProvider
from agentic_memory.schemas.edge import EdgeCreate
from agentic_memory.schemas.node import MemoryNode, NodeCreate
from agentic_memory.schemas.search import NodeSearchResult
from agentic_memory.storage.base import EdgeStore, NodeStore
from agentic_memory.utils.logging import get_logger

logger = get_logger("archival")


class ArchivalMemory:
    """Long-term persistent memory backed by graph nodes."""

    def __init__(
        self,
        agent_id: str,
        node_store: NodeStore,
        edge_store: EdgeStore,
        embedder: BaseEmbeddingProvider,
        auto_link_threshold: float = 0.85,
        max_auto_links: int = 5,
    ) -> None:
        self._agent_id = agent_id
        self._nodes = node_store
        self._edges = edge_store
        self._embedder = embedder
        self._auto_link_threshold = auto_link_threshold
        self._max_auto_links = max_auto_links

    async def insert(
        self,
        text: str,
        label: str = "memory",
        importance: float = 0.5,
        metadata: dict | None = None,
        auto_link: bool = True,
    ) -> MemoryNode:
        """Insert text as a graph node with embedding and optional auto-links."""
        embedding = await self._embedder.embed(text)
        node = await self._nodes.put(
            NodeCreate(
                agent_id=self._agent_id,
                label=label,
                content=text,
                importance=importance,
                metadata=metadata or {},
            ),
            embedding=embedding,
        )
        logger.debug("inserted archival node '%s' (%d chars)", node.id, len(text))

        if auto_link:
            links = await self._auto_link(node, embedding)
            logger.debug("auto-linked node '%s' to %d similar nodes", node.id, len(links))

        return node

    async def search(
        self, query: str, limit: int = 10, expand_context: bool = False, expand_depth: int = 1
    ) -> list[NodeSearchResult]:
        """Semantic search over archival nodes, optionally expanding with graph neighbors."""
        query_embedding = await self._embedder.embed(query)
        raw = await self._nodes.search_by_embedding(
            query_embedding, agent_id=self._agent_id, limit=limit
        )

        results: list[NodeSearchResult] = []
        for node, score in raw:
            context: list[MemoryNode] = []
            if expand_context:
                context = await self._nodes.get_neighbors(node.id, max_depth=expand_depth)
            results.append(NodeSearchResult(node=node, score=score, context=context))

        return results

    async def get(self, node_id: str) -> MemoryNode:
        """Retrieve a single node by ID."""
        return await self._nodes.get(node_id)

    async def delete(self, node_id: str) -> bool:
        """Soft-delete a node."""
        return await self._nodes.delete(node_id)

    async def get_context(self, node_id: str, max_depth: int = 2) -> list[MemoryNode]:
        """Retrieve node plus its graph neighborhood."""
        node = await self._nodes.get(node_id)
        neighbors = await self._nodes.get_neighbors(node_id, max_depth=max_depth)
        return [node, *neighbors]

    async def list_all(self, limit: int = 100) -> list[MemoryNode]:
        """List all archival nodes for this agent."""
        return await self._nodes.list_by_agent(self._agent_id, limit=limit)

    async def _auto_link(
        self, new_node: MemoryNode, new_embedding: list[float]
    ) -> list[MemoryNode]:
        """Find semantically similar existing nodes and create edges."""
        candidates = await self._nodes.search_by_embedding(
            new_embedding, agent_id=self._agent_id, limit=self._max_auto_links + 1
        )
        linked: list[MemoryNode] = []
        for candidate, score in candidates:
            if candidate.id == new_node.id:
                continue
            if score < self._auto_link_threshold:
                continue
            if len(linked) >= self._max_auto_links:
                break
            await self._edges.create(
                EdgeCreate(
                    source_node_id=new_node.id,
                    target_node_id=candidate.id,
                    relation="related_to",
                    weight=score,
                )
            )
            linked.append(candidate)

        return linked
