"""HydraDB graph engine: the core differentiator.

Combines vector similarity with graph traversal for contextually rich
memory retrieval. Each memory is a node; semantically similar memories
are connected via auto-generated edges.
"""

from __future__ import annotations

from agentic_memory.embeddings.base import BaseEmbeddingProvider
from agentic_memory.graph.clustering import find_clusters
from agentic_memory.graph.decay import DecayFunction
from agentic_memory.graph.importance import batch_score_nodes
from agentic_memory.graph.traversal import (
    breadth_first_search,
    depth_first_search,
    importance_weighted_traverse,
    recency_weighted_traverse,
)
from agentic_memory.schemas.edge import EdgeCreate, MemoryEdge
from agentic_memory.schemas.node import MemoryNode, NodeCreate
from agentic_memory.schemas.search import NodeSearchResult
from agentic_memory.storage.base import EdgeStore, NodeStore
from agentic_memory.types import TraversalStrategy
from agentic_memory.utils.logging import get_logger

logger = get_logger("graph")


class GraphEngine:
    """HydraDB: graph memory layer for stateful AI agents.

    Provides node/edge operations, semantic search with graph expansion,
    auto-linking, traversal, and clustering.
    """

    def __init__(
        self,
        agent_id: str,
        node_store: NodeStore,
        edge_store: EdgeStore,
        embedder: BaseEmbeddingProvider,
        auto_link_threshold: float = 0.85,
        max_auto_links: int = 5,
        decay_half_life_hours: float = 168.0,
    ) -> None:
        self._agent_id = agent_id
        self._nodes = node_store
        self._edges = edge_store
        self._embedder = embedder
        self._auto_link_threshold = auto_link_threshold
        self._max_auto_links = max_auto_links
        self._decay = DecayFunction(half_life_hours=decay_half_life_hours)

    # -- Node operations ---------------------------------------------------

    async def add_memory(
        self,
        text: str,
        label: str = "memory",
        importance: float = 0.5,
        metadata: dict | None = None,
        auto_link: bool = True,
    ) -> MemoryNode:
        """Add a memory node, generate embedding, optionally auto-link."""
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
        logger.info("added memory node '%s'", node.id)

        if auto_link:
            await self.auto_link(node.id, embedding=embedding)

        return node

    async def get_node(self, node_id: str) -> MemoryNode:
        """Retrieve a single node."""
        return await self._nodes.get(node_id)

    async def delete_node(self, node_id: str) -> bool:
        """Soft-delete a node."""
        return await self._nodes.delete(node_id)

    async def list_nodes(self, limit: int = 100) -> list[MemoryNode]:
        """List all nodes for this agent."""
        return await self._nodes.list_by_agent(self._agent_id, limit=limit)

    # -- Edge operations ---------------------------------------------------

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryEdge:
        """Create an explicit relationship between two memories."""
        edge = await self._edges.create(
            EdgeCreate(
                source_node_id=source_id,
                target_node_id=target_id,
                relation=relation,
                weight=weight,
                metadata=metadata or {},
            )
        )
        logger.debug("added edge '%s' --[%s]--> '%s'", source_id, relation, target_id)
        return edge

    async def get_edges(self, node_id: str) -> list[MemoryEdge]:
        """Get all edges connected to a node."""
        return await self._edges.get_edges(node_id)

    async def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        return await self._edges.delete(edge_id)

    # -- Search ------------------------------------------------------------

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        expand_context: bool = True,
        expand_depth: int = 1,
    ) -> list[NodeSearchResult]:
        """Search by embedding, optionally expand results with graph neighbors."""
        query_embedding = await self._embedder.embed(query)
        raw = await self._nodes.search_by_embedding(
            query_embedding, agent_id=self._agent_id, limit=limit
        )

        results: list[NodeSearchResult] = []
        for node, score in raw:
            context: list[MemoryNode] = []
            if expand_context:
                context = await self._nodes.get_neighbors(
                    node.id, max_depth=expand_depth
                )
            results.append(NodeSearchResult(node=node, score=score, context=context))

        return results

    # -- Traversal ---------------------------------------------------------

    async def traverse(
        self,
        start_id: str,
        strategy: TraversalStrategy = TraversalStrategy.BFS,
        max_depth: int = 3,
        max_nodes: int = 50,
    ) -> list[MemoryNode]:
        """Traverse the graph from a starting node using the given strategy."""
        if strategy == TraversalStrategy.BFS:
            return await breadth_first_search(
                self._edges, self._nodes, start_id, max_depth, max_nodes
            )
        elif strategy == TraversalStrategy.DFS:
            return await depth_first_search(
                self._edges, self._nodes, start_id, max_depth, max_nodes
            )
        elif strategy == TraversalStrategy.IMPORTANCE_WEIGHTED:
            return await importance_weighted_traverse(
                self._edges, self._nodes, start_id, max_nodes
            )
        elif strategy == TraversalStrategy.RECENCY_WEIGHTED:
            return await recency_weighted_traverse(
                self._edges, self._nodes, start_id, max_nodes
            )
        else:
            raise ValueError(f"Unknown traversal strategy: {strategy}")

    # -- Auto-linking ------------------------------------------------------

    async def auto_link(
        self,
        node_id: str,
        embedding: list[float] | None = None,
        similarity_threshold: float | None = None,
        max_links: int | None = None,
    ) -> list[MemoryEdge]:
        """Create edges to semantically similar existing nodes."""
        threshold = similarity_threshold or self._auto_link_threshold
        limit = max_links or self._max_auto_links

        if embedding is None:
            node = await self._nodes.get(node_id)
            if node.embedding is None:
                embedding = await self._embedder.embed(node.content)
            else:
                embedding = node.embedding

        candidates = await self._nodes.search_by_embedding(
            embedding, agent_id=self._agent_id, limit=limit + 1
        )

        linked: list[MemoryEdge] = []
        for candidate, score in candidates:
            if candidate.id == node_id:
                continue
            if score < threshold:
                continue
            if len(linked) >= limit:
                break
            edge = await self._edges.create(
                EdgeCreate(
                    source_node_id=node_id,
                    target_node_id=candidate.id,
                    relation="related_to",
                    weight=score,
                )
            )
            linked.append(edge)

        return linked

    # -- Clustering --------------------------------------------------------

    async def cluster_memories(
        self,
        min_cluster_size: int = 2,
        max_cluster_size: int = 20,
    ) -> list[list[MemoryNode]]:
        """Group related memories into clusters using connected components."""
        nodes = await self._nodes.list_by_agent(self._agent_id, limit=1000)
        return await find_clusters(nodes, self._edges, min_cluster_size, max_cluster_size)

    # -- Merge -------------------------------------------------------------

    async def merge_nodes(self, source_id: str, target_id: str) -> MemoryNode:
        """Merge source node into target, re-wiring all edges."""
        target = await self._nodes.get(target_id)
        source_edges = await self._edges.get_edges(source_id)

        # Re-wire: replace source_id references with target_id.
        for edge in source_edges:
            new_source = target_id if edge.source_node_id == source_id else edge.source_node_id
            new_target = target_id if edge.target_node_id == source_id else edge.target_node_id
            if new_source != new_target:
                await self._edges.create(
                    EdgeCreate(
                        source_node_id=new_source,
                        target_node_id=new_target,
                        relation=edge.relation,
                        weight=edge.weight,
                    )
                )
            await self._edges.delete(edge.id)

        await self._nodes.delete(source_id)
        logger.info("merged node '%s' into '%s'", source_id, target_id)
        return target

    # -- Scoring -----------------------------------------------------------

    async def score_all_nodes(self) -> dict[str, float]:
        """Compute importance scores for all agent nodes."""
        nodes = await self._nodes.list_by_agent(self._agent_id, limit=1000)
        return await batch_score_nodes(nodes, self._edges)
