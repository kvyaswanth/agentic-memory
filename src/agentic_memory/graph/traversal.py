"""Graph traversal algorithms for memory retrieval.

Provides BFS, DFS, and weighted traversal strategies over the memory graph.
All algorithms are fully async and yield results lazily.
"""

from __future__ import annotations

from agentic_memory.schemas.node import MemoryNode
from agentic_memory.storage.base import EdgeStore, NodeStore
from agentic_memory.types import EdgeDirection
from agentic_memory.utils.logging import get_logger

logger = get_logger("traversal")


async def breadth_first_search(
    edge_store: EdgeStore,
    node_store: NodeStore,
    start_id: str,
    max_depth: int = 3,
    max_nodes: int = 50,
) -> list[MemoryNode]:
    """BFS from *start_id*, returning discovered nodes up to *max_depth* hops."""
    visited: set[str] = {start_id}
    frontier: list[str] = [start_id]
    results: list[MemoryNode] = []

    for _ in range(max_depth):
        if not frontier or len(results) >= max_nodes:
            break
        next_frontier: list[str] = []
        for node_id in frontier:
            edges = await edge_store.get_edges(node_id, direction=EdgeDirection.BOTH)
            for edge in edges:
                neighbor_id = (
                    edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
                )
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        for nid in next_frontier:
            try:
                node = await node_store.get(nid)
                results.append(node)
                if len(results) >= max_nodes:
                    break
            except Exception:
                continue
        frontier = next_frontier

    return results


async def depth_first_search(
    edge_store: EdgeStore,
    node_store: NodeStore,
    start_id: str,
    max_depth: int = 3,
    max_nodes: int = 50,
) -> list[MemoryNode]:
    """DFS from *start_id*, returning discovered nodes up to *max_depth* hops."""
    visited: set[str] = set()
    results: list[MemoryNode] = []

    async def _dfs(node_id: str, depth: int) -> None:
        if depth > max_depth or len(results) >= max_nodes or node_id in visited:
            return
        visited.add(node_id)
        try:
            node = await node_store.get(node_id)
            results.append(node)
        except Exception:
            return
        edges = await edge_store.get_edges(node_id, direction=EdgeDirection.BOTH)
        for edge in edges:
            neighbor_id = (
                edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            )
            await _dfs(neighbor_id, depth + 1)
            if len(results) >= max_nodes:
                return

    await _dfs(start_id, 0)
    return results


async def importance_weighted_traverse(
    edge_store: EdgeStore,
    node_store: NodeStore,
    start_id: str,
    max_nodes: int = 50,
) -> list[MemoryNode]:
    """Traverse neighbors sorted by importance score (highest first).

    Uses a priority queue to always expand the highest-importance frontier node.
    """
    import heapq

    visited: set[str] = set()
    results: list[MemoryNode] = []
    # Max-heap via negated importance.
    heap: list[tuple[float, str]] = []

    try:
        start_node = await node_store.get(start_id)
    except Exception:
        return []

    visited.add(start_id)
    heapq.heappush(heap, (-start_node.importance, start_id))

    while heap and len(results) < max_nodes:
        _, node_id = heapq.heappop(heap)
        try:
            node = await node_store.get(node_id)
            results.append(node)
        except Exception:
            continue

        edges = await edge_store.get_edges(node_id, direction=EdgeDirection.BOTH)
        for edge in edges:
            neighbor_id = (
                edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            )
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                try:
                    neighbor = await node_store.get(neighbor_id)
                    heapq.heappush(heap, (-neighbor.importance, neighbor_id))
                except Exception:
                    continue

    return results


async def recency_weighted_traverse(
    edge_store: EdgeStore,
    node_store: NodeStore,
    start_id: str,
    max_nodes: int = 50,
) -> list[MemoryNode]:
    """Traverse neighbors sorted by recency (most recently accessed first)."""
    import heapq

    visited: set[str] = set()
    results: list[MemoryNode] = []
    # Max-heap via negated timestamp (most recent first).
    heap: list[tuple[float, str]] = []

    try:
        start_node = await node_store.get(start_id)
    except Exception:
        return []

    visited.add(start_id)
    heapq.heappush(heap, (-start_node.last_accessed_at.timestamp(), start_id))

    while heap and len(results) < max_nodes:
        _, node_id = heapq.heappop(heap)
        try:
            node = await node_store.get(node_id)
            results.append(node)
        except Exception:
            continue

        edges = await edge_store.get_edges(node_id, direction=EdgeDirection.BOTH)
        for edge in edges:
            neighbor_id = (
                edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            )
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                try:
                    neighbor = await node_store.get(neighbor_id)
                    heapq.heappush(heap, (-neighbor.last_accessed_at.timestamp(), neighbor_id))
                except Exception:
                    continue

    return results
