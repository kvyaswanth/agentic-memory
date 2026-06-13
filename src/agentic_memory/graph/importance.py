"""Importance scoring algorithms for memory nodes."""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_memory.schemas.node import MemoryNode
from agentic_memory.storage.base import EdgeStore


async def compute_importance(
    node: MemoryNode,
    edge_count: int,
    recent_access_window_hours: float = 24.0,
) -> float:
    """Score a node from 0.0 to 1.0 based on access frequency, connectivity, and recency.

    Factors:
    - Access frequency (log-scaled)
    - Graph connectivity (degree centrality)
    - Recency boost (accessed within window)
    """
    import math

    # Access frequency component (log-scaled, capped at 1.0).
    access_score = min(1.0, math.log1p(node.access_count) / math.log1p(100))

    # Connectivity component (degree centrality, capped at 1.0).
    connectivity_score = min(1.0, edge_count / 20.0)

    # Recency boost.
    now = datetime.now(UTC).replace(tzinfo=None)
    hours_since_access = (now - node.last_accessed_at).total_seconds() / 3600
    recency_boost = 1.0 if hours_since_access < recent_access_window_hours else 0.0

    # Weighted combination.
    raw = (0.4 * access_score) + (0.35 * connectivity_score) + (0.25 * recency_boost)

    # Blend with explicit importance if set.
    blended = (0.6 * raw) + (0.4 * node.importance)
    return max(0.0, min(1.0, blended))


async def batch_score_nodes(
    nodes: list[MemoryNode],
    edge_store: EdgeStore,
) -> dict[str, float]:
    """Score all nodes in a single pass. Returns node_id -> score mapping."""
    scores: dict[str, float] = {}
    for node in nodes:
        edges = await edge_store.get_edges(node.id)
        score = await compute_importance(node, len(edges))
        scores[node.id] = score
    return scores
