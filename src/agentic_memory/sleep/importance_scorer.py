"""Batch importance scoring for sleep-time consolidation.

Scores all nodes for an agent by combining intrinsic importance,
graph connectivity, access frequency, and time-based decay.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_memory.graph.decay import DecayFunction
from agentic_memory.graph.importance import compute_importance
from agentic_memory.schemas.node import MemoryNode
from agentic_memory.storage.base import EdgeStore, NodeStore
from agentic_memory.utils.logging import get_logger

logger = get_logger("importance_scorer")


class ImportanceScorer:
    """Batch scorer that evaluates all nodes for an agent."""

    def __init__(
        self,
        node_store: NodeStore,
        edge_store: EdgeStore,
        decay_half_life_hours: float = 168.0,
    ) -> None:
        self._nodes = node_store
        self._edges = edge_store
        self._decay = DecayFunction(half_life_hours=decay_half_life_hours)

    async def score_all(self, agent_id: str) -> list[tuple[MemoryNode, float]]:
        """Score all nodes for *agent_id*. Returns list of (node, score) sorted descending."""
        nodes = await self._nodes.list_by_agent(agent_id, limit=10_000)
        now = datetime.now(UTC).replace(tzinfo=None)
        scored: list[tuple[MemoryNode, float]] = []

        for node in nodes:
            edges = await self._edges.get_edges(node.id)
            intrinsic = await compute_importance(node, len(edges))

            hours_since = (now - node.last_accessed_at).total_seconds() / 3600
            decayed = self._decay.compute_with_access_boost(
                intrinsic, hours_since, node.access_count
            )
            scored.append((node, decayed))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "scored %d nodes for agent '%s' (top score: %.3f)",
            len(scored), agent_id, scored[0][1] if scored else 0.0,
        )
        return scored

    async def get_below_threshold(
        self, agent_id: str, threshold: float = 0.1
    ) -> list[MemoryNode]:
        """Return nodes that have fallen below the pruning threshold."""
        scored = await self.score_all(agent_id)
        return [node for node, score in scored if score < threshold]
