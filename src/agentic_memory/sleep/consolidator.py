"""6-phase memory consolidation pipeline.

Runs during sleep-time to score, cluster, summarize, prune, and strengthen
the memory graph for a given agent.
"""

from __future__ import annotations

from agentic_memory.graph.clustering import find_clusters
from agentic_memory.memory.core import CoreMemory
from agentic_memory.schemas.consolidation import ConsolidationResult
from agentic_memory.schemas.node import NodeCreate
from agentic_memory.sleep.importance_scorer import ImportanceScorer
from agentic_memory.sleep.summarizer import Summarizer
from agentic_memory.storage.base import EdgeStore, NodeStore
from agentic_memory.utils.logging import get_logger

logger = get_logger("consolidator")


class Consolidator:
    """Memory consolidation engine.

    Phases:
    1. Score all nodes for importance + decay
    2. Identify clusters of related memories
    3. Summarize clusters into higher-level nodes
    4. Prune below-threshold nodes
    5. Strengthen frequently-traversed edges
    6. Update core memory blocks if needed
    """

    def __init__(
        self,
        node_store: NodeStore,
        edge_store: EdgeStore,
        core_memory: CoreMemory,
        summarizer: Summarizer,
        scorer: ImportanceScorer,
        pruning_threshold: float = 0.1,
        min_cluster_size: int = 2,
        max_cluster_size: int = 20,
    ) -> None:
        self._nodes = node_store
        self._edges = edge_store
        self._core = core_memory
        self._summarizer = summarizer
        self._scorer = scorer
        self._pruning_threshold = pruning_threshold
        self._min_cluster_size = min_cluster_size
        self._max_cluster_size = max_cluster_size

    async def consolidate(self, agent_id: str) -> ConsolidationResult:
        """Run a full consolidation cycle for *agent_id*."""
        logger.info("starting consolidation for agent '%s'", agent_id)
        result = ConsolidationResult(agent_id=agent_id)

        # Phase 1: Score all nodes.
        scored = await self._scorer.score_all(agent_id)
        result.nodes_scored = len(scored)
        logger.info("phase 1: scored %d nodes", result.nodes_scored)

        if not scored:
            return result

        # Phase 2: Cluster related memories.
        nodes = [n for n, _ in scored]
        clusters = await find_clusters(
            nodes, self._edges, self._min_cluster_size, self._max_cluster_size
        )
        result.clusters_formed = len(clusters)
        logger.info("phase 2: formed %d clusters", result.clusters_formed)

        # Phase 3: Summarize clusters into higher-level nodes.
        for cluster in clusters:
            summary_text = await self._summarizer.summarize(cluster)
            if summary_text:
                await self._nodes.put(
                    NodeCreate(
                        agent_id=agent_id,
                        label="summary",
                        content=summary_text,
                        importance=max(n.importance for n in cluster),
                        metadata={
                            "source_node_ids": [n.id for n in cluster],
                            "consolidated": True,
                        },
                    ),
                    embedding=None,  # Will be embedded lazily on next search.
                )
                result.summaries_created += 1
        logger.info("phase 3: created %d summaries", result.summaries_created)

        # Phase 4: Prune below-threshold nodes.
        for node, score in scored:
            if score < self._pruning_threshold and not node.metadata.get("consolidated"):
                await self._nodes.delete(node.id)
                result.nodes_pruned += 1
        logger.info(
            "phase 4: pruned %d nodes (threshold=%.2f)",
            result.nodes_pruned, self._pruning_threshold,
        )

        # Phase 5: Strengthen frequently-traversed edges.
        for node, _score in scored[:50]:  # Top 50 active nodes.
            edges = await self._edges.get_edges(node.id)
            for edge in edges:
                if edge.weight < 1.0:
                    new_weight = min(1.0, edge.weight + 0.05)
                    await self._edges.update_weight(edge.id, new_weight)
                    result.edges_strengthened += 1
        logger.info("phase 5: strengthened %d edges", result.edges_strengthened)

        # Phase 6: Update core memory if consolidation surfaces critical info.
        if result.summaries_created > 0:
            try:
                core_ctx = self._core.get("context")
                current = core_ctx.value
                if len(current) < core_ctx.limit - 100:
                    summary_note = (
                        f"[Consolidated {result.summaries_created} memory clusters, "
                        f"pruned {result.nodes_pruned} stale memories]"
                    )
                    await self._core.append("context", summary_note)
                    result.core_memory_updated = True
            except Exception:
                pass  # context block may not exist.
        logger.info("phase 6: core_memory_updated=%s", result.core_memory_updated)

        logger.info("consolidation complete: %s", result)
        return result
