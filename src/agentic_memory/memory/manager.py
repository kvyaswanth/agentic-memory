"""MemoryManager: unified facade that wires all tiers, graph engine, and sleep scheduler.

This is the internal orchestrator. The public entry point is ``AgenticMemoryClient``.
"""

from __future__ import annotations

from agentic_memory.config import MemoryConfig
from agentic_memory.embeddings.base import BaseEmbeddingProvider
from agentic_memory.embeddings.fake import FakeEmbedding
from agentic_memory.graph.engine import GraphEngine
from agentic_memory.memory.archival import ArchivalMemory
from agentic_memory.memory.core import CoreMemory
from agentic_memory.memory.recall import RecallMemory
from agentic_memory.schemas.consolidation import ConsolidationResult
from agentic_memory.sleep.consolidator import Consolidator
from agentic_memory.sleep.importance_scorer import ImportanceScorer
from agentic_memory.sleep.scheduler import SleepScheduler
from agentic_memory.sleep.summarizer import Summarizer
from agentic_memory.storage.base import BlockStore, EdgeStore, NodeStore, PassageStore
from agentic_memory.storage.sqlite import create_sqlite_stores
from agentic_memory.utils.logging import get_logger

logger = get_logger("manager")


class MemoryManager:
    """Orchestrates all memory subsystems for one or more agents.

    Call ``initialize()`` to set up storage and start the sleep scheduler.
    Call ``shutdown()`` to tear down gracefully.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._agent_ids: set[str] = set()

        # Populated during initialize().
        self._node_store: NodeStore | None = None
        self._edge_store: EdgeStore | None = None
        self._block_store: BlockStore | None = None
        self._passage_store: PassageStore | None = None
        self._embedder: BaseEmbeddingProvider | None = None

        # Per-agent instances (lazy-created on first access).
        self._core_memories: dict[str, CoreMemory] = {}
        self._recall_memories: dict[str, RecallMemory] = {}
        self._archival_memories: dict[str, ArchivalMemory] = {}
        self._graph_engines: dict[str, GraphEngine] = {}

        self._sleep_scheduler: SleepScheduler | None = None

    async def initialize(self) -> None:
        """Set up storage backends, embedding provider, and sleep scheduler."""
        # -- Storage -------------------------------------------------------
        if self._config.backend == "sqlite":
            stores = await create_sqlite_stores(self._config.database_url)
            self._node_store, self._edge_store, self._block_store, self._passage_store = stores
        else:
            raise NotImplementedError(
                f"Backend '{self._config.backend}' not yet implemented. Use 'sqlite'."
            )

        # -- Embeddings ----------------------------------------------------
        self._embedder = self._create_embedder()

        # -- Sleep scheduler -----------------------------------------------
        consolidator = Consolidator(
            node_store=self._node_store,
            edge_store=self._edge_store,
            core_memory=None,  # type: ignore[arg-type]  # Set per-agent later.
            summarizer=Summarizer(max_words=self._config.summary_max_words),
            scorer=ImportanceScorer(
                self._node_store, self._edge_store,
                decay_half_life_hours=self._config.importance_decay_half_life_hours,
            ),
            pruning_threshold=self._config.pruning_threshold,
            max_cluster_size=self._config.max_cluster_size,
        )
        self._sleep_scheduler = SleepScheduler(
            consolidator=consolidator,
            interval_seconds=self._config.consolidation_interval_seconds,
        )
        await self._sleep_scheduler.start()

        logger.info("memory manager initialized (backend=%s)", self._config.backend)

    async def shutdown(self) -> None:
        """Graceful teardown of scheduler and connections."""
        if self._sleep_scheduler:
            await self._sleep_scheduler.stop()
        logger.info("memory manager shut down")

    def get_or_create_agent(self, agent_id: str) -> None:
        """Register an agent and create its per-agent memory instances."""
        if agent_id in self._agent_ids:
            return
        self._agent_ids.add(agent_id)

        assert self._node_store is not None
        assert self._edge_store is not None
        assert self._block_store is not None
        assert self._passage_store is not None
        assert self._embedder is not None

        core = CoreMemory(agent_id, self._block_store)
        recall = RecallMemory(agent_id, self._passage_store, self._embedder)
        archival = ArchivalMemory(
            agent_id, self._node_store, self._edge_store, self._embedder,
            auto_link_threshold=self._config.auto_link_threshold,
            max_auto_links=self._config.max_auto_links,
        )
        graph = GraphEngine(
            agent_id, self._node_store, self._edge_store, self._embedder,
            auto_link_threshold=self._config.auto_link_threshold,
            max_auto_links=self._config.max_auto_links,
            decay_half_life_hours=self._config.importance_decay_half_life_hours,
        )

        self._core_memories[agent_id] = core
        self._recall_memories[agent_id] = recall
        self._archival_memories[agent_id] = archival
        self._graph_engines[agent_id] = graph

        logger.info("registered agent '%s'", agent_id)

    def core(self, agent_id: str) -> CoreMemory:
        return self._core_memories[agent_id]

    def recall(self, agent_id: str) -> RecallMemory:
        return self._recall_memories[agent_id]

    def archival(self, agent_id: str) -> ArchivalMemory:
        return self._archival_memories[agent_id]

    def graph(self, agent_id: str) -> GraphEngine:
        return self._graph_engines[agent_id]

    async def consolidate(self, agent_id: str) -> ConsolidationResult:
        """Manually trigger consolidation for an agent."""
        if self._sleep_scheduler is None:
            raise RuntimeError("Manager not initialized")
        return await self._sleep_scheduler.run_once(agent_id)

    def _create_embedder(self) -> BaseEmbeddingProvider:
        if self._config.embedding_provider == "fake":
            return FakeEmbedding(dimension=self._config.embedding_dimension)
        elif self._config.embedding_provider == "openai":
            from agentic_memory.embeddings.openai import OpenAIEmbedding
            return OpenAIEmbedding(
                model=self._config.embedding_model,
                dimension=self._config.embedding_dimension,
                api_key=self._config.openai_api_key,
            )
        elif self._config.embedding_provider == "sentence_transformers":
            from agentic_memory.embeddings.sentence_transformers import SentenceTransformerEmbedding
            return SentenceTransformerEmbedding(model_name=self._config.embedding_model)
        else:
            raise ValueError(f"Unknown embedding provider: {self._config.embedding_provider}")
