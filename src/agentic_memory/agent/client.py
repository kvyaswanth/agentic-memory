"""AgenticMemoryClient: the main public entry point.

Usage::

    from agentic_memory import AgenticMemoryClient, MemoryConfig

    client = AgenticMemoryClient(MemoryConfig(backend="sqlite"))
    await client.initialize()

    # Core memory
    await client.core("agent-1").create("persona", "A helpful AI assistant")
    await client.core("agent-1").append("human", "Name: Alice")

    # Archival + graph
    node = await client.archival("agent-1").insert("Alice presented graph research")
    results = await client.archival("agent-1").search("Alice's research")

    # Sleep-time consolidation
    result = await client.consolidate("agent-1")

    await client.shutdown()
"""

from __future__ import annotations

from agentic_memory.agent.context_builder import ContextBuilder
from agentic_memory.agent.tools import MEMORY_TOOLS, ToolExecutor
from agentic_memory.config import MemoryConfig
from agentic_memory.graph.engine import GraphEngine
from agentic_memory.memory.archival import ArchivalMemory
from agentic_memory.memory.core import CoreMemory
from agentic_memory.memory.manager import MemoryManager
from agentic_memory.memory.recall import RecallMemory
from agentic_memory.schemas.consolidation import ConsolidationResult
from agentic_memory.utils.logging import get_logger

logger = get_logger("client")


class AgenticMemoryClient:
    """Main entry point for the Agentic Memory library.

    Wraps ``MemoryManager`` and exposes the three memory tiers, graph engine,
    tools, and context builder through a clean async API.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()
        self._manager: MemoryManager | None = None
        self._default_agent: str = "default"

    async def initialize(self, agent_id: str | None = None) -> None:
        """Initialize all subsystems and optionally register a default agent.

        Args:
            agent_id: Optional agent ID. If provided, automatically registered.
        """
        self._manager = MemoryManager(self._config)
        await self._manager.initialize()

        if agent_id:
            self._default_agent = agent_id
        self._manager.get_or_create_agent(self._default_agent)

        logger.info(
            "client initialized (backend=%s, agent=%s)",
            self._config.backend, self._default_agent,
        )

    def register_agent(self, agent_id: str) -> None:
        """Register an additional agent."""
        self._assert_initialized()
        self._manager.get_or_create_agent(agent_id)

    # -- Memory tier accessors ---------------------------------------------

    def core(self, agent_id: str | None = None) -> CoreMemory:
        """Access core memory (in-context blocks) for an agent."""
        self._assert_initialized()
        return self._manager.core(agent_id or self._default_agent)

    def recall(self, agent_id: str | None = None) -> RecallMemory:
        """Access recall memory (conversation history) for an agent."""
        self._assert_initialized()
        return self._manager.recall(agent_id or self._default_agent)

    def archival(self, agent_id: str | None = None) -> ArchivalMemory:
        """Access archival memory (long-term nodes) for an agent."""
        self._assert_initialized()
        return self._manager.archival(agent_id or self._default_agent)

    def graph(self, agent_id: str | None = None) -> GraphEngine:
        """Access the HydraDB graph engine for an agent."""
        self._assert_initialized()
        return self._manager.graph(agent_id or self._default_agent)

    # -- Tools and context -------------------------------------------------

    def tools(self, agent_id: str | None = None) -> ToolExecutor:
        """Get a tool executor for integrating with LLM function calling."""
        self._assert_initialized()
        aid = agent_id or self._default_agent
        return ToolExecutor(
            core=self._manager.core(aid),
            recall=self._manager.recall(aid),
            archival=self._manager.archival(aid),
        )

    @staticmethod
    def tool_schemas() -> list[dict]:
        """Return the JSON schemas for all memory tools."""
        return MEMORY_TOOLS

    def context_builder(self, agent_id: str | None = None) -> ContextBuilder:
        """Get a context builder for assembling prompts."""
        self._assert_initialized()
        aid = agent_id or self._default_agent
        return ContextBuilder(core=self._manager.core(aid))

    # -- Consolidation -----------------------------------------------------

    async def consolidate(self, agent_id: str | None = None) -> ConsolidationResult:
        """Manually trigger sleep-time consolidation for an agent."""
        self._assert_initialized()
        return await self._manager.consolidate(agent_id or self._default_agent)

    # -- Lifecycle ---------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if self._manager:
            await self._manager.shutdown()
            self._manager = None
        logger.info("client shut down")

    async def __aenter__(self) -> AgenticMemoryClient:
        await self.initialize()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.shutdown()

    def _assert_initialized(self) -> None:
        if self._manager is None:
            raise RuntimeError(
                "Client not initialized. Call 'await client.initialize()' first."
            )
