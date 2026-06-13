"""Agentic Memory: graph-based memory layer with sleep-time compute for stateful AI agents.

Quick start::

    from agentic_memory import AgenticMemoryClient, MemoryConfig

    client = AgenticMemoryClient(MemoryConfig(backend="sqlite"))
    await client.initialize()

    await client.core().create("persona", "A helpful assistant")
    node = await client.archival().insert("User likes graph databases")
    results = await client.graph().semantic_search("graph databases")

    await client.shutdown()
"""

from agentic_memory.agent.client import AgenticMemoryClient
from agentic_memory.agent.context_builder import ContextBuilder
from agentic_memory.agent.tools import MEMORY_TOOLS, ToolExecutor
from agentic_memory.config import MemoryConfig
from agentic_memory.exceptions import (
    AgenticMemoryError,
    BlockNotFoundError,
    ConsolidationError,
    EdgeNotFoundError,
    EmbeddingError,
    MemoryLimitExceededError,
    NodeNotFoundError,
    PassageNotFoundError,
    StorageError,
)
from agentic_memory.schemas.consolidation import ConsolidationResult
from agentic_memory.schemas.edge import EdgeCreate, MemoryEdge
from agentic_memory.schemas.memory_block import BlockCreate, MemoryBlock
from agentic_memory.schemas.node import MemoryNode, NodeCreate
from agentic_memory.schemas.passage import Passage
from agentic_memory.schemas.search import NodeSearchResult, SearchQuery
from agentic_memory.types import TraversalStrategy

__all__ = [
    "MEMORY_TOOLS",
    "AgenticMemoryClient",
    "AgenticMemoryError",
    "BlockCreate",
    "BlockNotFoundError",
    "ConsolidationError",
    "ConsolidationResult",
    "ContextBuilder",
    "EdgeCreate",
    "EdgeDirection",
    "EdgeNotFoundError",
    "EmbeddingError",
    "MemoryBlock",
    "MemoryConfig",
    "MemoryEdge",
    "MemoryLimitExceededError",
    "MemoryNode",
    "NodeCreate",
    "NodeNotFoundError",
    "NodeSearchResult",
    "Passage",
    "PassageNotFoundError",
    "SearchQuery",
    "StorageError",
    "ToolExecutor",
    "TraversalStrategy",
]

__version__ = "0.1.0"
