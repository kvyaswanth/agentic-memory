"""Pydantic v2 data contracts for all memory operations."""

from agentic_memory.schemas.consolidation import ConsolidationResult
from agentic_memory.schemas.edge import EdgeCreate, EdgeUpdate, MemoryEdge
from agentic_memory.schemas.memory_block import BlockCreate, BlockUpdate, MemoryBlock
from agentic_memory.schemas.node import MemoryNode, NodeCreate, NodeUpdate
from agentic_memory.schemas.passage import Passage, PassageCreate
from agentic_memory.schemas.search import (
    NodeSearchResult,
    PassageSearchResult,
    SearchQuery,
)

__all__ = [
    "BlockCreate",
    "BlockUpdate",
    "ConsolidationResult",
    "EdgeCreate",
    "EdgeUpdate",
    "MemoryBlock",
    "MemoryEdge",
    "MemoryNode",
    "NodeCreate",
    "NodeSearchResult",
    "NodeUpdate",
    "Passage",
    "PassageCreate",
    "PassageSearchResult",
    "SearchQuery",
]
