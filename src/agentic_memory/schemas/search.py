"""Pydantic schemas for search queries and results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agentic_memory.schemas.node import MemoryNode
from agentic_memory.schemas.passage import Passage


class SearchQuery(BaseModel):
    """Parameters for a memory search."""

    query: str
    limit: int = 10
    min_score: float = 0.0
    agent_id: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    expand_context: bool = False
    expand_depth: int = 1


class NodeSearchResult(BaseModel):
    """A single search result from node / archival memory."""

    node: MemoryNode
    score: float
    context: list[MemoryNode] = Field(default_factory=list)


class PassageSearchResult(BaseModel):
    """A single search result from recall memory."""

    passage: Passage
    score: float
