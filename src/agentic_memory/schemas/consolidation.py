"""Pydantic schemas for sleep-time consolidation results."""

from __future__ import annotations

from pydantic import BaseModel


class ConsolidationResult(BaseModel):
    """Summary of a single consolidation cycle."""

    agent_id: str
    nodes_scored: int = 0
    clusters_formed: int = 0
    summaries_created: int = 0
    nodes_pruned: int = 0
    edges_strengthened: int = 0
    core_memory_updated: bool = False
