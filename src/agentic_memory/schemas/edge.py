"""Pydantic schemas for graph edges."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class EdgeCreate(BaseModel):
    """Payload for creating a new graph edge."""

    source_node_id: str
    target_node_id: str
    relation: str
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)


class EdgeUpdate(BaseModel):
    """Payload for updating an existing graph edge."""

    weight: float | None = None
    metadata: dict | None = None


class MemoryEdge(BaseModel):
    """A directed edge connecting two memory nodes."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    source_node_id: str
    target_node_id: str
    relation: str
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
