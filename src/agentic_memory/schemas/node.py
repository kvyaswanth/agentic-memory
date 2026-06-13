"""Pydantic schemas for memory nodes."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class NodeCreate(BaseModel):
    """Payload for creating a new memory node."""

    agent_id: str
    label: str
    content: str
    importance: float = 0.5
    metadata: dict = Field(default_factory=dict)


class NodeUpdate(BaseModel):
    """Payload for updating an existing memory node."""

    content: str | None = None
    importance: float | None = None
    metadata: dict | None = None


class MemoryNode(BaseModel):
    """A single memory node in the graph."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    label: str
    content: str
    embedding: list[float] | None = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: datetime = Field(default_factory=_utcnow)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
