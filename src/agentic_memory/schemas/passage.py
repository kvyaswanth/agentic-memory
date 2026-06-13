"""Pydantic schemas for recall passages (conversation history)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PassageCreate(BaseModel):
    """Payload for creating a new passage."""

    agent_id: str
    content: str
    role: str  # "user" | "assistant" | "system"
    metadata: dict = Field(default_factory=dict)


class Passage(BaseModel):
    """A single passage from conversation history."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    content: str
    role: str
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
