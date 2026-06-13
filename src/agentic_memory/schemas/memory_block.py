"""Pydantic schemas for core memory blocks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BlockCreate(BaseModel):
    """Payload for creating a new core memory block."""

    agent_id: str
    label: str
    value: str
    limit: int = 2000
    description: str | None = None
    read_only: bool = False


class BlockUpdate(BaseModel):
    """Payload for updating a core memory block."""

    value: str | None = None
    limit: int | None = None
    description: str | None = None


class MemoryBlock(BaseModel):
    """An in-context memory block that lives in the agent's prompt."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    label: str
    value: str
    limit: int = 2000
    description: str | None = None
    read_only: bool = False
