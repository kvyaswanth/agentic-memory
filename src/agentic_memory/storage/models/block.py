"""ORM model for core memory blocks."""

from __future__ import annotations

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from agentic_memory.storage.models.base import Base, TimestampMixin


class BlockModel(Base, TimestampMixin):
    """Persistent storage for in-context memory blocks."""

    __tablename__ = "memory_blocks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    limit: Mapped[int] = mapped_column(Integer, default=2000, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    read_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
