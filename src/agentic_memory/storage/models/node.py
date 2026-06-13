"""ORM model for memory nodes."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from agentic_memory.storage.models.base import Base, SoftDeleteMixin, TimestampMixin


class NodeModel(Base, TimestampMixin, SoftDeleteMixin):
    """Persistent storage for memory graph nodes."""

    __tablename__ = "memory_nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Stored as JSON blob for SQLite; pgvector column added via migration for Postgres.
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    metadata_: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
