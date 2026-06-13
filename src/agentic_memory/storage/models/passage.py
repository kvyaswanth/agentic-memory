"""ORM model for recall passages."""

from __future__ import annotations

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from agentic_memory.storage.models.base import Base, TimestampMixin


class PassageModel(Base, TimestampMixin):
    """Persistent storage for conversation recall passages."""

    __tablename__ = "memory_passages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)

    # Stored as JSON blob for SQLite; pgvector column added via migration for Postgres.
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
