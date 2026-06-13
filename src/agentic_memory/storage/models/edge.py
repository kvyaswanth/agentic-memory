"""ORM model for graph edges."""

from __future__ import annotations

from sqlalchemy import JSON, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from agentic_memory.storage.models.base import Base, TimestampMixin


class EdgeModel(Base, TimestampMixin):
    """Persistent storage for directed graph edges."""

    __tablename__ = "memory_edges"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_node_id: Mapped[str] = mapped_column(
        String(64), index=True, nullable=False
    )
    target_node_id: Mapped[str] = mapped_column(
        String(64), index=True, nullable=False
    )
    relation: Mapped[str] = mapped_column(String(128), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    metadata_: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
