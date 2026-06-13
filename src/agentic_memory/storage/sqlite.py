"""SQLite + aiosqlite storage backend (zero-config development mode).

Stores embeddings as JSON blobs. Uses numpy cosine similarity for vector search.
Auto-creates tables on first connection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentic_memory.exceptions import (
    BlockNotFoundError,
    EdgeNotFoundError,
    NodeNotFoundError,
    PassageNotFoundError,
)
from agentic_memory.schemas.edge import EdgeCreate, MemoryEdge
from agentic_memory.schemas.memory_block import BlockCreate, MemoryBlock
from agentic_memory.schemas.node import MemoryNode, NodeCreate, NodeUpdate
from agentic_memory.schemas.passage import Passage, PassageCreate
from agentic_memory.storage.base import BlockStore, EdgeStore, NodeStore, PassageStore
from agentic_memory.storage.models import Base
from agentic_memory.storage.models.block import BlockModel
from agentic_memory.storage.models.edge import EdgeModel
from agentic_memory.storage.models.node import NodeModel
from agentic_memory.storage.models.passage import PassageModel
from agentic_memory.types import (
    BLOCK_PREFIX,
    EDGE_PREFIX,
    NODE_PREFIX,
    PASSAGE_PREFIX,
    EdgeDirection,
)
from agentic_memory.utils.hashing import make_id


def _embedding_to_json(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return json.dumps(vec)


def _json_to_embedding(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    return json.loads(raw)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# Node store
# ---------------------------------------------------------------------------


class SQLiteNodeStore(NodeStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, node_id: str) -> MemoryNode:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NodeModel).where(
                    NodeModel.id == node_id, NodeModel.is_deleted == False  # noqa: E712
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise NodeNotFoundError(node_id)
            return self._to_schema(row)

    async def put(self, data: NodeCreate, embedding: list[float] | None = None) -> MemoryNode:
        node_id = make_id(NODE_PREFIX)
        async with self._session_factory() as session:
            model = NodeModel(
                id=node_id,
                agent_id=data.agent_id,
                label=data.label,
                content=data.content,
                embedding_json=_embedding_to_json(embedding),
                importance=data.importance,
                metadata_=data.metadata,
            )
            session.add(model)
            await session.commit()
            return self._to_schema(model)

    async def batch_put(
        self, items: list[tuple[NodeCreate, list[float] | None]]
    ) -> list[MemoryNode]:
        models: list[NodeModel] = []
        for data, emb in items:
            models.append(
                NodeModel(
                    id=make_id(NODE_PREFIX),
                    agent_id=data.agent_id,
                    label=data.label,
                    content=data.content,
                    embedding_json=_embedding_to_json(emb),
                    importance=data.importance,
                    metadata_=data.metadata,
                )
            )
        async with self._session_factory() as session:
            session.add_all(models)
            await session.commit()
            return [self._to_schema(m) for m in models]

    async def update(self, node_id: str, data: NodeUpdate) -> MemoryNode:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NodeModel).where(
                    NodeModel.id == node_id, NodeModel.is_deleted == False  # noqa: E712
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise NodeNotFoundError(node_id)
            if data.content is not None:
                row.content = data.content
            if data.importance is not None:
                row.importance = data.importance
            if data.metadata is not None:
                row.metadata_ = data.metadata
            row.updated_at = datetime.now(UTC).replace(tzinfo=None)
            await session.commit()
            return self._to_schema(row)

    async def delete(self, node_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                update(NodeModel)
                .where(NodeModel.id == node_id)
                .values(is_deleted=True, updated_at=datetime.now(UTC).replace(tzinfo=None))
            )
            await session.commit()
            return result.rowcount > 0

    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[MemoryNode]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NodeModel)
                .where(NodeModel.agent_id == agent_id, NodeModel.is_deleted == False)  # noqa: E712
                .order_by(NodeModel.created_at.desc())
                .limit(limit)
            )
            return [self._to_schema(r) for r in result.scalars().all()]

    async def get_neighbors(
        self,
        node_id: str,
        direction: EdgeDirection = EdgeDirection.BOTH,
        max_depth: int = 1,
    ) -> list[MemoryNode]:
        if max_depth < 1:
            return []
        async with self._session_factory() as session:
            # Gather edge IDs at each depth level.
            visited: set[str] = {node_id}
            frontier: set[str] = {node_id}
            for _ in range(max_depth):
                next_frontier: set[str] = set()
                conditions = []
                if direction in (EdgeDirection.OUTGOING, EdgeDirection.BOTH):
                    conditions.append(EdgeModel.source_node_id.in_(frontier))
                if direction in (EdgeDirection.INCOMING, EdgeDirection.BOTH):
                    conditions.append(EdgeModel.target_node_id.in_(frontier))
                from sqlalchemy import or_
                edge_result = await session.execute(
                    select(EdgeModel).where(or_(*conditions))
                )
                for edge in edge_result.scalars().all():
                    for nid in (edge.source_node_id, edge.target_node_id):
                        if nid not in visited:
                            next_frontier.add(nid)
                visited |= next_frontier
                frontier = next_frontier
                if not frontier:
                    break
            visited.discard(node_id)
            if not visited:
                return []
            node_result = await session.execute(
                select(NodeModel).where(
                    NodeModel.id.in_(visited), NodeModel.is_deleted == False  # noqa: E712
                )
            )
            return [self._to_schema(r) for r in node_result.scalars().all()]

    async def search_by_embedding(
        self,
        embedding: list[float],
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryNode, float]]:
        async with self._session_factory() as session:
            stmt = select(NodeModel).where(NodeModel.is_deleted == False)  # noqa: E712
            if agent_id:
                stmt = stmt.where(NodeModel.agent_id == agent_id)
            result = await session.execute(stmt)
            scored: list[tuple[MemoryNode, float]] = []
            for row in result.scalars().all():
                emb = _json_to_embedding(row.embedding_json)
                if emb is None:
                    continue
                score = _cosine_similarity(embedding, emb)
                scored.append((self._to_schema(row), score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

    @staticmethod
    def _to_schema(row: NodeModel) -> MemoryNode:
        return MemoryNode(
            id=row.id,
            agent_id=row.agent_id,
            label=row.label,
            content=row.content,
            embedding=_json_to_embedding(row.embedding_json),
            importance=row.importance,
            access_count=row.access_count,
            last_accessed_at=row.last_accessed_at,
            metadata=row.metadata_,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


# ---------------------------------------------------------------------------
# Edge store
# ---------------------------------------------------------------------------


class SQLiteEdgeStore(EdgeStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(self, data: EdgeCreate) -> MemoryEdge:
        edge_id = make_id(EDGE_PREFIX)
        async with self._session_factory() as session:
            model = EdgeModel(
                id=edge_id,
                source_node_id=data.source_node_id,
                target_node_id=data.target_node_id,
                relation=data.relation,
                weight=data.weight,
                metadata_=data.metadata,
            )
            session.add(model)
            await session.commit()
            return self._to_schema(model)

    async def get(self, edge_id: str) -> MemoryEdge:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EdgeModel).where(EdgeModel.id == edge_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise EdgeNotFoundError(edge_id)
            return self._to_schema(row)

    async def get_edges(
        self, node_id: str, direction: EdgeDirection = EdgeDirection.BOTH
    ) -> list[MemoryEdge]:
        async with self._session_factory() as session:
            conditions = []
            if direction in (EdgeDirection.OUTGOING, EdgeDirection.BOTH):
                conditions.append(EdgeModel.source_node_id == node_id)
            if direction in (EdgeDirection.INCOMING, EdgeDirection.BOTH):
                conditions.append(EdgeModel.target_node_id == node_id)
            from sqlalchemy import or_
            result = await session.execute(select(EdgeModel).where(or_(*conditions)))
            return [self._to_schema(r) for r in result.scalars().all()]

    async def get_edges_by_relation(self, relation: str, agent_id: str) -> list[MemoryEdge]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EdgeModel)
                .join(NodeModel, NodeModel.id == EdgeModel.source_node_id)
                .where(
                    EdgeModel.relation == relation,
                    NodeModel.agent_id == agent_id,
                )
            )
            return [self._to_schema(r) for r in result.scalars().all()]

    async def update_weight(self, edge_id: str, weight: float) -> MemoryEdge:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EdgeModel).where(EdgeModel.id == edge_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise EdgeNotFoundError(edge_id)
            row.weight = weight
            await session.commit()
            return self._to_schema(row)

    async def delete(self, edge_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(EdgeModel).where(EdgeModel.id == edge_id)
            )
            await session.commit()
            return result.rowcount > 0

    @staticmethod
    def _to_schema(row: EdgeModel) -> MemoryEdge:
        return MemoryEdge(
            id=row.id,
            source_node_id=row.source_node_id,
            target_node_id=row.target_node_id,
            relation=row.relation,
            weight=row.weight,
            metadata=row.metadata_,
            created_at=row.created_at,
        )


# ---------------------------------------------------------------------------
# Block store
# ---------------------------------------------------------------------------


class SQLiteBlockStore(BlockStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, block_id: str) -> MemoryBlock:
        async with self._session_factory() as session:
            result = await session.execute(
                select(BlockModel).where(BlockModel.id == block_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise BlockNotFoundError(block_id)
            return self._to_schema(row)

    async def get_by_label(self, agent_id: str, label: str) -> MemoryBlock | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(BlockModel).where(
                    BlockModel.agent_id == agent_id, BlockModel.label == label
                )
            )
            row = result.scalar_one_or_none()
            return self._to_schema(row) if row else None

    async def put(self, data: BlockCreate) -> MemoryBlock:
        block_id = make_id(BLOCK_PREFIX)
        async with self._session_factory() as session:
            model = BlockModel(
                id=block_id,
                agent_id=data.agent_id,
                label=data.label,
                value=data.value,
                limit=data.limit,
                description=data.description,
                read_only=data.read_only,
            )
            session.add(model)
            await session.commit()
            return self._to_schema(model)

    async def update_value(self, block_id: str, value: str) -> MemoryBlock:
        async with self._session_factory() as session:
            result = await session.execute(
                select(BlockModel).where(BlockModel.id == block_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise BlockNotFoundError(block_id)
            row.value = value
            row.updated_at = datetime.now(UTC).replace(tzinfo=None)
            await session.commit()
            return self._to_schema(row)

    async def get_by_agent(self, agent_id: str) -> list[MemoryBlock]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(BlockModel).where(BlockModel.agent_id == agent_id)
            )
            return [self._to_schema(r) for r in result.scalars().all()]

    async def delete(self, block_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(BlockModel).where(BlockModel.id == block_id)
            )
            await session.commit()
            return result.rowcount > 0

    @staticmethod
    def _to_schema(row: BlockModel) -> MemoryBlock:
        return MemoryBlock(
            id=row.id,
            agent_id=row.agent_id,
            label=row.label,
            value=row.value,
            limit=row.limit,
            description=row.description,
            read_only=row.read_only,
        )


# ---------------------------------------------------------------------------
# Passage store
# ---------------------------------------------------------------------------


class SQLitePassageStore(PassageStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def put(
        self, data: PassageCreate, embedding: list[float] | None = None
    ) -> Passage:
        passage_id = make_id(PASSAGE_PREFIX)
        async with self._session_factory() as session:
            model = PassageModel(
                id=passage_id,
                agent_id=data.agent_id,
                content=data.content,
                role=data.role,
                embedding_json=_embedding_to_json(embedding),
                metadata_=data.metadata,
            )
            session.add(model)
            await session.commit()
            return self._to_schema(model)

    async def batch_put(
        self, items: list[tuple[PassageCreate, list[float] | None]]
    ) -> list[Passage]:
        models: list[PassageModel] = []
        for data, emb in items:
            models.append(
                PassageModel(
                    id=make_id(PASSAGE_PREFIX),
                    agent_id=data.agent_id,
                    content=data.content,
                    role=data.role,
                    embedding_json=_embedding_to_json(emb),
                    metadata_=data.metadata,
                )
            )
        async with self._session_factory() as session:
            session.add_all(models)
            await session.commit()
            return [self._to_schema(m) for m in models]

    async def get(self, passage_id: str) -> Passage:
        async with self._session_factory() as session:
            result = await session.execute(
                select(PassageModel).where(PassageModel.id == passage_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise PassageNotFoundError(passage_id)
            return self._to_schema(row)

    async def search(
        self,
        embedding: list[float],
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Passage, float]]:
        async with self._session_factory() as session:
            stmt = select(PassageModel)
            if agent_id:
                stmt = stmt.where(PassageModel.agent_id == agent_id)
            result = await session.execute(stmt)
            scored: list[tuple[Passage, float]] = []
            for row in result.scalars().all():
                emb = _json_to_embedding(row.embedding_json)
                if emb is None:
                    continue
                score = _cosine_similarity(embedding, emb)
                scored.append((self._to_schema(row), score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

    async def list_by_agent(
        self, agent_id: str, limit: int = 100, offset: int = 0
    ) -> list[Passage]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(PassageModel)
                .where(PassageModel.agent_id == agent_id)
                .order_by(PassageModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return [self._to_schema(r) for r in result.scalars().all()]

    async def delete(self, passage_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(PassageModel).where(PassageModel.id == passage_id)
            )
            await session.commit()
            return result.rowcount > 0

    @staticmethod
    def _to_schema(row: PassageModel) -> Passage:
        return Passage(
            id=row.id,
            agent_id=row.agent_id,
            content=row.content,
            role=row.role,
            embedding=_json_to_embedding(row.embedding_json),
            metadata=row.metadata_,
            created_at=row.created_at,
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


async def create_sqlite_stores(
    database_url: str = "sqlite+aiosqlite:///:memory:",
) -> tuple[SQLiteNodeStore, SQLiteEdgeStore, SQLiteBlockStore, SQLitePassageStore]:
    """Create all SQLite stores with a shared async engine.

    Creates tables automatically. Returns (node_store, edge_store, block_store, passage_store).
    """
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    return (
        SQLiteNodeStore(session_factory),
        SQLiteEdgeStore(session_factory),
        SQLiteBlockStore(session_factory),
        SQLitePassageStore(session_factory),
    )
