"""Abstract storage interfaces for all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_memory.schemas.edge import EdgeCreate, MemoryEdge
from agentic_memory.schemas.memory_block import BlockCreate, MemoryBlock
from agentic_memory.schemas.node import MemoryNode, NodeCreate, NodeUpdate
from agentic_memory.schemas.passage import Passage, PassageCreate
from agentic_memory.types import EdgeDirection


class NodeStore(ABC):
    """Async CRUD + vector search for memory nodes."""

    @abstractmethod
    async def get(self, node_id: str) -> MemoryNode:
        ...

    @abstractmethod
    async def put(self, data: NodeCreate, embedding: list[float] | None = None) -> MemoryNode:
        ...

    @abstractmethod
    async def batch_put(
        self, items: list[tuple[NodeCreate, list[float] | None]]
    ) -> list[MemoryNode]:
        ...

    @abstractmethod
    async def update(self, node_id: str, data: NodeUpdate) -> MemoryNode:
        ...

    @abstractmethod
    async def delete(self, node_id: str) -> bool:
        ...

    @abstractmethod
    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[MemoryNode]:
        ...

    @abstractmethod
    async def get_neighbors(
        self, node_id: str, direction: EdgeDirection = EdgeDirection.BOTH, max_depth: int = 1
    ) -> list[MemoryNode]:
        ...

    @abstractmethod
    async def search_by_embedding(
        self, embedding: list[float], agent_id: str | None = None, limit: int = 10
    ) -> list[tuple[MemoryNode, float]]:
        ...


class EdgeStore(ABC):
    """Async CRUD for graph edges."""

    @abstractmethod
    async def create(self, data: EdgeCreate) -> MemoryEdge:
        ...

    @abstractmethod
    async def get(self, edge_id: str) -> MemoryEdge:
        ...

    @abstractmethod
    async def get_edges(
        self, node_id: str, direction: EdgeDirection = EdgeDirection.BOTH
    ) -> list[MemoryEdge]:
        ...

    @abstractmethod
    async def get_edges_by_relation(self, relation: str, agent_id: str) -> list[MemoryEdge]:
        ...

    @abstractmethod
    async def update_weight(self, edge_id: str, weight: float) -> MemoryEdge:
        ...

    @abstractmethod
    async def delete(self, edge_id: str) -> bool:
        ...


class BlockStore(ABC):
    """Async CRUD for core memory blocks."""

    @abstractmethod
    async def get(self, block_id: str) -> MemoryBlock:
        ...

    @abstractmethod
    async def get_by_label(self, agent_id: str, label: str) -> MemoryBlock | None:
        ...

    @abstractmethod
    async def put(self, data: BlockCreate) -> MemoryBlock:
        ...

    @abstractmethod
    async def update_value(self, block_id: str, value: str) -> MemoryBlock:
        ...

    @abstractmethod
    async def get_by_agent(self, agent_id: str) -> list[MemoryBlock]:
        ...

    @abstractmethod
    async def delete(self, block_id: str) -> bool:
        ...


class PassageStore(ABC):
    """Async CRUD + vector search for recall passages."""

    @abstractmethod
    async def put(
        self, data: PassageCreate, embedding: list[float] | None = None
    ) -> Passage:
        ...

    @abstractmethod
    async def batch_put(
        self, items: list[tuple[PassageCreate, list[float] | None]]
    ) -> list[Passage]:
        ...

    @abstractmethod
    async def get(self, passage_id: str) -> Passage:
        ...

    @abstractmethod
    async def search(
        self, embedding: list[float], agent_id: str | None = None, limit: int = 10
    ) -> list[tuple[Passage, float]]:
        ...

    @abstractmethod
    async def list_by_agent(
        self, agent_id: str, limit: int = 100, offset: int = 0
    ) -> list[Passage]:
        ...

    @abstractmethod
    async def delete(self, passage_id: str) -> bool:
        ...
