"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Pluggable embedding provider.

    Implementations must provide async ``embed`` and ``embed_batch`` methods
    and declare their output vector dimension.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...
