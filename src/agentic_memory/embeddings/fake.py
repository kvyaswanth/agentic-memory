"""Deterministic fake embeddings for testing."""

from __future__ import annotations

import hashlib

from agentic_memory.embeddings.base import BaseEmbeddingProvider


class FakeEmbedding(BaseEmbeddingProvider):
    """Produces deterministic unit vectors from text content.

    Uses SHA-256 hashing to generate repeatable vectors. Not meaningful
    for semantic search but perfect for unit tests.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic pseudo-random unit vector.

        Uses the integer value of SHA-256 bytes (mod 10000) mapped to
        [-1, 1] to avoid NaN/Inf issues from raw float reinterpretation.
        """
        raw = []
        for i in range(self._dimension):
            h = hashlib.sha256(f"{text}:{i}".encode()).digest()
            # Use first 4 bytes as an integer, map to [-1, 1].
            int_val = int.from_bytes(h[:4], "big")
            val = (int_val % 10000) / 5000.0 - 1.0  # Range: [-1.0, 1.0]
            raw.append(val)
        # Normalize to unit length.
        norm = sum(x * x for x in raw) ** 0.5
        if norm == 0:
            norm = 1.0
        return [x / norm for x in raw]
