"""Pluggable embedding providers."""

from agentic_memory.embeddings.base import BaseEmbeddingProvider
from agentic_memory.embeddings.fake import FakeEmbedding

__all__ = ["BaseEmbeddingProvider", "FakeEmbedding"]
