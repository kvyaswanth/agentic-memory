"""Local sentence-transformers embedding provider."""

from __future__ import annotations

from agentic_memory.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerEmbedding(BaseEmbeddingProvider):
    """Embeddings via sentence-transformers (runs locally, no API key needed).

    Requires the ``local-embeddings`` extra: ``pip install agentic-memory[local-embeddings]``
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install agentic-memory[local-embeddings]"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension() or 384

    async def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
