"""OpenAI embedding provider."""

from __future__ import annotations

from agentic_memory.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbedding(BaseEmbeddingProvider):
    """Embeddings via the OpenAI API.

    Requires the ``openai`` extra: ``pip install agentic-memory[openai]``
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install agentic-memory[openai]"
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            input=text, model=self._model, dimensions=self._dimension
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self._dimension
        )
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    @property
    def dimension(self) -> int:
        return self._dimension
