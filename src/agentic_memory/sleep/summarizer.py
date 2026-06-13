"""Memory summarization for consolidation.

Summarizes clusters of related memories into concise higher-level nodes.
Falls back to extractive summarization when no LLM is configured.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_memory.schemas.node import MemoryNode
from agentic_memory.utils.logging import get_logger

logger = get_logger("summarizer")


class Summarizer:
    """Summarize groups of related memories into concise text.

    If an LLM callable is provided, uses it for abstractive summarization.
    Otherwise falls back to extractive (first N words of each node).
    """

    def __init__(
        self,
        llm_callable: Callable | None = None,
        max_words: int = 100,
    ) -> None:
        self._llm = llm_callable
        self._max_words = max_words

    async def summarize(self, nodes: list[MemoryNode]) -> str:
        """Summarize a cluster of memory nodes into a single string."""
        if not nodes:
            return ""

        if self._llm is not None:
            return await self._abstractive(nodes)

        return self._extractive(nodes)

    async def _abstractive(self, nodes: list[MemoryNode]) -> str:
        """Use an LLM to generate an abstractive summary."""
        contents = "\n".join(f"- {n.content}" for n in nodes)
        prompt = (
            f"Summarize the following memories into a single concise paragraph "
            f"(max {self._max_words} words). Preserve key facts and relationships:\n\n"
            f"{contents}"
        )
        try:
            result = await self._llm(prompt)  # type: ignore[misc]
            return str(result).strip()
        except Exception as exc:
            logger.warning("LLM summarization failed, falling back to extractive: %s", exc)
            return self._extractive(nodes)

    def _extractive(self, nodes: list[MemoryNode]) -> str:
        """Extractive fallback: join first N words from each node."""
        parts: list[str] = []
        total_words = 0
        for node in sorted(nodes, key=lambda n: n.importance, reverse=True):
            words = node.content.split()
            remaining = self._max_words - total_words
            if remaining <= 0:
                break
            chunk = " ".join(words[:remaining])
            parts.append(chunk)
            total_words += len(words[:remaining])

        summary = "; ".join(parts)
        logger.debug("extractive summary: %d words from %d nodes", total_words, len(nodes))
        return summary
