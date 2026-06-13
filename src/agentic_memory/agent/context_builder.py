"""Build LLM prompt sections from memory state.

Compiles core memory blocks, archival summaries, and search results into
structured text suitable for injection into the system prompt or conversation.
"""

from __future__ import annotations

from agentic_memory.memory.archival import ArchivalMemory
from agentic_memory.memory.core import CoreMemory
from agentic_memory.schemas.search import NodeSearchResult, PassageSearchResult
from agentic_memory.utils.logging import get_logger
from agentic_memory.utils.token_counting import estimate_tokens

logger = get_logger("context_builder")


class ContextBuilder:
    """Build LLM prompt sections from memory state."""

    def __init__(
        self,
        core: CoreMemory,
        max_context_tokens: int = 4000,
    ) -> None:
        self._core = core
        self._max_tokens = max_context_tokens

    def build_core_section(self) -> str:
        """Render core memory blocks as XML for prompt injection.

        Output::

            <memory_blocks>
              <block label="persona">...</block>
              <block label="human">...</block>
            </memory_blocks>
        """
        return self._core.compile()

    async def build_archival_summary(self, archival: ArchivalMemory) -> str:
        """Build a brief summary of archival memory stats for the system prompt."""
        nodes = await archival.list_all(limit=1_000)
        total = len(nodes)
        if total == 0:
            return "<archival_memory count=\"0\" />"

        labels: dict[str, int] = {}
        for n in nodes:
            labels[n.label] = labels.get(n.label, 0) + 1

        parts = [f'<archival_memory count="{total}">']
        for label, count in sorted(labels.items()):
            parts.append(f"  <label name=\"{label}\" count=\"{count}\" />")
        parts.append("</archival_memory>")
        return "\n".join(parts)

    def build_retrieved_context(
        self,
        results: list[NodeSearchResult] | list[PassageSearchResult],
        max_tokens: int | None = None,
    ) -> str:
        """Format search results for injection into conversation.

        Truncates output to fit within the token budget.
        """
        budget = max_tokens or self._max_tokens
        lines: list[str] = ["<retrieved_memories>"]
        token_count = estimate_tokens(lines[0])

        for i, result in enumerate(results, 1):
            if isinstance(result, NodeSearchResult):
                entry = (
                    f"  <memory index=\"{i}\" score=\"{result.score:.3f}\" "
                    f"id=\"{result.node.id}\">\n"
                    f"    {result.node.content[:500]}\n"
                    f"  </memory>"
                )
            else:
                entry = (
                    f"  <passage index=\"{i}\" score=\"{result.score:.3f}\" "
                    f"role=\"{result.passage.role}\">\n"
                    f"    {result.passage.content[:500]}\n"
                    f"  </passage>"
                )

            entry_tokens = estimate_tokens(entry)
            if token_count + entry_tokens > budget:
                lines.append(f"  <!-- truncated: {len(results) - i + 1} more results -->")
                break

            lines.append(entry)
            token_count += entry_tokens

        lines.append("</retrieved_memories>")
        return "\n".join(lines)

    def build_system_prompt(self, archival_summary: str, core_section: str | None = None) -> str:
        """Assemble the full system prompt from memory components."""
        parts: list[str] = []

        core = core_section or self.build_core_section()
        if core.strip() != "<memory_blocks />":
            parts.append(core)

        if archival_summary.strip() != '<archival_memory count="0" />':
            parts.append(archival_summary)

        return "\n\n".join(parts)
