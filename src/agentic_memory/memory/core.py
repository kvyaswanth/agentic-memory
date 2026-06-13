"""Core memory tier: in-context blocks that live in the agent's prompt.

Mirrors Letta's ``BasicBlockMemory``. Each block has a label (e.g. "persona",
"human", "context") and a text value that is compiled into the system prompt.
"""

from __future__ import annotations

from agentic_memory.exceptions import BlockNotFoundError, MemoryLimitExceededError
from agentic_memory.schemas.memory_block import BlockCreate, MemoryBlock
from agentic_memory.storage.base import BlockStore
from agentic_memory.utils.logging import get_logger

logger = get_logger("core")


class CoreMemory:
    """In-context memory blocks always available to the agent.

    Blocks are stored persistently via ``BlockStore`` but also cached locally
    for fast synchronous reads. Mutations go through the store and update the
    local cache.
    """

    def __init__(self, agent_id: str, block_store: BlockStore) -> None:
        self._agent_id = agent_id
        self._store = block_store
        self._blocks: dict[str, MemoryBlock] = {}

    async def load(self) -> None:
        """Load all blocks for this agent from the store into local cache."""
        blocks = await self._store.get_by_agent(self._agent_id)
        self._blocks = {b.label: b for b in blocks}
        logger.debug("loaded %d core memory blocks", len(self._blocks))

    def get(self, label: str) -> MemoryBlock:
        """Get a block by label (sync, from local cache)."""
        if label not in self._blocks:
            raise BlockNotFoundError(label)
        return self._blocks[label]

    def get_all(self) -> dict[str, MemoryBlock]:
        """Return all blocks keyed by label."""
        return dict(self._blocks)

    async def create(self, label: str, value: str, limit: int = 2000) -> MemoryBlock:
        """Create a new core memory block."""
        if len(value) > limit:
            raise MemoryLimitExceededError(
                f"Block content ({len(value)} chars) exceeds limit ({limit} chars)"
            )
        block = await self._store.put(
            BlockCreate(
                agent_id=self._agent_id,
                label=label,
                value=value,
                limit=limit,
            )
        )
        self._blocks[label] = block
        logger.debug("created core block '%s' (%d chars)", label, len(value))
        return block

    async def update(self, label: str, value: str) -> MemoryBlock:
        """Replace the entire value of a block."""
        block = self.get(label)
        if block.read_only:
            raise PermissionError(f"Block '{label}' is read-only")
        if len(value) > block.limit:
            raise MemoryLimitExceededError(
                f"Block content ({len(value)} chars) exceeds limit ({block.limit} chars)"
            )
        updated = await self._store.update_value(block.id, value)
        self._blocks[label] = updated
        logger.debug("updated core block '%s'", label)
        return updated

    async def append(self, label: str, content: str) -> MemoryBlock:
        """Append content to an existing block."""
        block = self.get(label)
        new_value = block.value + "\n" + content
        return await self.update(label, new_value)

    async def replace(self, label: str, old: str, new: str) -> MemoryBlock:
        """Replace a substring within a block (mirrors Letta's core_memory_replace)."""
        block = self.get(label)
        if old not in block.value:
            raise ValueError(f"Substring '{old}' not found in block '{label}'")
        new_value = block.value.replace(old, new, 1)
        return await self.update(label, new_value)

    def compile(self) -> str:
        """Render all blocks into an XML string for prompt injection.

        Output format::

            <memory_blocks>
              <block label="persona">...</block>
              <block label="human">...</block>
            </memory_blocks>
        """
        if not self._blocks:
            return "<memory_blocks />\n"
        lines = ["<memory_blocks>"]
        for label, block in sorted(self._blocks.items()):
            lines.append(f'  <block label="{label}">{block.value}</block>')
        lines.append("</memory_blocks>")
        return "\n".join(lines)

    def to_messages(self) -> list[dict[str, str]]:
        """Export as a system message payload for LLM APIs."""
        compiled = self.compile()
        if compiled.strip() == "<memory_blocks />":
            return []
        return [{"role": "system", "content": compiled}]
