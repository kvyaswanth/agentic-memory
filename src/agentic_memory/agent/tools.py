"""Agent-callable tool definitions compatible with OpenAI function calling.

These tool schemas can be registered with any LLM that supports function
calling. The ``execute_tool`` function dispatches to the appropriate memory
operation.
"""

from __future__ import annotations

from agentic_memory.memory.archival import ArchivalMemory
from agentic_memory.memory.core import CoreMemory
from agentic_memory.memory.recall import RecallMemory
from agentic_memory.schemas.search import NodeSearchResult, PassageSearchResult

# -- Tool JSON schemas -----------------------------------------------------

MEMORY_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": "Append content to a core memory block (persona, human, context).",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Block label (e.g. 'persona', 'human', 'context')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append",
                    },
                },
                "required": ["label", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": "Replace a substring within a core memory block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Block label",
                    },
                    "old": {
                        "type": "string",
                        "description": "Text to find",
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                },
                "required": ["label", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_insert",
            "description": "Insert a new long-term memory with automatic graph linking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Memory text to store",
                    },
                    "label": {
                        "type": "string",
                        "description": "Node label (default: 'memory')",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score 0.0-1.0 (default: 0.5)",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_search",
            "description": "Search long-term archival memory by semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_search",
            "description": "Search conversation history by semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Search the memory graph with automatic context expansion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                    },
                    "expand_context": {
                        "type": "boolean",
                        "description": "Expand results with graph neighbors (default: true)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_add_relation",
            "description": "Create an explicit relationship between two memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source memory node ID",
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Target memory node ID",
                    },
                    "relation": {
                        "type": "string",
                        "description": "Relationship type (e.g. 'caused', 'related_to')",
                    },
                },
                "required": ["source_id", "target_id", "relation"],
            },
        },
    },
]


# -- Tool executor ---------------------------------------------------------

class ToolExecutor:
    """Dispatches tool calls to the appropriate memory tier."""

    def __init__(
        self,
        core: CoreMemory,
        recall: RecallMemory,
        archival: ArchivalMemory,
    ) -> None:
        self._core = core
        self._recall = recall
        self._archival = archival

    async def execute(self, tool_name: str, args: dict) -> str:
        """Execute a named tool with the given arguments. Returns a text result."""
        if tool_name == "core_memory_append":
            block = await self._core.append(args["label"], args["content"])
            return f"Appended to block '{block.label}' (now {len(block.value)} chars)"

        elif tool_name == "core_memory_replace":
            block = await self._core.replace(args["label"], args["old"], args["new"])
            return f"Updated block '{block.label}' (now {len(block.value)} chars)"

        elif tool_name == "archival_insert":
            node = await self._archival.insert(
                text=args["content"],
                label=args.get("label", "memory"),
                importance=args.get("importance", 0.5),
            )
            return f"Inserted memory node '{node.id}'"

        elif tool_name == "archival_search":
            results = await self._archival.search(
                query=args["query"],
                limit=args.get("limit", 10),
            )
            return _format_node_results(results)

        elif tool_name == "recall_search":
            results = await self._recall.search(
                query=args["query"],
                limit=args.get("limit", 10),
            )
            return _format_passage_results(results)

        elif tool_name == "graph_search":

            # GraphEngine is accessed through archival's internal references.
            # For direct graph search, we use archival.search with expand_context.
            results = await self._archival.search(
                query=args["query"],
                limit=args.get("limit", 10),
                expand_context=args.get("expand_context", True),
            )
            return _format_node_results(results)

        elif tool_name == "graph_add_relation":
            edge = await self._archival._edges.create(
                __import__("agentic_memory.schemas.edge", fromlist=["EdgeCreate"]).EdgeCreate(
                    source_node_id=args["source_id"],
                    target_node_id=args["target_id"],
                    relation=args["relation"],
                )
            )
            return f"Created edge '{edge.id}' ({args['relation']})"

        else:
            raise ValueError(f"Unknown tool: {tool_name}")


def _format_node_results(results: list[NodeSearchResult]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [score={r.score:.3f}] {r.node.content[:200]}")
        if r.context:
            lines.append(f"   Context: {len(r.context)} related nodes")
    return "\n".join(lines)


def _format_passage_results(results: list[PassageSearchResult]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r.passage.role}] [score={r.score:.3f}] {r.passage.content[:200]}")
    return "\n".join(lines)
