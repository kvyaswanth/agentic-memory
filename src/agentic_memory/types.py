"""Shared enums, constants, and type aliases."""

from __future__ import annotations

from enum import StrEnum

# -- ID prefixes -----------------------------------------------------------

NODE_PREFIX = "node"
EDGE_PREFIX = "edge"
BLOCK_PREFIX = "block"
PASSAGE_PREFIX = "pass"
AGENT_PREFIX = "agent"

# -- Default labels --------------------------------------------------------

DEFAULT_BLOCK_LABELS = ("persona", "human", "context")


# -- Enums -----------------------------------------------------------------


class MemoryTier(StrEnum):
    """Memory tier identifiers."""

    CORE = "core"
    RECALL = "recall"
    ARCHIVAL = "archival"


class TraversalStrategy(StrEnum):
    """Graph traversal algorithm selection."""

    BFS = "bfs"
    DFS = "dfs"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    RECENCY_WEIGHTED = "recency_weighted"


class EdgeDirection(StrEnum):
    """Edge direction filter for neighbor queries."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"
