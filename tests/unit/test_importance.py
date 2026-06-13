"""Tests for importance scoring."""

from __future__ import annotations

import pytest

from agentic_memory.graph.importance import compute_importance


@pytest.mark.asyncio
async def test_importance_bounds(client):
    """Importance scores should be between 0.0 and 1.0."""

    from agentic_memory.schemas.node import MemoryNode

    node = MemoryNode(
        id="test-node",
        agent_id="test-agent",
        label="test",
        content="Test content",
        importance=0.5,
        access_count=10,
    )
    score = await compute_importance(node, edge_count=5)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_high_access_higher_score(client):
    """Nodes with more accesses should score higher (all else equal)."""
    from agentic_memory.schemas.node import MemoryNode

    low_access = MemoryNode(
        id="low", agent_id="a", label="t", content="x",
        importance=0.5, access_count=1,
    )
    high_access = MemoryNode(
        id="high", agent_id="a", label="t", content="x",
        importance=0.5, access_count=100,
    )
    s1 = await compute_importance(low_access, edge_count=0)
    s2 = await compute_importance(high_access, edge_count=0)
    assert s2 >= s1
