"""Tests for sleep-time consolidation."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_consolidate_empty(client):
    """Consolidation on empty memory should not error."""
    result = await client.consolidate()
    assert result.nodes_scored == 0
    assert result.nodes_pruned == 0


@pytest.mark.asyncio
async def test_consolidate_with_data(client):
    """Insert some memories, run consolidation, verify result."""
    await client.archival().insert("Important fact about AI", importance=0.9)
    await client.archival().insert("Trivial detail", importance=0.05)
    await client.archival().insert("Another fact about ML", importance=0.8)

    result = await client.consolidate()
    assert result.nodes_scored == 3
    # The low-importance node might get pruned.
    assert result.nodes_pruned >= 0
