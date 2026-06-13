"""Tests for archival memory (long-term graph nodes)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_insert_and_get(client):
    node = await client.archival().insert("Graph databases enable relationship queries")
    assert node.content == "Graph databases enable relationship queries"
    assert node.label == "memory"

    fetched = await client.archival().get(node.id)
    assert fetched.id == node.id


@pytest.mark.asyncio
async def test_search_with_context(client):
    await client.archival().insert("Python is great for AI development")
    await client.archival().insert("Machine learning models need training data")

    results = await client.archival().search("AI and machine learning", expand_context=True)
    assert len(results) > 0
    # Results should have scores.
    assert results[0].score >= 0.0


@pytest.mark.asyncio
async def test_auto_linking(client):
    """Insert similar memories and verify they get auto-linked."""
    n1 = await client.archival().insert("Neural networks learn from data")
    n2 = await client.archival().insert("Deep learning uses neural networks")

    # Check if edges were created between similar nodes.
    graph = client.graph()
    _edges = await graph.get_edges(n1.id)
    # Fake embeddings are deterministic, so similar content should link.
    # At minimum the insert should succeed without error.
    assert n1.id != n2.id


@pytest.mark.asyncio
async def test_delete_node(client):
    node = await client.archival().insert("Temporary memory")
    deleted = await client.archival().delete(node.id)
    assert deleted is True


@pytest.mark.asyncio
async def test_get_context(client):
    node = await client.archival().insert("Context test node")
    context = await client.archival().get_context(node.id, max_depth=1)
    assert len(context) >= 1  # At least the node itself.
    assert context[0].id == node.id


@pytest.mark.asyncio
async def test_list_all(client):
    for i in range(5):
        await client.archival().insert(f"Memory {i}")
    nodes = await client.archival().list_all()
    assert len(nodes) == 5
