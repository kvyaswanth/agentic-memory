"""Tests for the HydraDB graph engine."""

from __future__ import annotations

import pytest

from agentic_memory.types import TraversalStrategy


@pytest.mark.asyncio
async def test_add_memory(client):
    node = await client.graph().add_memory("Test memory node")
    assert node.content == "Test memory node"
    assert node.id.startswith("node-")


@pytest.mark.asyncio
async def test_add_edge(client):
    n1 = await client.graph().add_memory("Source node")
    n2 = await client.graph().add_memory("Target node")
    edge = await client.graph().add_edge(n1.id, n2.id, relation="caused")
    assert edge.relation == "caused"
    assert edge.source_node_id == n1.id


@pytest.mark.asyncio
async def test_semantic_search(client):
    await client.graph().add_memory("Graph databases store relationships")
    await client.graph().add_memory("Python is a programming language")

    results = await client.graph().semantic_search("databases", expand_context=True)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_traverse_bfs(client):
    n1 = await client.graph().add_memory("Root")
    n2 = await client.graph().add_memory("Child 1")
    n3 = await client.graph().add_memory("Child 2")
    await client.graph().add_edge(n1.id, n2.id)
    await client.graph().add_edge(n1.id, n3.id)

    neighbors = await client.graph().traverse(n1.id, strategy=TraversalStrategy.BFS, max_depth=1)
    assert len(neighbors) >= 2


@pytest.mark.asyncio
async def test_auto_link(client):
    n1 = await client.graph().add_memory("Machine learning basics")
    edges_before = await client.graph().get_edges(n1.id)

    n2 = await client.graph().add_memory("Machine learning advanced")
    edges_after = await client.graph().get_edges(n2.id)

    # Auto-linking should have created at least one edge.
    assert len(edges_after) >= len(edges_before)


@pytest.mark.asyncio
async def test_cluster_memories(client):
    n1 = await client.graph().add_memory("Topic A part 1")
    n2 = await client.graph().add_memory("Topic A part 2")
    await client.graph().add_edge(n1.id, n2.id, relation="related_to")

    clusters = await client.graph().cluster_memories()
    assert len(clusters) >= 1


@pytest.mark.asyncio
async def test_merge_nodes(client):
    n1 = await client.graph().add_memory("Duplicate A")
    n2 = await client.graph().add_memory("Duplicate B")
    await client.graph().add_edge(n1.id, n2.id, relation="duplicate")

    merged = await client.graph().merge_nodes(n1.id, n2.id)
    assert merged.id == n2.id


@pytest.mark.asyncio
async def test_score_all_nodes(client):
    await client.graph().add_memory("Node 1", importance=0.9)
    await client.graph().add_memory("Node 2", importance=0.1)

    scores = await client.graph().score_all_nodes()
    assert len(scores) == 2
    # Scores should be floats between 0 and 1.
    for _node_id, score in scores.items():
        assert 0.0 <= score <= 1.0
