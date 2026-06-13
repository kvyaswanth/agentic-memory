"""Tests for core memory (in-context blocks)."""

from __future__ import annotations

import pytest

from agentic_memory.exceptions import BlockNotFoundError, MemoryLimitExceededError


@pytest.mark.asyncio
async def test_create_and_get(client):
    block = await client.core().create("persona", "A helpful AI assistant")
    assert block.label == "persona"
    assert block.value == "A helpful AI assistant"

    fetched = client.core().get("persona")
    assert fetched.id == block.id


@pytest.mark.asyncio
async def test_append(client):
    await client.core().create("human", "Name: Alice")
    updated = await client.core().append("human", "Role: Engineer")
    assert "Alice" in updated.value
    assert "Engineer" in updated.value


@pytest.mark.asyncio
async def test_replace(client):
    await client.core().create("persona", "A helpful assistant")
    updated = await client.core().replace("persona", "helpful", "precise")
    assert "precise" in updated.value
    assert "helpful" not in updated.value


@pytest.mark.asyncio
async def test_char_limit_enforced(client):
    await client.core().create("test", "x" * 100, limit=100)
    with pytest.raises(MemoryLimitExceededError):
        await client.core().update("test", "x" * 101)


@pytest.mark.asyncio
async def test_get_nonexistent_raises(client):
    with pytest.raises(BlockNotFoundError):
        client.core().get("nonexistent")


@pytest.mark.asyncio
async def test_compile_xml(client):
    await client.core().create("persona", "Assistant")
    await client.core().create("human", "Alice")
    xml = client.core().compile()
    assert "<memory_blocks>" in xml
    assert 'label="persona"' in xml
    assert 'label="human"' in xml
    assert "Assistant" in xml


@pytest.mark.asyncio
async def test_to_messages(client):
    await client.core().create("context", "Test context")
    msgs = client.core().to_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert "Test context" in msgs[0]["content"]
