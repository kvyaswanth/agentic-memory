"""Tests for recall memory (conversation history)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_add_and_search(client):
    await client.recall().add("What is graph RAG?", "user")
    await client.recall().add("Graph RAG uses knowledge graphs for retrieval", "assistant")
    await client.recall().add("What is the weather?", "user")

    results = await client.recall().search("graph RAG")
    assert len(results) > 0
    assert results[0].passage.role in ("user", "assistant")


@pytest.mark.asyncio
async def test_list_recent(client):
    for i in range(5):
        await client.recall().add(f"Message {i}", "user")

    recent = await client.recall().list_recent(limit=3)
    assert len(recent) <= 3


@pytest.mark.asyncio
async def test_search_by_date(client):
    await client.recall().add("Old message", "user")
    from datetime import UTC, datetime, timedelta
    start = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    end = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)
    results = await client.recall().search_by_date(start=start, end=end)
    assert len(results) >= 1
