# Agentic Memory

**Graph-based memory layer with sleep-time compute for stateful AI agents.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Agentic Memory is a standalone, pip-installable Python library that gives AI agents persistent, structured memory. It combines a **3-tier memory architecture** (inspired by [Letta/MemGPT](https://github.com/letta-ai/letta)) with a custom **graph engine** (HydraDB) for relationship-aware retrieval and **sleep-time compute** for background memory consolidation.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  AI Agent                        │
├─────────────────────────────────────────────────┤
│  Tool Executor  │  Context Builder              │
├────────┬────────┴──────────┬────────────────────┤
│  Core  │   Recall          │   Archival         │
│ Memory │   Memory          │   Memory           │
│ (blocks│   (conversations) │   (graph nodes)    │
│ in-ctx)│                   │                    │
├────────┴───────────────────┴──────┬─────────────┤
│           HydraDB Graph Engine    │ Sleep-Time  │
│  (nodes, edges, traversal,       │ Compute      │
│   auto-linking, clustering)      │ (scheduler,  │
│                                  │  consolidator│
├──────────────────────────────────┴──────────────┤
│           Storage Backends                       │
│  SQLite (dev) │ PostgreSQL + pgvector (prod)     │
└─────────────────────────────────────────────────┘
```

## Quick Start

```python
import asyncio
from agentic_memory import AgenticMemoryClient, MemoryConfig

async def main():
    client = AgenticMemoryClient(MemoryConfig(backend="sqlite"))
    await client.initialize()

    # Core memory (always in the agent's prompt)
    await client.core().create("persona", "A helpful AI research assistant")
    await client.core().create("human", "Name: Alice, ML engineer")

    # Archival memory (long-term graph nodes)
    node = await client.archival().insert(
        "Alice presented a graph-based memory architecture at AIConf 2025"
    )

    # Semantic search with graph context expansion
    results = await client.archival().search("Alice's graph research", expand_context=True)
    for r in results:
        print(f"[{r.score:.3f}] {r.node.content}")

    # Sleep-time consolidation (score, cluster, summarize, prune)
    report = await client.consolidate()
    print(f"Pruned {report.nodes_pruned} nodes, created {report.summaries_created} summaries")

    await client.shutdown()

asyncio.run(main())
```

## Installation

```bash
# Core (SQLite backend, fake embeddings)
pip install agentic-memory

# With OpenAI embeddings
pip install agentic-memory[openai]

# With local embeddings (no API key needed)
pip install agentic-memory[local-embeddings]

# Everything
pip install agentic-memory[all]

# Development
pip install -e ".[dev]"
```

## Memory Tiers

### Core Memory
In-context blocks that live in the agent's system prompt. Always available, no retrieval needed.

```python
await client.core().create("persona", "You are a helpful assistant")
await client.core().append("human", "User prefers concise answers")
await client.core().replace("persona", "helpful", "precise")
print(client.core().compile())  # XML string for prompt injection
```

### Recall Memory
Searchable conversation history. Each message is embedded for semantic retrieval.

```python
await client.recall().add("What is graph RAG?", "user")
await client.recall().add("Graph RAG uses knowledge graphs...", "assistant")
results = await client.recall().search("explain RAG")
```

### Archival Memory
Long-term persistent storage backed by graph nodes. Each memory is automatically linked to semantically similar existing nodes.

```python
node = await client.archival().insert("Graph databases enable relationship queries")
results = await client.archival().search("relationship queries", expand_context=True)
# Each result includes graph neighbors for richer context
```

## Graph Engine (HydraDB)

The core differentiator. Every memory is a node; semantically similar nodes are connected via edges.

```python
graph = client.graph()

# Add memories (auto-linked to similar nodes)
n1 = await graph.add_memory("Python is great for AI")
n2 = await graph.add_memory("PyTorch is the leading ML framework")

# Traversal
neighbors = await graph.traverse(n1.id, strategy="importance_weighted")

# Semantic search with graph expansion
results = await graph.semantic_search("machine learning", expand_context=True)

# Manual edges
await graph.add_edge(n1.id, n2.id, relation="related_to")

# Clustering (groups related memories)
clusters = await graph.cluster_memories()
```

**Traversal strategies:** `BFS`, `DFS`, `IMPORTANCE_WEIGHTED`, `RECENCY_WEIGHTED`

## Sleep-Time Compute

Background memory consolidation that runs during idle cycles, inspired by how human sleep consolidates memories.

**6-phase pipeline:**
1. **Score** all nodes (importance + time decay + access frequency)
2. **Cluster** related memories (connected components)
3. **Summarize** clusters into higher-level nodes
4. **Prune** below-threshold memories
5. **Strengthen** frequently-traversed edges
6. **Update** core memory blocks

```python
# Manual trigger
report = await client.consolidate()

# Automatic (configurable interval)
config = MemoryConfig(consolidation_interval_seconds=300)
```

## Configuration

All settings configurable via environment variables with `AGENTIC_MEMORY_` prefix:

| Setting | Default | Description |
|---|---|---|
| `BACKEND` | `sqlite` | Storage backend (`sqlite` or `postgres`) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./memory.db` | Database connection URL |
| `EMBEDDING_PROVIDER` | `fake` | `openai`, `sentence_transformers`, or `fake` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIMENSION` | `1536` | Vector dimension |
| `AUTO_LINK_THRESHOLD` | `0.85` | Cosine similarity threshold for auto-linking |
| `CONSOLIDATION_INTERVAL_SECONDS` | `300` | Sleep-time consolidation interval |
| `IMPORTANCE_DECAY_HALF_LIFE_HOURS` | `168` | Memory importance half-life (1 week) |
| `PRUNING_THRESHOLD` | `0.1` | Score below which memories are pruned |
| `DEFAULT_BLOCK_CHAR_LIMIT` | `2000` | Core memory block size limit |

## Agent Integration

### Tool Definitions (OpenAI Function Calling)

```python
schemas = AgenticMemoryClient.tool_schemas()
# Returns JSON schemas for: core_memory_append, core_memory_replace,
# archival_insert, archival_search, recall_search, graph_search, graph_add_relation

executor = client.tools()
result = await executor.execute("archival_search", {"query": "graph databases"})
```

### Context Builder

```python
builder = client.context_builder()
core_xml = builder.build_core_section()
archival_summary = await builder.build_archival_summary(client.archival())
system_prompt = builder.build_system_prompt(archival_summary, core_xml)
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # Run tests
ruff check src/           # Lint
mypy src/                 # Type check
```

## License

Apache 2.0
