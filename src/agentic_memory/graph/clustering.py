"""Graph clustering for grouping related memories.

Uses connected-component analysis over the memory graph to identify clusters
of related nodes. Used by the sleep-time consolidation pipeline.
"""

from __future__ import annotations

from collections import defaultdict

from agentic_memory.schemas.node import MemoryNode
from agentic_memory.storage.base import EdgeStore
from agentic_memory.utils.logging import get_logger

logger = get_logger("clustering")


async def find_clusters(
    nodes: list[MemoryNode],
    edge_store: EdgeStore,
    min_cluster_size: int = 2,
    max_cluster_size: int = 20,
) -> list[list[MemoryNode]]:
    """Group nodes into connected components via Union-Find.

    Returns a list of clusters, each a list of ``MemoryNode``.
    Clusters smaller than *min_cluster_size* are discarded.
    """
    if not nodes:
        return []

    node_map: dict[str, MemoryNode] = {n.id: n for n in nodes}
    node_ids = set(node_map.keys())

    # Union-Find
    parent: dict[str, str] = {nid: nid for nid in node_ids}
    rank: dict[str, int] = {nid: 0 for nid in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Gather all edges between our nodes.
    for node in nodes:
        edges = await edge_store.get_edges(node.id)
        for edge in edges:
            if edge.source_node_id in node_ids and edge.target_node_id in node_ids:
                union(edge.source_node_id, edge.target_node_id)

    # Group by root.
    groups: dict[str, list[str]] = defaultdict(list)
    for nid in node_ids:
        groups[find(nid)].append(nid)

    # Convert to node lists, filter by size.
    clusters: list[list[MemoryNode]] = []
    for member_ids in groups.values():
        if len(member_ids) < min_cluster_size:
            continue
        cluster = [node_map[nid] for nid in member_ids if nid in node_map]
        # Cap cluster size to max_cluster_size (take highest importance first).
        cluster.sort(key=lambda n: n.importance, reverse=True)
        cluster = cluster[:max_cluster_size]
        clusters.append(cluster)

    logger.debug("found %d clusters from %d nodes", len(clusters), len(nodes))
    return clusters
