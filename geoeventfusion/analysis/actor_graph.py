"""Actor co-occurrence graph construction and analysis for GEOEventFusion.

Builds a NetworkX graph from actor co-occurrence triples, computes centrality
metrics (degree, betweenness, PageRank), and detects communities.

All NetworkX calls guard against empty graphs before computing centrality
or PageRank — both raise on empty graphs (per CLAUDE.md known gotcha #10.5).

Pure functions — no I/O or external calls.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def build_actor_graph(
    co_occurrence_triples: List[Tuple[str, str, str]],
    hub_top_n: int = 5,
    broker_ratio_threshold: float = 1.5,
    pagerank_max_iter: int = 200,
) -> "ActorGraph":  # type: ignore[name-defined]
    """Build an actor graph from co-occurrence triples and compute centrality metrics.

    Args:
        co_occurrence_triples: List of (actor_a, actor_b, date) tuples.
        hub_top_n: Number of top-degree actors to classify as Hub.
        broker_ratio_threshold: Betweenness/degree ratio threshold for Broker classification.
        pagerank_max_iter: Maximum iterations for PageRank computation.

    Returns:
        ActorGraph with nodes, edges, centrality metrics, and community detection.
    """
    from geoeventfusion.models.actors import (
        ActorEdge,
        ActorGraph,
        ActorNode,
        CentralityResult,
        CommunityResult,
        TemporalCommunityShift,
    )

    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx is required for actor graph construction")
        return ActorGraph()

    # Build edge weight accumulator
    edge_weights: dict = defaultdict(int)
    edge_dates: dict = defaultdict(list)
    for actor_a, actor_b, date in co_occurrence_triples:
        key = tuple(sorted([actor_a, actor_b]))
        edge_weights[key] += 1
        if date:
            edge_dates[key].append(date)

    # Construct NetworkX graph
    G = nx.Graph()
    for (a, b), weight in edge_weights.items():
        G.add_edge(a, b, weight=weight)

    if G.number_of_nodes() == 0:
        logger.info("Actor graph: no nodes — returning empty graph")
        return ActorGraph()

    # Remove isolated nodes (no edges)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    if G.number_of_nodes() == 0:
        return ActorGraph()

    # ── Centrality computation ─────────────────────────────────────────────────
    degree_centrality = nx.degree_centrality(G)

    # Betweenness centrality (weighted)
    try:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception as exc:
        logger.warning("Betweenness centrality failed: %s — using zeros", exc)
        betweenness = {n: 0.0 for n in G.nodes()}

    # PageRank (weighted)
    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=pagerank_max_iter)
    except Exception as exc:
        logger.warning("PageRank failed: %s — using degree centrality", exc)
        pagerank = degree_centrality

    # ── Role classification ────────────────────────────────────────────────────
    sorted_by_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    hub_names = {name for name, _ in sorted_by_degree[:hub_top_n]}

    def _classify_role(name: str) -> str:
        if name in hub_names:
            return "Hub"
        deg = degree_centrality.get(name, 0.0)
        bet = betweenness.get(name, 0.0)
        if deg > 0 and (bet / deg) >= broker_ratio_threshold:
            return "Broker"
        return "Peripheral"

    # ── Build node list ────────────────────────────────────────────────────────
    nodes = []
    node_by_name = {}
    centrality_results = []

    for name in G.nodes():
        role = _classify_role(name)
        node = ActorNode(
            name=name,
            role=role,
            mention_count=G.degree(name),
            degree_centrality=round(degree_centrality.get(name, 0.0), 6),
            betweenness_centrality=round(betweenness.get(name, 0.0), 6),
            pagerank=round(pagerank.get(name, 0.0), 8),
        )
        nodes.append(node)
        node_by_name[name] = node
        centrality_results.append(
            CentralityResult(
                actor=name,
                degree=node.degree_centrality,
                betweenness=node.betweenness_centrality,
                pagerank=node.pagerank,
                role=role,
            )
        )

    # ── Build edge list ────────────────────────────────────────────────────────
    edges = []
    for (a, b), weight in edge_weights.items():
        if a in G.nodes() and b in G.nodes():
            edges.append(
                ActorEdge(
                    actor_a=a,
                    actor_b=b,
                    weight=weight,
                    dates=sorted(set(edge_dates.get((a, b), []))),
                )
            )

    # ── Community detection ────────────────────────────────────────────────────
    communities = _detect_communities(G)
    community_results = [
        CommunityResult(
            community_id=i,
            members=sorted(comm),
            size=len(comm),
        )
        for i, comm in enumerate(communities)
    ]

    # Assign community_id to nodes
    for comm_result in community_results:
        for member in comm_result.members:
            if member in node_by_name:
                node_by_name[member].community_id = comm_result.community_id

    # ── Temporal community shift ───────────────────────────────────────────────
    temporal_shift = _compute_temporal_shift(
        co_occurrence_triples, hub_top_n, broker_ratio_threshold, pagerank_max_iter
    )

    return ActorGraph(
        nodes=nodes,
        edges=edges,
        centrality=centrality_results,
        communities=community_results,
        temporal_shift=temporal_shift,
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
        node_by_name=node_by_name,
    )


def _detect_communities(G: "nx.Graph") -> list:  # type: ignore[name-defined]
    """Run greedy modularity community detection on a graph.

    Args:
        G: NetworkX Graph.

    Returns:
        List of frozensets (each frozenset is a community of node names).
    """
    if G.number_of_nodes() == 0:
        return []
    try:
        import networkx.algorithms.community as nx_comm

        return list(nx_comm.greedy_modularity_communities(G))
    except Exception as exc:
        logger.warning("Community detection failed: %s", exc)
        return []


def _compute_temporal_shift(
    triples: List[Tuple[str, str, str]],
    hub_top_n: int,
    broker_ratio_threshold: float,
    pagerank_max_iter: int,
) -> Optional["TemporalCommunityShift"]:  # type: ignore[name-defined]
    """Measure community reorganization across early/mid/late thirds of the analysis window.

    Args:
        triples: Full list of (actor_a, actor_b, date) triples.
        hub_top_n: Hub classification threshold.
        broker_ratio_threshold: Broker classification threshold.
        pagerank_max_iter: PageRank iteration limit.

    Returns:
        TemporalCommunityShift with reorganization score, or None if insufficient data.
    """
    from geoeventfusion.models.actors import CommunityResult, TemporalCommunityShift

    dated = [(a, b, d) for a, b, d in triples if d]
    if len(dated) < 9:
        return None

    try:
        import networkx as nx

        dated_sorted = sorted(dated, key=lambda x: x[2])
        n = len(dated_sorted)
        thirds = [dated_sorted[:n // 3], dated_sorted[n // 3:2 * n // 3], dated_sorted[2 * n // 3:]]

        community_sets = []
        for segment in thirds:
            G_seg = nx.Graph()
            for a, b, _ in segment:
                G_seg.add_edge(a, b)
            if G_seg.number_of_nodes() > 0:
                communities = _detect_communities(G_seg)
                community_sets.append({frozenset(c) for c in communities})
            else:
                community_sets.append(set())

        # Jaccard reorganization score between early and late communities
        early, _, late = community_sets
        if not early or not late:
            return None

        # Member sets
        early_members = frozenset(m for comm in early for m in comm)
        late_members = frozenset(m for comm in late for m in comm)
        shared = early_members & late_members

        if not shared:
            return TemporalCommunityShift(reorganization_score=1.0)

        # Measure fraction of shared actors that changed communities
        changed = 0
        for member in shared:
            early_comm = next((c for c in early if member in c), frozenset())
            late_comm = next((c for c in late if member in c), frozenset())
            if early_comm != late_comm:
                changed += 1

        reorg_score = changed / len(shared) if shared else 0.0

        def _to_results(comms: set) -> List["CommunityResult"]:
            return [
                CommunityResult(community_id=i, members=sorted(c), size=len(c))
                for i, c in enumerate(comms)
            ]

        return TemporalCommunityShift(
            early_communities=_to_results(early),
            mid_communities=_to_results(community_sets[1]),
            late_communities=_to_results(late),
            reorganization_score=round(reorg_score, 4),
        )

    except Exception as exc:
        logger.warning("Temporal community shift computation failed: %s", exc)
        return None
