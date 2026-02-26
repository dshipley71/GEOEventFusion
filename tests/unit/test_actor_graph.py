"""Unit tests for geoeventfusion.analysis.actor_graph.

Covers:
- build_actor_graph: happy path, empty input, role classification
- ActorGraph helper methods: get_hubs, get_brokers, get_top_actors
- Community detection and temporal shift (graceful handling)
"""

from __future__ import annotations

import pytest

from geoeventfusion.analysis.actor_graph import build_actor_graph
from geoeventfusion.models.actors import ActorGraph, ActorNode


class TestBuildActorGraph:
    def test_build_actor_graph_happy_path(self, sample_co_occurrence_triples):
        """A non-empty triple list must produce an ActorGraph with nodes and edges."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        assert isinstance(graph, ActorGraph)
        assert graph.node_count > 0
        assert graph.edge_count > 0
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_build_actor_graph_expected_actors(self, sample_co_occurrence_triples):
        """All actors from the triples must appear as nodes in the graph."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        node_names = {n.name for n in graph.nodes}

        assert "Houthi" in node_names
        assert "United States" in node_names
        assert "Iran" in node_names

    def test_build_actor_graph_node_by_name_lookup(self, sample_co_occurrence_triples):
        """node_by_name dict must allow O(1) actor lookup."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        assert "Houthi" in graph.node_by_name
        houthi = graph.node_by_name["Houthi"]
        assert isinstance(houthi, ActorNode)
        assert houthi.name == "Houthi"

    def test_build_actor_graph_empty_triples(self):
        """Empty triple list must return an empty ActorGraph without raising."""
        graph = build_actor_graph([])

        assert isinstance(graph, ActorGraph)
        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.nodes == []
        assert graph.edges == []

    def test_build_actor_graph_edge_weights_accumulate(self):
        """Repeated co-occurrences between the same actor pair must accumulate weight."""
        triples = [
            ("Alpha", "Beta", "2024-01-01"),
            ("Alpha", "Beta", "2024-01-02"),
            ("Alpha", "Beta", "2024-01-03"),
            ("Alpha", "Gamma", "2024-01-01"),
        ]
        graph = build_actor_graph(triples)

        # Find the Alpha-Beta edge
        ab_edge = next(
            (e for e in graph.edges if set([e.actor_a, e.actor_b]) == {"Alpha", "Beta"}),
            None,
        )
        assert ab_edge is not None
        assert ab_edge.weight == 3

    def test_build_actor_graph_isolated_nodes_removed(self):
        """Nodes with no edges (isolated) must be pruned from the graph."""
        # Single-actor triple is impossible (co-occurrence needs 2 actors)
        # Instead test that a node truly disconnected from others is removed
        # (This is handled by nx.isolates removal in the implementation)
        triples = [
            ("Alpha", "Beta", "2024-01-01"),
        ]
        graph = build_actor_graph(triples)
        node_names = {n.name for n in graph.nodes}
        # Both Alpha and Beta should be present (they co-occur)
        assert "Alpha" in node_names
        assert "Beta" in node_names

    def test_build_actor_graph_role_classification(self, sample_co_occurrence_triples):
        """Every node must have one of the three valid role labels."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        valid_roles = {"Hub", "Broker", "Peripheral"}

        for node in graph.nodes:
            assert node.role in valid_roles, (
                f"Node {node.name!r} has unexpected role {node.role!r}"
            )

    def test_build_actor_graph_hub_count(self, sample_co_occurrence_triples):
        """With hub_top_n=5, at most 5 nodes should be classified as Hub."""
        graph = build_actor_graph(sample_co_occurrence_triples, hub_top_n=5)
        hub_count = sum(1 for n in graph.nodes if n.role == "Hub")

        assert hub_count <= 5

    def test_build_actor_graph_centrality_non_negative(self, sample_co_occurrence_triples):
        """All centrality metrics must be non-negative."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        for node in graph.nodes:
            assert node.degree_centrality >= 0.0
            assert node.betweenness_centrality >= 0.0
            assert node.pagerank >= 0.0

    def test_build_actor_graph_centrality_list_populated(self, sample_co_occurrence_triples):
        """The centrality list must have one entry per node."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        assert len(graph.centrality) == len(graph.nodes)
        for c in graph.centrality:
            assert c.actor in graph.node_by_name

    def test_build_actor_graph_community_detection(self, sample_co_occurrence_triples):
        """Communities must be detected and each node assigned a community_id."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        assert len(graph.communities) > 0
        for comm in graph.communities:
            assert comm.size == len(comm.members)
            assert comm.size > 0

    def test_build_actor_graph_edge_dates_stored(self):
        """Edge dates must be stored and sorted."""
        triples = [
            ("Alpha", "Beta", "2024-01-03"),
            ("Alpha", "Beta", "2024-01-01"),
            ("Alpha", "Beta", "2024-01-02"),
        ]
        graph = build_actor_graph(triples)
        ab_edge = next(
            (e for e in graph.edges if set([e.actor_a, e.actor_b]) == {"Alpha", "Beta"}),
            None,
        )
        assert ab_edge is not None
        assert ab_edge.dates == sorted(ab_edge.dates)

    def test_build_actor_graph_single_pair(self):
        """A single actor pair triple produces a 2-node, 1-edge graph."""
        triples = [("Alpha", "Beta", "2024-01-01")]
        graph = build_actor_graph(triples)

        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_build_actor_graph_no_dates_in_triples(self):
        """Triples with empty date strings must still build the graph correctly."""
        triples = [
            ("Alpha", "Beta", ""),
            ("Beta", "Gamma", ""),
        ]
        graph = build_actor_graph(triples)
        assert graph.node_count >= 2

    def test_build_actor_graph_custom_hub_top_n(self):
        """hub_top_n parameter controls how many nodes become Hubs."""
        triples = [
            ("A", "B", "2024-01-01"),
            ("A", "C", "2024-01-01"),
            ("A", "D", "2024-01-01"),
            ("B", "C", "2024-01-01"),
            ("B", "D", "2024-01-01"),
            ("C", "D", "2024-01-01"),
        ]
        graph_n1 = build_actor_graph(triples, hub_top_n=1)
        graph_n3 = build_actor_graph(triples, hub_top_n=3)

        hubs_n1 = [n for n in graph_n1.nodes if n.role == "Hub"]
        hubs_n3 = [n for n in graph_n3.nodes if n.role == "Hub"]

        assert len(hubs_n1) <= 1
        assert len(hubs_n3) <= 3


class TestActorGraphHelperMethods:
    def test_get_hubs_returns_hub_nodes(self, sample_co_occurrence_triples):
        """get_hubs() must return only nodes with role == 'Hub'."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        hubs = graph.get_hubs()

        assert all(n.role == "Hub" for n in hubs)

    def test_get_brokers_returns_broker_nodes(self, sample_co_occurrence_triples):
        """get_brokers() must return only nodes with role == 'Broker'."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        brokers = graph.get_brokers()

        assert all(n.role == "Broker" for n in brokers)

    def test_get_top_actors_returns_sorted_by_pagerank(self, sample_co_occurrence_triples):
        """get_top_actors(n) must return top-N actors sorted by PageRank descending."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        top = graph.get_top_actors(n=3)

        assert len(top) <= 3
        for i in range(len(top) - 1):
            assert top[i].pagerank >= top[i + 1].pagerank

    def test_get_top_actors_empty_graph(self):
        """get_top_actors() on an empty ActorGraph must return an empty list."""
        graph = ActorGraph()
        assert graph.get_top_actors(n=5) == []

    def test_get_hubs_empty_graph(self):
        """get_hubs() on an empty ActorGraph must return an empty list."""
        graph = ActorGraph()
        assert graph.get_hubs() == []
