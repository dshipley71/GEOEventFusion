"""Actor graph data models for GEOEventFusion.

Defines typed structures for actor nodes, edges, centrality metrics,
and the overall actor co-occurrence graph produced by the GDELTAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ActorNode:
    """A single actor extracted from article titles in the GDELT article pool."""

    name: str
    role: str = "Peripheral"   # "Hub", "Broker", or "Peripheral"
    mention_count: int = 0
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    pagerank: float = 0.0
    community_id: Optional[int] = None


@dataclass
class ActorEdge:
    """A co-occurrence relationship between two actors."""

    actor_a: str
    actor_b: str
    weight: int = 1       # Number of co-occurrence instances
    dates: List[str] = field(default_factory=list)   # ISO dates when co-occurrence observed


@dataclass
class CentralityResult:
    """Centrality metrics computed for a single actor."""

    actor: str
    degree: float
    betweenness: float
    pagerank: float
    role: str   # "Hub", "Broker", or "Peripheral"


@dataclass
class CommunityResult:
    """A community detected via greedy modularity maximization."""

    community_id: int
    members: List[str] = field(default_factory=list)
    size: int = 0


@dataclass
class TemporalCommunityShift:
    """Measures how much community structure changes across the analysis window."""

    early_communities: List[CommunityResult] = field(default_factory=list)
    mid_communities: List[CommunityResult] = field(default_factory=list)
    late_communities: List[CommunityResult] = field(default_factory=list)
    # Jaccard reorganization score: 0 = no change, 1 = complete restructuring
    reorganization_score: float = 0.0
    # Candidate phase boundary date derived from high-reorganization window
    phase_boundary_candidate: Optional[str] = None


@dataclass
class ActorGraph:
    """Complete actor co-occurrence graph with centrality metrics and community structure."""

    nodes: List[ActorNode] = field(default_factory=list)
    edges: List[ActorEdge] = field(default_factory=list)
    centrality: List[CentralityResult] = field(default_factory=list)
    communities: List[CommunityResult] = field(default_factory=list)
    temporal_shift: Optional[TemporalCommunityShift] = None
    node_count: int = 0
    edge_count: int = 0

    # Actor lookup maps for fast access
    node_by_name: Dict[str, ActorNode] = field(default_factory=dict)

    def get_hubs(self) -> List[ActorNode]:
        """Return all actors classified as Hub."""
        return [n for n in self.nodes if n.role == "Hub"]

    def get_brokers(self) -> List[ActorNode]:
        """Return all actors classified as Broker."""
        return [n for n in self.nodes if n.role == "Broker"]

    def get_top_actors(self, n: int = 10) -> List[ActorNode]:
        """Return top N actors sorted by PageRank descending."""
        return sorted(self.nodes, key=lambda x: x.pagerank, reverse=True)[:n]
