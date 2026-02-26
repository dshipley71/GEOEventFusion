"""Fusion agent data models for GEOEventFusion.

Defines the output schema for the FusionAgent — event clusters, contradiction flags,
and fusion statistics produced after linking events across all source pools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContradictionFlag:
    """A detected contradiction between two events within or across fusion clusters."""

    cluster_id: str
    event_a_summary: str
    event_b_summary: str
    dimension: str        # Which dimension showed contradiction (e.g., "actor", "event_type")
    severity: str = "WARNING"   # "WARNING" or "ERROR"
    detail: str = ""


@dataclass
class FusionCluster:
    """A cluster of events from multiple sources that describe the same real-world incident."""

    cluster_id: str
    events: List[Any] = field(default_factory=list)   # List of extracted event dicts
    source_types: List[str] = field(default_factory=list)   # ["gdelt", "rss", "acled"]
    fusion_confidence: float = 0.0
    temporal_span: Dict[str, str] = field(default_factory=dict)   # {"start": ..., "end": ...}
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    primary_actors: List[str] = field(default_factory=list)
    contradiction_flags: List[ContradictionFlag] = field(default_factory=list)
    corroboration_count: int = 0
    event_type: str = ""
    country: str = ""


@dataclass
class FusionStats:
    """Summary statistics for the fusion run."""

    total_events_in: int = 0
    total_clusters: int = 0
    mean_cluster_size: float = 0.0
    contradiction_rate: float = 0.0
    corroboration_rate: float = 0.0


@dataclass
class FusionAgentResult:
    """Complete output from the FusionAgent."""

    clusters: List[FusionCluster] = field(default_factory=list)
    unclustered_events: List[Any] = field(default_factory=list)
    fusion_stats: Optional[FusionStats] = None
    warnings: List[str] = field(default_factory=list)
