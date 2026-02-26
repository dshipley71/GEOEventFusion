"""GEOEventFusion analysis package.

Pure analytical functions only — no I/O, no API calls, no side effects.
All functions operate on typed models from geoeventfusion.models.
"""

from geoeventfusion.analysis.spike_detector import detect_spikes, rank_spikes
from geoeventfusion.analysis.tone_analyzer import (
    analyze_tone_distribution,
    compute_language_stats,
    compute_country_stats,
)
from geoeventfusion.analysis.actor_graph import build_actor_graph
from geoeventfusion.analysis.query_builder import QueryBuilder

__all__ = [
    "detect_spikes",
    "rank_spikes",
    "analyze_tone_distribution",
    "compute_language_stats",
    "compute_country_stats",
    "build_actor_graph",
    "QueryBuilder",
]
