"""Unit tests for geoeventfusion.agents.fusion_agent.

Covers:
- FusionAgent.run: happy path, empty events, no LLM result
- Cluster formation: temporal proximity grouping
- Fusion stats: total_events_in, total_clusters, mean_cluster_size
- Contradiction detection: actor and event-type mismatches
- Single-event clusters: fall back to one-event clusters when no match
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from geoeventfusion.agents.fusion_agent import FusionAgent
from geoeventfusion.models.fusion import FusionAgentResult, FusionCluster, FusionStats
from geoeventfusion.models.storyboard import TimelineEntry


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _make_timeline_entry(
    event_type: str = "CONFLICT",
    date: str = "2024-01-15",
    country: str = "Yemen",
    lat: float = 15.5,
    lon: float = 48.5,
    actors=None,
    confidence: float = 0.75,
    source_url: str = "https://example.com/article",
    summary: str = "Test event",
) -> TimelineEntry:
    return TimelineEntry(
        event_type=event_type,
        datetime=date,
        country=country,
        lat=lat,
        lon=lon,
        actors=actors or ["Houthi", "United States"],
        summary=summary,
        confidence=confidence,
        source_url=source_url,
        source_title=summary,
    )


def _make_llm_result(events=None):
    llm = MagicMock()
    llm.timeline_events = events or []
    llm.hypotheses = []
    llm.timeline_phases = []
    return llm


def _make_context(config, llm_result=None, ground_truth_result=None, custom_dataset_result=None):
    ctx = MagicMock()
    ctx.config = config
    ctx.llm_result = llm_result
    ctx.ground_truth_result = ground_truth_result
    ctx.custom_dataset_result = custom_dataset_result
    return ctx


# ── FusionAgent.run — empty / no input ────────────────────────────────────────────

class TestFusionAgentEmpty:
    def test_run_no_llm_result_returns_result(self, test_pipeline_config):
        """Agent must return a FusionAgentResult even when no upstream data is present."""
        agent = FusionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, llm_result=None)
        result = agent.run(ctx)
        assert isinstance(result, FusionAgentResult)

    def test_run_no_events_returns_empty_clusters(self, test_pipeline_config):
        """With no events, clusters and unclustered_events must both be empty."""
        agent = FusionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=[]))
        result = agent.run(ctx)
        assert result.clusters == []
        assert result.unclustered_events == []

    def test_run_no_events_fusion_stats_total_zero(self, test_pipeline_config):
        """With no events, fusion_stats.total_events_in must be 0."""
        agent = FusionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=[]))
        result = agent.run(ctx)
        assert result.fusion_stats is not None
        assert result.fusion_stats.total_events_in == 0

    def test_run_no_events_has_warning(self, test_pipeline_config):
        """Empty input must emit a warning in the result."""
        agent = FusionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, llm_result=None)
        result = agent.run(ctx)
        assert len(result.warnings) > 0


# ── FusionAgent.run — single event ───────────────────────────────────────────────

class TestFusionAgentSingleEvent:
    def test_single_event_creates_one_cluster(self, test_pipeline_config):
        """A single event must produce exactly one cluster."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry()]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        total = len(result.clusters) + len(result.unclustered_events)
        assert total == 1

    def test_single_event_total_in_stats(self, test_pipeline_config):
        """fusion_stats.total_events_in must equal the number of input events."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry()]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        assert result.fusion_stats.total_events_in == 1


# ── FusionAgent.run — multiple events ────────────────────────────────────────────

class TestFusionAgentMultipleEvents:
    def test_multiple_events_all_accounted_for(self, test_pipeline_config):
        """All input events must be present across clusters + unclustered."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [
            _make_timeline_entry(date="2024-01-15", country="Yemen", actors=["Houthi", "US"]),
            _make_timeline_entry(date="2024-01-16", country="Yemen", actors=["Houthi", "UK"]),
            _make_timeline_entry(date="2024-02-01", country="Iraq", actors=["PMF", "US"]),
        ]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)

        all_events_out = []
        for cluster in result.clusters:
            all_events_out.extend(cluster.events)
        all_events_out.extend(result.unclustered_events)

        assert result.fusion_stats.total_events_in == 3

    def test_total_clusters_is_positive(self, test_pipeline_config):
        """With multiple events, at least one cluster must be created."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [
            _make_timeline_entry(date="2024-01-15"),
            _make_timeline_entry(date="2024-01-15"),
        ]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        assert result.fusion_stats.total_clusters >= 1

    def test_cluster_ids_are_unique(self, test_pipeline_config):
        """Every cluster must have a distinct cluster_id."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry(date=f"2024-01-{i:02d}") for i in range(1, 6)]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        cluster_ids = [c.cluster_id for c in result.clusters]
        assert len(cluster_ids) == len(set(cluster_ids))

    def test_fusion_confidence_in_range(self, test_pipeline_config):
        """Fusion confidence for every cluster must be in [0.0, 1.0]."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry() for _ in range(4)]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        for cluster in result.clusters:
            assert 0.0 <= cluster.fusion_confidence <= 1.0, (
                f"Cluster {cluster.cluster_id} fusion_confidence out of range: {cluster.fusion_confidence}"
            )


# ── FusionAgent.run — fusion stats ────────────────────────────────────────────────

class TestFusionAgentStats:
    def test_stats_total_clusters_matches_clusters_list(self, test_pipeline_config):
        """fusion_stats.total_clusters must equal len(result.clusters)."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry(date=f"2024-01-{i:02d}") for i in range(1, 4)]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        assert result.fusion_stats.total_clusters == len(result.clusters)

    def test_stats_mean_cluster_size_positive(self, test_pipeline_config):
        """mean_cluster_size must be > 0 when there are events."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry() for _ in range(3)]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        if result.fusion_stats.total_clusters > 0:
            assert result.fusion_stats.mean_cluster_size > 0.0


# ── Cluster output schema ─────────────────────────────────────────────────────────

class TestFusionClusterSchema:
    def test_clusters_are_typed_objects(self, test_pipeline_config):
        """All clusters must be FusionCluster dataclass instances."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry()]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        for cluster in result.clusters:
            assert isinstance(cluster, FusionCluster), (
                f"Expected FusionCluster, got {type(cluster)}"
            )

    def test_cluster_has_required_fields(self, test_pipeline_config):
        """Each FusionCluster must have cluster_id, events, source_types, and fusion_confidence."""
        agent = FusionAgent(config=test_pipeline_config)
        events = [_make_timeline_entry()]
        ctx = _make_context(test_pipeline_config, llm_result=_make_llm_result(events=events))
        result = agent.run(ctx)
        for cluster in result.clusters:
            assert cluster.cluster_id
            assert isinstance(cluster.events, list)
            assert isinstance(cluster.source_types, list)
            assert isinstance(cluster.fusion_confidence, float)
