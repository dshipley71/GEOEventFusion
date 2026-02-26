"""Integration tests for GEOEventFusion pipeline phases.

These tests exercise multiple modules working together end-to-end,
using fixture data rather than real API calls. They verify that:

- Phase 1: Fixture data flows correctly through spike detection
- Phase 2: Actor graph construction from co-occurrence triples
- Phase 3: Tone analysis chain (tonechart → ToneStats → trend)
- Phase 4: PipelineConfig and PipelineContext creation and defaults
- Phase 5: JSON persistence round-trip with analysis outputs
- Phase 6: Query builder integrating with spike metadata

No real HTTP calls, LLM calls, or file-system side effects leak between tests
(all file writes go to pytest's tmp_path fixture).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import PipelineConfig
from geoeventfusion.models.pipeline import PipelineContext, PhaseRecord
from geoeventfusion.analysis.spike_detector import detect_spikes, find_phase_boundaries
from geoeventfusion.analysis.actor_graph import build_actor_graph
from geoeventfusion.analysis.tone_analyzer import (
    analyze_tone_distribution,
    compute_language_stats,
    compute_tone_trend,
)
from geoeventfusion.analysis.query_builder import QueryBuilder
from geoeventfusion.analysis.visual_intel import compute_novelty_score, parse_image_collage_response
from geoeventfusion.io.persistence import ensure_output_dir, load_json, save_json
from geoeventfusion.models.events import (
    Article,
    GDELTAgentResult,
    SpikeWindow,
    TimelineStep,
    ToneChartBin,
)


# ── Phase 1: Spike detection pipeline ───────────────────────────────────────────

class TestPhase1SpikeDetection:
    """Simulate Phase 1: fixture timeline → spike detection → phase boundaries."""

    def test_spike_detection_produces_correct_count(self, sample_timeline_steps):
        """Fixture timeline must produce exactly 2 spikes."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5, query="Houthi Red Sea")
        assert len(spikes) == 2

    def test_spike_detection_highest_z_first(self, sample_timeline_steps):
        """Top-ranked spike must have the highest Z-score."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        assert spikes[0].z_score > spikes[1].z_score
        assert spikes[0].rank == 1

    def test_phase_boundaries_include_spike_dates(self, sample_timeline_steps):
        """Phase boundaries derived from spikes must include the top spike dates."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        boundaries = find_phase_boundaries(spikes)

        assert len(boundaries) >= 2
        assert "2024-01-31" in boundaries
        assert "2024-03-16" in boundaries

    def test_spike_metadata_preserved_end_to_end(self, sample_timeline_steps):
        """Spike query metadata must survive from detect_spikes to phase boundaries."""
        query = "Houthi Red Sea attacks"
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5, query=query)

        for spike in spikes:
            assert spike.query == query

    def test_spike_windows_stored_in_gdelt_result(self, sample_timeline_steps):
        """GDELTAgentResult.spikes must be populated from detect_spikes output."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        result = GDELTAgentResult(
            timeline_volinfo=sample_timeline_steps,
            spikes=spikes,
        )

        assert len(result.spikes) == 2
        assert result.spikes[0].rank == 1


# ── Phase 2: Actor graph pipeline ───────────────────────────────────────────────

class TestPhase2ActorGraph:
    """Simulate Phase 2: co-occurrence triples → actor graph → centrality."""

    def test_actor_graph_built_from_fixture_triples(self, sample_co_occurrence_triples):
        """build_actor_graph must succeed with fixture triples."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        assert graph.node_count > 0
        assert graph.edge_count > 0

    def test_hubs_present_in_high_connectivity_graph(self, sample_co_occurrence_triples):
        """With ≥5 nodes and dense connectivity, Hub nodes must be assigned."""
        graph = build_actor_graph(sample_co_occurrence_triples, hub_top_n=3)
        hubs = graph.get_hubs()

        assert len(hubs) > 0

    def test_houthi_is_most_central_actor(self, sample_co_occurrence_triples):
        """Houthi appears in the most edges in the fixture and should rank highly."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        top_actors = graph.get_top_actors(n=3)
        top_names = [a.name for a in top_actors]

        # Houthi co-occurs with US, Yemen, UK, EU, Iran — highest connectivity
        assert "Houthi" in top_names

    def test_community_detection_runs_without_error(self, sample_co_occurrence_triples):
        """Community detection must complete and assign community_id to all nodes."""
        graph = build_actor_graph(sample_co_occurrence_triples)

        assert len(graph.communities) > 0
        for node in graph.nodes:
            assert node.community_id is not None

    def test_actor_graph_stored_in_gdelt_result(self, sample_co_occurrence_triples):
        """GDELTAgentResult.actor_graph can store the built graph."""
        graph = build_actor_graph(sample_co_occurrence_triples)
        result = GDELTAgentResult(actor_graph=graph)

        assert result.actor_graph is not None
        assert result.actor_graph.node_count > 0


# ── Phase 3: Tone analysis pipeline ─────────────────────────────────────────────

class TestPhase3ToneAnalysis:
    """Simulate Phase 3: tonechart + timeline_tone → ToneStats + trend."""

    def test_tone_stats_from_fixture_tonechart(self, sample_tonechart_bins):
        """Fixture tonechart must produce valid ToneStats."""
        stats = analyze_tone_distribution(sample_tonechart_bins)

        assert stats.total_articles > 0
        assert stats.modal_tone < 0  # fixture is negatively skewed
        assert stats.mean_tone < 0
        assert stats.polarity_ratio > 0.5

    def test_tone_trend_computed_from_timeline(self):
        """A deteriorating timeline must produce deteriorating trend."""
        steps = [
            TimelineStep(date=f"2024-01-{i:02d}", value=-2.0 - i * 0.3)
            for i in range(1, 10)
        ]
        trend = compute_tone_trend(steps)

        assert trend["trend_direction"] == "deteriorating"
        assert trend["trend_delta"] < 0

    def test_tone_stats_stored_in_gdelt_result(self, sample_tonechart_bins):
        """GDELTAgentResult.tone_stats can store ToneStats."""
        stats = analyze_tone_distribution(sample_tonechart_bins)
        result = GDELTAgentResult(
            tonechart=sample_tonechart_bins,
            tone_stats=stats,
        )

        assert result.tone_stats is not None
        assert result.tone_stats.total_articles > 0

    def test_language_stats_from_language_steps(self):
        """Language stats must be derived from timeline_lang steps."""
        steps = [
            TimelineStep(date="2024-01-01", value=60.0, label="English"),
            TimelineStep(date="2024-01-04", value=30.0, label="Arabic"),
            TimelineStep(date="2024-01-07", value=10.0, label="Russian"),
        ]
        lang_stats = compute_language_stats(steps)

        assert len(lang_stats.top_languages) == 3
        assert lang_stats.top_languages[0]["language"] == "English"
        assert lang_stats.diversity_index > 0.0

    def test_full_tone_pipeline_chain(self, sample_tonechart_bins):
        """Full tone pipeline: tonechart → stats → polarity check → result storage."""
        stats = analyze_tone_distribution(sample_tonechart_bins)
        assert stats.polarity_ratio > 0.5

        result = GDELTAgentResult(
            tonechart=sample_tonechart_bins,
            tone_stats=stats,
        )
        # GDELTAgentResult.tone_stats must reflect the computed stats
        assert result.tone_stats.mean_tone < 0


# ── Phase 4: Config and Context ──────────────────────────────────────────────────

class TestPhase4ConfigContext:
    """Test PipelineConfig and PipelineContext creation and defaults."""

    def test_pipeline_config_defaults(self):
        """Default PipelineConfig must use values from config.defaults."""
        from config.defaults import SPIKE_Z_THRESHOLD, MAX_CONFIDENCE, MAX_RECORDS

        config = PipelineConfig(query="test")

        assert config.spike_z_threshold == SPIKE_Z_THRESHOLD
        assert config.max_confidence == MAX_CONFIDENCE
        assert config.max_records == MAX_RECORDS

    def test_pipeline_config_max_confidence_clamped(self):
        """max_confidence above the hard cap must be clamped to 0.82."""
        config = PipelineConfig(query="test", max_confidence=1.0)
        assert config.max_confidence == 0.82

    def test_pipeline_config_test_mode(self, test_pipeline_config):
        """test_pipeline_config fixture must set test_mode=True."""
        assert test_pipeline_config.test_mode is True

    def test_pipeline_context_creation(self, test_pipeline_config, tmp_path):
        """PipelineContext must be created with all agent result fields as None."""
        ctx = PipelineContext(
            config=test_pipeline_config,
            run_id="20240115_120000_test",
            output_dir=tmp_path,
        )

        assert ctx.gdelt_result is None
        assert ctx.rss_result is None
        assert ctx.llm_result is None
        assert ctx.fusion_result is None
        assert ctx.storyboard_result is None
        assert ctx.validation_result is None
        assert ctx.export_result is None

    def test_pipeline_context_add_warning(self, test_pipeline_context):
        """add_warning must append to the warnings list."""
        test_pipeline_context.add_warning("Test warning")
        assert "Test warning" in test_pipeline_context.warnings

    def test_pipeline_context_add_error(self, test_pipeline_context):
        """add_error must append to the errors list."""
        test_pipeline_context.add_error("Test error")
        assert "Test error" in test_pipeline_context.errors

    def test_pipeline_context_log_phase_start(self, test_pipeline_context):
        """log_phase_start must create a PhaseRecord and add it to phase_log."""
        record = test_pipeline_context.log_phase_start("GDELT_FETCH")

        assert isinstance(record, PhaseRecord)
        assert record.phase_name == "GDELT_FETCH"
        assert record.start_time is not None
        assert record in test_pipeline_context.phase_log

    def test_pipeline_context_log_phase_end(self, test_pipeline_context):
        """log_phase_end must set end_time and status on the record."""
        record = test_pipeline_context.log_phase_start("RSS_FETCH")
        test_pipeline_context.log_phase_end(record, status="PARTIAL")

        assert record.end_time is not None
        assert record.status == "PARTIAL"
        assert record.elapsed_seconds >= 0.0

    def test_fusion_weights_must_sum_to_one(self):
        """FusionWeights default values must sum to 1.0."""
        from config.settings import FusionWeights
        weights = FusionWeights()
        total = (
            weights.temporal + weights.geographic + weights.actor
            + weights.semantic + weights.event_type
        )
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_fusion_weights_invalid_sum_raises(self):
        """FusionWeights that do not sum to 1.0 must raise ValueError."""
        from config.settings import FusionWeights
        with pytest.raises(ValueError, match="must sum to 1.0"):
            FusionWeights(temporal=0.5, geographic=0.5, actor=0.5, semantic=0.1, event_type=0.1)


# ── Phase 5: Persistence pipeline ───────────────────────────────────────────────

class TestPhase5Persistence:
    """Test end-to-end persistence: spike results → JSON → reload."""

    def test_spike_results_saved_and_loaded(self, sample_timeline_steps, tmp_path):
        """Spike detection output must survive a full save_json/load_json round-trip."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5, query="Houthi")
        spike_data = [
            {"date": s.date, "z_score": s.z_score, "volume": s.volume, "rank": s.rank}
            for s in spikes
        ]

        output_path = tmp_path / "spikes.json"
        save_json(spike_data, output_path)
        loaded = load_json(output_path)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["rank"] == 1

    def test_tone_stats_saved_and_loaded(self, sample_tonechart_bins, tmp_path):
        """ToneStats must be serializable to JSON and reload correctly."""
        import dataclasses

        stats = analyze_tone_distribution(sample_tonechart_bins)
        stats_dict = dataclasses.asdict(stats)

        output_path = tmp_path / "tone_stats.json"
        save_json(stats_dict, output_path)
        loaded = load_json(output_path)

        assert loaded is not None
        assert loaded["total_articles"] == stats.total_articles
        assert abs(loaded["mean_tone"] - stats.mean_tone) < 1e-4

    def test_ensure_output_dir_with_run_id(self, tmp_path):
        """ensure_output_dir must create structured run directory."""
        run_id = "20240115_120000_houthi_red_sea"
        run_dir = ensure_output_dir(tmp_path, run_id)

        assert (run_dir / "charts").is_dir()

    def test_multiple_artifacts_in_run_dir(self, sample_timeline_steps, sample_tonechart_bins, tmp_path):
        """Multiple artifact JSON files can be written to the same run directory."""
        import dataclasses

        run_dir = ensure_output_dir(tmp_path, "test_run")

        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        spike_data = [dataclasses.asdict(s) for s in spikes]
        save_json(spike_data, run_dir / "spikes.json")

        stats = analyze_tone_distribution(sample_tonechart_bins)
        save_json(dataclasses.asdict(stats), run_dir / "tone_stats.json")

        assert (run_dir / "spikes.json").exists()
        assert (run_dir / "tone_stats.json").exists()

        loaded_spikes = load_json(run_dir / "spikes.json")
        assert len(loaded_spikes) == 2


# ── Phase 6: Query builder integration ──────────────────────────────────────────

class TestPhase6QueryBuilder:
    """Test QueryBuilder producing queries used in real GDELT fetches."""

    def test_query_builder_produces_valid_gdelt_query(self):
        """The full query composition must produce a non-empty, well-structured string."""
        qb = QueryBuilder(near_min_term_length=5, near_window=15, repeat_threshold=3)
        query = qb.build_base_query(
            "Houthi Red Sea",
            aliases=["Ansar Allah", "huthis"],
            gkg_themes=["MARITIME_SECURITY"],
            add_repeat=True,
        )

        assert len(query) > 0
        assert '"Houthi Red Sea"' in query
        assert "Ansar Allah" in query
        assert "theme:MARITIME_SECURITY" in query

    def test_spike_date_used_in_query_metadata(self, sample_timeline_steps):
        """Spike dates from detection can be used to scope follow-up GDELT queries."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        top_spike = spikes[0]

        # Spike date is available for date-scoped fetch
        assert top_spike.date == "2024-01-31"
        assert top_spike.z_score > 3.0

    def test_imagetag_query_for_spike_context(self):
        """Visual intelligence query must be built correctly for spike enrichment."""
        qb = QueryBuilder()
        imagetag_query = qb.build_imagetag_query(["military", "weapon", "explosion"])

        assert 'imagetag:"military"' in imagetag_query
        assert 'imagetag:"explosion"' in imagetag_query
        assert "OR" in imagetag_query

    def test_country_scoped_query_for_bilateral_analysis(self):
        """Country-scoped queries must be correctly formatted for bilateral analysis."""
        qb = QueryBuilder()
        base = '"Houthi Red Sea"'

        ir_query = qb.build_source_country_query(base, "IR")
        us_query = qb.build_source_country_query(base, "US")

        assert "sourcecountry:ir" in ir_query
        assert "sourcecountry:us" in us_query
        assert base in ir_query
        assert base in us_query

    def test_authoritative_domain_query_for_validation(self):
        """Authority domain query must be buildable for validation enrichment."""
        qb = QueryBuilder()
        domain_query = qb.build_authoritative_domain_query(
            '"Houthi Red Sea"',
            ["un.org", "state.gov", "nato.int"]
        )

        assert "domainis:un.org" in domain_query
        assert "domainis:state.gov" in domain_query
        assert "domainis:nato.int" in domain_query


# ── Article pool integration ──────────────────────────────────────────────────────

class TestArticlePoolIntegration:
    """Test that Article objects from fixtures behave correctly in aggregated pools."""

    def test_all_articles_deduplication(self, sample_articles):
        """GDELTAgentResult.all_articles() must deduplicate by URL."""
        # Add the same articles to two different pools
        result = GDELTAgentResult(
            articles_recent=sample_articles,
            articles_negative=sample_articles[:3],  # overlap with recent
        )

        all_articles = result.all_articles()
        urls = [a.url for a in all_articles]

        # Must not have duplicate URLs
        assert len(urls) == len(set(urls))

    def test_all_articles_union_of_all_pools(self, sample_articles):
        """all_articles() must return the full union across all 9 pools."""
        extra_articles = [
            Article(
                url="https://extra.com/article1",
                title="Extra article",
                published_at="2024-01-20",
                source="extra.com",
            )
        ]
        result = GDELTAgentResult(
            articles_recent=sample_articles,
            articles_authoritative=extra_articles,
        )

        all_articles = result.all_articles()
        all_urls = {a.url for a in all_articles}

        assert "https://extra.com/article1" in all_urls
        assert all(a.url in all_urls for a in sample_articles)

    def test_article_hash_equality_by_url(self, sample_articles):
        """Articles with the same URL must be equal (hash by URL)."""
        a1 = sample_articles[0]
        a2 = Article(
            url=a1.url,
            title="Different title",
            published_at="2024-01-20",
            source="different.com",
        )

        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_articles_in_set_deduplicated(self, sample_articles):
        """Articles added to a set must be deduplicated by URL hash."""
        article_set = set(sample_articles)
        # Add duplicates
        for a in sample_articles:
            article_set.add(a)

        assert len(article_set) == len(sample_articles)
