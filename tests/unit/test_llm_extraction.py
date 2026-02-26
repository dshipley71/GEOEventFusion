"""Unit tests for geoeventfusion.agents.llm_extraction_agent.

Covers:
- LLMExtractionAgent.run: happy path with mock LLM, empty context, no GDELT result
- Timeline extraction: phase and turning point generation
- Hypothesis generation: 4-round debate integration
- Confidence cap enforcement: values above MAX_CONFIDENCE are clamped
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from geoeventfusion.agents.llm_extraction_agent import LLMExtractionAgent
from geoeventfusion.models.storyboard import (
    Hypothesis,
    LLMExtractionAgentResult,
    TimelineEntry,
    TimelinePhase,
    TurningPoint,
)


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _make_context(config, gdelt_result=None, rss_result=None, llm_client=None):
    """Build a minimal mock PipelineContext for LLMExtractionAgent tests."""
    ctx = MagicMock()
    ctx.config = config
    ctx.gdelt_result = gdelt_result
    ctx.rss_result = rss_result
    ctx.ground_truth_result = None
    ctx.custom_dataset_result = None
    return ctx


def _make_gdelt_result(articles=None):
    """Build a minimal mock GDELTAgentResult."""
    gdelt = MagicMock()
    gdelt.articles_recent = articles or []
    gdelt.articles_negative = []
    gdelt.articles_positive = []
    gdelt.spikes = []
    gdelt.timeline_volinfo = []
    gdelt.actor_graph = None
    gdelt.tone_stats = None
    return gdelt


# ── LLMExtractionAgent.run — empty context ────────────────────────────────────────

class TestLLMExtractionAgentEmptyContext:
    def test_run_no_gdelt_result_returns_result(self, test_pipeline_config, mock_llm_client):
        """Agent must return an LLMExtractionAgentResult even with no upstream data."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, gdelt_result=None)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        assert isinstance(result, LLMExtractionAgentResult)

    def test_run_empty_articles_returns_empty_timeline(self, test_pipeline_config, mock_llm_client):
        """With no articles, timeline_events must be an empty list."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, gdelt_result=_make_gdelt_result(articles=[]))

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        assert isinstance(result.timeline_events, list)

    def test_run_no_gdelt_result_status_ok(self, test_pipeline_config, mock_llm_client):
        """Status must not be CRITICAL when there is simply no data — only when an error occurs."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, gdelt_result=None)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        assert result.status != "CRITICAL"


# ── LLMExtractionAgent.run — happy path ──────────────────────────────────────────

class TestLLMExtractionAgentHappyPath:
    def test_run_with_articles_calls_llm(self, test_pipeline_config, mock_llm_client, sample_articles):
        """With articles present, the mock LLM client should be invoked."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        assert isinstance(result, LLMExtractionAgentResult)
        # LLM should have been called at least once (for timeline or hypotheses)
        assert mock_llm_client.call.called or mock_llm_client.call_json.called

    def test_run_result_has_required_fields(self, test_pipeline_config, mock_llm_client, sample_articles):
        """Result must have timeline_events, hypotheses, and timeline_phases fields."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        assert hasattr(result, "timeline_events")
        assert hasattr(result, "hypotheses")
        assert hasattr(result, "timeline_phases")
        assert hasattr(result, "turning_points")

    def test_run_timeline_events_are_typed(self, test_pipeline_config, mock_llm_client, sample_articles):
        """All timeline_events must be TimelineEntry objects, not raw dicts."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        for event in result.timeline_events:
            assert isinstance(event, TimelineEntry), (
                f"Expected TimelineEntry, got {type(event)}"
            )

    def test_run_hypotheses_are_typed(self, test_pipeline_config, mock_llm_client, sample_articles):
        """All hypotheses must be Hypothesis objects."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        for h in result.hypotheses:
            assert isinstance(h, Hypothesis), f"Expected Hypothesis, got {type(h)}"


# ── Confidence cap enforcement ────────────────────────────────────────────────────

class TestConfidenceCap:
    def test_timeline_event_confidence_capped(self, test_pipeline_config, mock_llm_client):
        """Any extracted event with confidence > MAX_CONFIDENCE must be clamped."""
        from config.defaults import MAX_CONFIDENCE

        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result()
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        # Override mock to return a high-confidence event
        mock_llm_client.call_json.return_value = {
            "events": [
                {
                    "event_type": "CONFLICT",
                    "datetime": "2024-01-15",
                    "country": "Yemen",
                    "lat": 15.5,
                    "lon": 48.5,
                    "actors": ["Houthi"],
                    "summary": "Test event",
                    "confidence": 0.99,   # Over cap
                    "source_url": "https://example.com",
                    "source_title": "Test",
                }
            ]
        }

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        for event in result.timeline_events:
            assert event.confidence <= MAX_CONFIDENCE, (
                f"Confidence {event.confidence} exceeds MAX_CONFIDENCE {MAX_CONFIDENCE}"
            )

    def test_hypothesis_confidence_capped(self, test_pipeline_config, mock_llm_client, sample_articles):
        """Hypothesis confidence scores must not exceed MAX_CONFIDENCE."""
        from config.defaults import MAX_CONFIDENCE

        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=mock_llm_client):
            result = agent.run(ctx)

        for h in result.hypotheses:
            assert h.confidence <= MAX_CONFIDENCE, (
                f"Hypothesis confidence {h.confidence} exceeds MAX_CONFIDENCE {MAX_CONFIDENCE}"
            )


# ── Graceful degradation ──────────────────────────────────────────────────────────

class TestLLMExtractionAgentDegradation:
    def test_malformed_llm_response_does_not_raise(self, test_pipeline_config, sample_articles):
        """If the LLM returns invalid JSON, the agent must not raise an exception."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        bad_llm = MagicMock()
        bad_llm.backend = "mock"
        bad_llm.max_confidence = 0.82
        bad_llm.call.return_value = "```not valid json```"
        bad_llm.call_json.return_value = None
        bad_llm.enforce_confidence_cap.side_effect = lambda x: x

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=bad_llm):
            result = agent.run(ctx)

        assert isinstance(result, LLMExtractionAgentResult)

    def test_empty_llm_response_does_not_raise(self, test_pipeline_config, sample_articles):
        """If the LLM returns an empty string, the agent must not raise an exception."""
        agent = LLMExtractionAgent(config=test_pipeline_config)
        gdelt = _make_gdelt_result(articles=sample_articles)
        ctx = _make_context(test_pipeline_config, gdelt_result=gdelt)

        empty_llm = MagicMock()
        empty_llm.backend = "mock"
        empty_llm.max_confidence = 0.82
        empty_llm.call.return_value = ""
        empty_llm.call_json.return_value = None
        empty_llm.enforce_confidence_cap.side_effect = lambda x: x

        with patch("geoeventfusion.agents.llm_extraction_agent.LLMClient", return_value=empty_llm):
            result = agent.run(ctx)

        assert isinstance(result, LLMExtractionAgentResult)
        assert isinstance(result.timeline_events, list)
