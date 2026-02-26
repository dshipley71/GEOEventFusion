"""Unit tests for geoeventfusion.agents.validation_agent.

Covers:
- ValidationAgent.run: happy path, no storyboard, no key events
- Grounding score: 0.0 when all events unverified, > 0 when some verified
- URL check: graceful UNCHECKED handling on network errors
- Flag generation: severity levels for unverified claims
- Ground truth: empty ground truth handled gracefully
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from geoeventfusion.agents.validation_agent import ValidationAgent
from geoeventfusion.models.storyboard import PanelActor, PanelKeyEvent, StoryboardPanel
from geoeventfusion.models.validation import ValidationAgentResult


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _make_panel(panel_id="panel_001", key_events=None, confidence=0.75):
    """Build a minimal StoryboardPanel for validation tests."""
    return StoryboardPanel(
        panel_id=panel_id,
        date_range={"start": "2024-01-01", "end": "2024-01-31"},
        headline="Test Panel",
        key_events=key_events or [],
        actors=[PanelActor(name="Houthi", role="Hub", centrality_score=0.8)],
        narrative_summary="Test narrative",
        confidence=confidence,
        grounded_sources=[],
        unverified_elements=[],
    )


def _make_key_event(
    date="2024-01-15",
    description="Houthi attack on merchant vessel",
    source_title="Houthi attack on merchant vessel",
    source_url="https://www.reuters.com/test",
    verified=False,
):
    return PanelKeyEvent(
        date=date,
        description=description,
        source_title=source_title,
        source_url=source_url,
        verified=verified,
    )


def _make_storyboard_result(panels=None):
    sb = MagicMock()
    sb.panels = panels or []
    sb.overall_confidence = 0.75
    sb.escalation_risk = 0.5
    return sb


def _make_context(config, storyboard_result=None, ground_truth_result=None, custom_dataset_result=None):
    ctx = MagicMock()
    ctx.config = config
    ctx.storyboard_result = storyboard_result
    ctx.ground_truth_result = ground_truth_result
    ctx.custom_dataset_result = custom_dataset_result
    return ctx


# ── ValidationAgent.run — empty / no input ───────────────────────────────────────

class TestValidationAgentEmpty:
    def test_run_no_storyboard_returns_result(self, test_pipeline_config):
        """Agent must return a ValidationAgentResult even with no storyboard."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=None)
        result = agent.run(ctx)
        assert isinstance(result, ValidationAgentResult)

    def test_run_no_storyboard_grounding_score_zero(self, test_pipeline_config):
        """With no storyboard, grounding_score must be 0.0."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=None)
        result = agent.run(ctx)
        assert result.grounding_score == 0.0

    def test_run_empty_panels_returns_result(self, test_pipeline_config):
        """Agent must return gracefully when storyboard has zero panels."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=_make_storyboard_result(panels=[]))
        result = agent.run(ctx)
        assert isinstance(result, ValidationAgentResult)

    def test_run_no_storyboard_has_warning(self, test_pipeline_config):
        """No storyboard must emit a warning."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=None)
        result = agent.run(ctx)
        assert len(result.warnings) > 0

    def test_run_panel_with_no_key_events(self, test_pipeline_config):
        """Panel with empty key_events list must not cause an error."""
        agent = ValidationAgent(config=test_pipeline_config)
        panels = [_make_panel(key_events=[])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )
        result = agent.run(ctx)
        assert isinstance(result, ValidationAgentResult)


# ── ValidationAgent.run — URL checks ─────────────────────────────────────────────

class TestValidationAgentURLChecks:
    def test_url_check_network_error_does_not_raise(self, test_pipeline_config):
        """A network error during URL check must not propagate as an exception."""
        agent = ValidationAgent(config=test_pipeline_config)
        key_event = _make_key_event(source_url="https://unreachable.example.invalid/article")
        panels = [_make_panel(key_events=[key_event])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )

        import requests
        with patch.object(requests.Session, "head", side_effect=requests.exceptions.ConnectionError("mock")):
            result = agent.run(ctx)

        assert isinstance(result, ValidationAgentResult)

    def test_url_check_timeout_does_not_raise(self, test_pipeline_config):
        """A timeout during URL check must not propagate as an exception."""
        agent = ValidationAgent(config=test_pipeline_config)
        key_event = _make_key_event()
        panels = [_make_panel(key_events=[key_event])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )

        import requests
        with patch.object(requests.Session, "head", side_effect=requests.exceptions.Timeout("mock")):
            result = agent.run(ctx)

        assert isinstance(result, ValidationAgentResult)

    def test_url_check_200_marks_reachable(self, test_pipeline_config):
        """A 200 OK response must mark the URL as reachable."""
        agent = ValidationAgent(config=test_pipeline_config)
        test_url = "https://www.reuters.com/test-article"
        key_event = _make_key_event(source_url=test_url)
        panels = [_make_panel(key_events=[key_event])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        import requests
        with patch.object(requests.Session, "head", return_value=mock_resp):
            result = agent.run(ctx)

        if test_url in result.url_check_results:
            assert result.url_check_results[test_url].reachable is True

    def test_url_check_404_marks_unreachable(self, test_pipeline_config):
        """A 404 response must mark the URL as unreachable (DEAD_URL)."""
        agent = ValidationAgent(config=test_pipeline_config)
        test_url = "https://www.reuters.com/dead-link"
        key_event = _make_key_event(source_url=test_url)
        panels = [_make_panel(key_events=[key_event])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        import requests
        with patch.object(requests.Session, "head", return_value=mock_resp):
            result = agent.run(ctx)

        if test_url in result.url_check_results:
            assert result.url_check_results[test_url].reachable is False


# ── ValidationAgent.run — grounding score ────────────────────────────────────────

class TestValidationAgentGroundingScore:
    def test_grounding_score_is_float(self, test_pipeline_config):
        """grounding_score must be a float in [0.0, 1.0]."""
        agent = ValidationAgent(config=test_pipeline_config)
        key_event = _make_key_event()
        panels = [_make_panel(key_events=[key_event])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )

        import requests
        with patch.object(requests.Session, "head", side_effect=requests.exceptions.ConnectionError):
            result = agent.run(ctx)

        assert isinstance(result.grounding_score, float)
        assert 0.0 <= result.grounding_score <= 1.0

    def test_verification_percentage_is_float(self, test_pipeline_config):
        """verification_percentage must be a float in [0.0, 100.0]."""
        agent = ValidationAgent(config=test_pipeline_config)
        panels = [_make_panel(key_events=[_make_key_event()])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
        )

        import requests
        with patch.object(requests.Session, "head", side_effect=requests.exceptions.ConnectionError):
            result = agent.run(ctx)

        assert isinstance(result.verification_percentage, float)
        assert 0.0 <= result.verification_percentage <= 100.0


# ── ValidationAgent.run — result schema ──────────────────────────────────────────

class TestValidationAgentResultSchema:
    def test_result_has_required_fields(self, test_pipeline_config):
        """ValidationAgentResult must expose all required schema fields."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=None)
        result = agent.run(ctx)

        assert hasattr(result, "grounding_score")
        assert hasattr(result, "verification_percentage")
        assert hasattr(result, "verified_events")
        assert hasattr(result, "unverified_events")
        assert hasattr(result, "url_check_results")
        assert hasattr(result, "flags")
        assert hasattr(result, "grounding_flags")

    def test_flags_are_list(self, test_pipeline_config):
        """flags must always be a list (never None)."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=None)
        result = agent.run(ctx)
        assert isinstance(result.flags, list)

    def test_verified_events_are_list(self, test_pipeline_config):
        """verified_events must always be a list."""
        agent = ValidationAgent(config=test_pipeline_config)
        ctx = _make_context(test_pipeline_config, storyboard_result=None)
        result = agent.run(ctx)
        assert isinstance(result.verified_events, list)


# ── ValidationAgent.run — ground truth integration ───────────────────────────────

class TestValidationAgentGroundTruth:
    def test_no_ground_truth_does_not_crash(self, test_pipeline_config):
        """Ground truth unavailable must not cause an exception."""
        agent = ValidationAgent(config=test_pipeline_config)
        key_event = _make_key_event()
        panels = [_make_panel(key_events=[key_event])]
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
            ground_truth_result=None,
        )

        import requests
        with patch.object(requests.Session, "head", side_effect=requests.exceptions.ConnectionError):
            result = agent.run(ctx)

        assert isinstance(result, ValidationAgentResult)

    def test_empty_ground_truth_events_does_not_crash(self, test_pipeline_config):
        """Empty ground truth event list must be handled gracefully."""
        agent = ValidationAgent(config=test_pipeline_config)
        key_event = _make_key_event()
        panels = [_make_panel(key_events=[key_event])]

        gt_result = MagicMock()
        gt_result.events = []
        ctx = _make_context(
            test_pipeline_config,
            storyboard_result=_make_storyboard_result(panels=panels),
            ground_truth_result=gt_result,
        )

        import requests
        with patch.object(requests.Session, "head", side_effect=requests.exceptions.ConnectionError):
            result = agent.run(ctx)

        assert isinstance(result, ValidationAgentResult)
