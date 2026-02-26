"""Integration tests for geoeventfusion.agents.export_agent.

Tests the ExportAgent end-to-end against a temporary output directory,
verifying that each artifact is written to disk correctly.

Covers:
- Minimal run: run_metadata.json always written
- JSON artifacts: storyboard.json, timeline.json, hypotheses.json, validation_report.json
- Output directory structure: charts/ subdirectory created
- Partial pipeline results: agent continues when upstream results are None
- Manifest: returned ExportAgentResult contains exported path keys
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from geoeventfusion.agents.export_agent import ExportAgent
from geoeventfusion.models.export import ExportAgentResult
from geoeventfusion.models.storyboard import (
    Hypothesis,
    LLMExtractionAgentResult,
    PanelActor,
    PanelKeyEvent,
    StoryboardAgentResult,
    StoryboardPanel,
    TimelineEntry,
    TimelinePhase,
    TurningPoint,
)
from geoeventfusion.models.validation import ValidationAgentResult


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _make_minimal_context(config, output_dir: Path):
    """Build a minimal PipelineContext mock — all upstream results are None."""
    ctx = MagicMock()
    ctx.config = config
    ctx.run_id = "20240115_120000_test"
    ctx.output_dir = output_dir
    ctx.gdelt_result = None
    ctx.rss_result = None
    ctx.ground_truth_result = None
    ctx.custom_dataset_result = None
    ctx.llm_result = None
    ctx.fusion_result = None
    ctx.storyboard_result = None
    ctx.validation_result = None
    ctx.start_time = None
    ctx.end_time = None
    ctx.phase_log = []
    ctx.warnings = []
    ctx.errors = []
    return ctx


def _make_storyboard_result():
    """Build a minimal StoryboardAgentResult with one panel."""
    panel = StoryboardPanel(
        panel_id="panel_001",
        date_range={"start": "2024-01-01", "end": "2024-01-31"},
        headline="Test Panel Headline",
        key_events=[
            PanelKeyEvent(
                date="2024-01-15",
                description="Test event description",
                source_title="Test article",
                source_url="https://example.com/test",
                verified=True,
            )
        ],
        actors=[PanelActor(name="Houthi", role="Hub", centrality_score=0.8)],
        narrative_summary="Test narrative summary.",
        confidence=0.74,
        grounded_sources=["https://example.com/test"],
        unverified_elements=[],
        recommended_followup=["Follow-up query"],
    )
    result = StoryboardAgentResult(
        query="Houthi Red Sea attacks",
        date_range={"start": "2024-01-01", "end": "2024-01-31"},
        panels=[panel],
        overall_confidence=0.74,
        escalation_risk=0.55,
        recommended_followup=[],
    )
    return result


def _make_llm_result():
    """Build a minimal LLMExtractionAgentResult."""
    result = LLMExtractionAgentResult(
        timeline_events=[
            TimelineEntry(
                event_type="CONFLICT",
                datetime="2024-01-15",
                country="Yemen",
                lat=15.5,
                lon=48.5,
                actors=["Houthi", "US"],
                summary="Test event",
                confidence=0.72,
                source_url="https://example.com/test",
                source_title="Test article",
            )
        ],
        timeline_phases=[
            TimelinePhase(
                label="Phase 1",
                date_range={"start": "2024-01-01", "end": "2024-01-31"},
                description="Initial phase",
            )
        ],
        turning_points=[
            TurningPoint(
                date="2024-01-15",
                description="Key turning point",
                evidence_title="Test article",
                evidence_url="https://example.com/test",
            )
        ],
        hypotheses=[
            Hypothesis(
                id=1,
                dimension="MILITARY",
                claim="Houthi forces are escalating attacks",
                supporting_evidence=["Evidence A"],
                counter_evidence=["Counter B"],
                confidence=0.70,
                stress_test_result="Hypothesis survives stress test",
            )
        ],
        timeline_summary="Test summary",
        timeline_confidence=0.74,
    )
    return result


# ── Minimal run ───────────────────────────────────────────────────────────────────

class TestExportAgentMinimal:
    def test_run_empty_context_returns_result(self, test_pipeline_config, tmp_path):
        """ExportAgent must return an ExportAgentResult even with all upstream results None."""
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, tmp_path / "run_001")
        result = agent.run(ctx)
        assert isinstance(result, ExportAgentResult)

    def test_run_creates_output_directory(self, test_pipeline_config, tmp_path):
        """The output directory must be created if it doesn't exist."""
        output_dir = tmp_path / "new_run_dir"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        agent.run(ctx)
        assert output_dir.exists()

    def test_run_creates_charts_subdirectory(self, test_pipeline_config, tmp_path):
        """A charts/ subdirectory must always be created."""
        output_dir = tmp_path / "run_002"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        agent.run(ctx)
        assert (output_dir / "charts").exists()

    def test_run_metadata_json_always_written(self, test_pipeline_config, tmp_path):
        """run_metadata.json must be written on every pipeline run."""
        output_dir = tmp_path / "run_003"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        agent.run(ctx)
        assert (output_dir / "run_metadata.json").exists()

    def test_run_metadata_json_is_valid_json(self, test_pipeline_config, tmp_path):
        """run_metadata.json must be parseable as valid JSON."""
        output_dir = tmp_path / "run_004"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        agent.run(ctx)
        with open(output_dir / "run_metadata.json", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ── JSON artifact export ──────────────────────────────────────────────────────────

class TestExportAgentJSONArtifacts:
    def test_storyboard_json_written(self, test_pipeline_config, tmp_path):
        """storyboard.json must be written when storyboard_result is present."""
        output_dir = tmp_path / "run_sb"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.storyboard_result = _make_storyboard_result()
        agent.run(ctx)
        assert (output_dir / "storyboard.json").exists()

    def test_storyboard_json_valid(self, test_pipeline_config, tmp_path):
        """storyboard.json must be valid JSON with a 'panels' key."""
        output_dir = tmp_path / "run_sb2"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.storyboard_result = _make_storyboard_result()
        agent.run(ctx)
        with open(output_dir / "storyboard.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "panels" in data

    def test_timeline_json_written(self, test_pipeline_config, tmp_path):
        """timeline.json must be written when llm_result is present."""
        output_dir = tmp_path / "run_tl"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.llm_result = _make_llm_result()
        agent.run(ctx)
        assert (output_dir / "timeline.json").exists()

    def test_timeline_json_has_events_key(self, test_pipeline_config, tmp_path):
        """timeline.json must contain an 'events' key."""
        output_dir = tmp_path / "run_tl2"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.llm_result = _make_llm_result()
        agent.run(ctx)
        with open(output_dir / "timeline.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "events" in data

    def test_hypotheses_json_written(self, test_pipeline_config, tmp_path):
        """hypotheses.json must be written when llm_result has hypotheses."""
        output_dir = tmp_path / "run_hyp"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.llm_result = _make_llm_result()
        agent.run(ctx)
        assert (output_dir / "hypotheses.json").exists()

    def test_validation_report_json_written(self, test_pipeline_config, tmp_path):
        """validation_report.json must be written when validation_result is present."""
        output_dir = tmp_path / "run_vr"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.validation_result = ValidationAgentResult(
            grounding_score=0.72,
            verification_percentage=80.0,
        )
        agent.run(ctx)
        assert (output_dir / "validation_report.json").exists()


# ── ExportAgentResult schema ──────────────────────────────────────────────────────

class TestExportAgentResultSchema:
    def test_result_has_exported_paths(self, test_pipeline_config, tmp_path):
        """ExportAgentResult must have an exported_paths dict."""
        output_dir = tmp_path / "run_schema"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        result = agent.run(ctx)
        assert hasattr(result, "exported_paths")
        assert isinstance(result.exported_paths, dict)

    def test_result_run_metadata_path_in_exported_paths(self, test_pipeline_config, tmp_path):
        """After successful export, 'run_metadata' key must appear in exported_paths."""
        output_dir = tmp_path / "run_paths"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        result = agent.run(ctx)
        assert "run_metadata" in result.exported_paths

    def test_result_status_ok_for_successful_export(self, test_pipeline_config, tmp_path):
        """Status must be 'OK' when export completes without critical failures."""
        output_dir = tmp_path / "run_status"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        result = agent.run(ctx)
        assert result.status == "OK"


# ── Partial pipeline graceful degradation ────────────────────────────────────────

class TestExportAgentPartialPipeline:
    def test_missing_storyboard_does_not_prevent_metadata_export(
        self, test_pipeline_config, tmp_path
    ):
        """If storyboard_result is None, run_metadata.json must still be written."""
        output_dir = tmp_path / "run_partial_sb"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.storyboard_result = None
        agent.run(ctx)
        assert (output_dir / "run_metadata.json").exists()

    def test_missing_llm_result_does_not_crash(self, test_pipeline_config, tmp_path):
        """If llm_result is None, the export must complete without raising."""
        output_dir = tmp_path / "run_partial_llm"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.llm_result = None
        result = agent.run(ctx)
        assert isinstance(result, ExportAgentResult)

    def test_all_results_present_writes_all_json_artifacts(
        self, test_pipeline_config, tmp_path
    ):
        """With all upstream results present, all JSON artifacts must be written."""
        output_dir = tmp_path / "run_full"
        agent = ExportAgent(config=test_pipeline_config)
        ctx = _make_minimal_context(test_pipeline_config, output_dir)
        ctx.storyboard_result = _make_storyboard_result()
        ctx.llm_result = _make_llm_result()
        ctx.validation_result = ValidationAgentResult(grounding_score=0.75)
        agent.run(ctx)

        assert (output_dir / "run_metadata.json").exists()
        assert (output_dir / "storyboard.json").exists()
        assert (output_dir / "timeline.json").exists()
        assert (output_dir / "hypotheses.json").exists()
        assert (output_dir / "validation_report.json").exists()
