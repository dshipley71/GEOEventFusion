"""GEOEventFusion pipeline orchestrator.

Manages agent execution order, phase-level caching, SIGTERM handling, and
the PipelineContext lifecycle for a complete intelligence pipeline run.

Pipeline phase order (per AGENTS.md §1.1):
  Phase 1  — GDELTAgent
  Phase 2  — RSSAgent, GroundTruthAgent, CustomDatasetAgent (parallel-safe)
  Phase 3+ — LLMExtractionAgent
  Phase 5  — FusionAgent, StoryboardAgent, ValidationAgent
  Phase 7  — ExportAgent

Usage:
    from config.settings import PipelineConfig
    from geoeventfusion.pipeline import run

    config = PipelineConfig(query="Houthi Red Sea attacks", days_back=90)
    context = run(config)
"""

from __future__ import annotations

import logging
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import PipelineConfig
from geoeventfusion.agents.base import AgentStatus, BaseAgent
from geoeventfusion.agents.custom_dataset_agent import CustomDatasetAgent
from geoeventfusion.agents.export_agent import ExportAgent
from geoeventfusion.agents.fusion_agent import FusionAgent
from geoeventfusion.agents.gdelt_agent import GDELTAgent
from geoeventfusion.agents.ground_truth_agent import GroundTruthAgent
from geoeventfusion.agents.llm_extraction_agent import LLMExtractionAgent
from geoeventfusion.agents.rss_agent import RSSAgent
from geoeventfusion.agents.storyboard_agent import StoryboardAgent
from geoeventfusion.agents.validation_agent import ValidationAgent
from geoeventfusion.io.persistence import ensure_output_dir
from geoeventfusion.models.pipeline import PhaseRecord, PipelineContext

logger = logging.getLogger(__name__)

# Module-level flag set by SIGTERM handler — checked between phases
_PIPELINE_INTERRUPTED: bool = False


def _sigterm_handler(signum: int, frame: object) -> None:  # pragma: no cover
    """Handle SIGTERM by requesting graceful pipeline shutdown after current phase."""
    global _PIPELINE_INTERRUPTED
    logger.warning(
        "GEOEventFusion: SIGTERM received (signal %d) — pipeline will stop after current phase",
        signum,
    )
    _PIPELINE_INTERRUPTED = True


def _make_run_id(query: str) -> str:
    """Generate a sortable run ID from UTC timestamp and query slug.

    Args:
        query: Pipeline query string.

    Returns:
        Run ID string in the form ``YYYYMMDD_HHMMSS_<slug>``.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower())[:40].strip("_") or "run"
    return f"{timestamp}_{slug}"


def _run_phase(
    context: PipelineContext,
    phase_name: str,
    agent: BaseAgent,
    result_attr: str,
) -> bool:
    """Execute a single pipeline phase and record its timing.

    If the context already has a non-None result for ``result_attr`` (i.e., this
    phase was already completed in a previous call), the phase is skipped.

    Args:
        context: Shared pipeline context.
        phase_name: Human-readable phase label for logs and phase_log.
        agent: Agent instance with a ``run(context)`` method.
        result_attr: Name of the PipelineContext field to write the result to.

    Returns:
        True if the phase completed successfully, False on CRITICAL failure.
    """
    # Skip if already populated (resumable pipeline support)
    if getattr(context, result_attr) is not None:
        logger.info("Pipeline: %s already complete — skipping", phase_name)
        return True

    record: PhaseRecord = context.log_phase_start(phase_name)
    logger.info("Pipeline: starting %s", phase_name)

    try:
        result = agent.run(context)
        setattr(context, result_attr, result)
        status = getattr(result, "status", AgentStatus.OK)
        context.log_phase_end(record, status=str(status))

        if status == AgentStatus.CRITICAL:
            logger.error("Pipeline: %s returned CRITICAL — halting pipeline", phase_name)
            context.add_error(f"{phase_name} CRITICAL: pipeline halted")
            return False

        if hasattr(result, "warnings"):
            for w in result.warnings:
                context.add_warning(f"[{phase_name}] {w}")

        logger.info("Pipeline: %s complete (%.1fs, status=%s)",
                    phase_name, record.elapsed_seconds, status)
        return True

    except Exception as exc:
        context.log_phase_end(record, status=AgentStatus.FAILED)
        logger.exception("Pipeline: %s raised unhandled exception: %s", phase_name, exc)
        context.add_error(f"{phase_name} failed with exception: {exc}")
        return False


def run(
    config: PipelineConfig,
    resume_context: Optional[PipelineContext] = None,
) -> PipelineContext:
    """Execute the full GEOEventFusion pipeline.

    Agents run in the canonical order defined in AGENTS.md §1.1. Each phase is
    guarded by:
      - A SIGTERM interrupt check (pipeline halts after the current phase)
      - A CRITICAL status check (pipeline halts with an error if GDELT fails)
      - A graceful-degradation policy (all other phases continue on failure)

    Args:
        config: Fully-populated PipelineConfig (query, credentials, thresholds).
        resume_context: Optional existing PipelineContext to resume from. If
            provided, phases whose result fields are already populated are skipped.

    Returns:
        PipelineContext with all populated result fields and phase timing log.
    """
    global _PIPELINE_INTERRUPTED
    _PIPELINE_INTERRUPTED = False

    # Register SIGTERM handler so containerised/batch runs exit cleanly
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Initialise context ────────────────────────────────────────────────────
    if resume_context is not None:
        context = resume_context
        logger.info("Pipeline: resuming run %s", context.run_id)
    else:
        run_id = _make_run_id(config.query)
        output_dir = ensure_output_dir(config.output_root, run_id)
        context = PipelineContext(
            config=config,
            run_id=run_id,
            output_dir=output_dir,
        )
        logger.info("Pipeline: starting run %s → %s", run_id, output_dir)

    context.start_time = datetime.utcnow()

    # ── Phase 1: GDELT data ingestion ─────────────────────────────────────────
    if _PIPELINE_INTERRUPTED:
        logger.warning("Pipeline: interrupted before Phase 1 — exiting early")
        _finalise(context)
        return context

    critical = not _run_phase(context, "GDELTAgent", GDELTAgent(), "gdelt_result")
    if critical:
        _finalise(context)
        return context

    # ── Phase 2a: RSS enrichment ──────────────────────────────────────────────
    if _PIPELINE_INTERRUPTED:
        logger.warning("Pipeline: interrupted after Phase 1 — exiting early")
        _finalise(context)
        return context

    _run_phase(context, "RSSAgent", RSSAgent(), "rss_result")

    # ── Phase 2b: Ground-truth dataset ───────────────────────────────────────
    if _PIPELINE_INTERRUPTED:
        _finalise(context)
        return context

    _run_phase(context, "GroundTruthAgent", GroundTruthAgent(), "ground_truth_result")

    # ── Phase 2c: Custom dataset cross-reference ──────────────────────────────
    if _PIPELINE_INTERRUPTED:
        _finalise(context)
        return context

    _run_phase(context, "CustomDatasetAgent", CustomDatasetAgent(), "custom_dataset_result")

    # ── Phase 3/4/6: LLM extraction (timeline, hypotheses, follow-up briefs) ──
    if _PIPELINE_INTERRUPTED:
        _finalise(context)
        return context

    _run_phase(context, "LLMExtractionAgent", LLMExtractionAgent(), "llm_result")

    # ── Phase 5a: Event fusion ────────────────────────────────────────────────
    if _PIPELINE_INTERRUPTED:
        _finalise(context)
        return context

    _run_phase(context, "FusionAgent", FusionAgent(), "fusion_result")

    # ── Phase 5b: Storyboard narrative generation ─────────────────────────────
    if _PIPELINE_INTERRUPTED:
        _finalise(context)
        return context

    _run_phase(context, "StoryboardAgent", StoryboardAgent(), "storyboard_result")

    # ── Phase 5c: Validation and grounding ───────────────────────────────────
    if _PIPELINE_INTERRUPTED:
        _finalise(context)
        return context

    _run_phase(context, "ValidationAgent", ValidationAgent(), "validation_result")

    # ── Phase 7: Export artifacts ─────────────────────────────────────────────
    if _PIPELINE_INTERRUPTED:
        logger.warning("Pipeline: interrupted before ExportAgent — skipping export")
        _finalise(context)
        return context

    _run_phase(context, "ExportAgent", ExportAgent(), "export_result")

    _finalise(context)
    return context


def _finalise(context: PipelineContext) -> None:
    """Record pipeline end time and emit a summary log line.

    Args:
        context: PipelineContext to finalise.
    """
    context.end_time = datetime.utcnow()
    elapsed = (context.end_time - context.start_time).total_seconds() if context.start_time else 0.0
    completed_phases = [p.phase_name for p in context.phase_log if p.status not in
                        (AgentStatus.FAILED, AgentStatus.CRITICAL)]
    logger.info(
        "Pipeline: run %s complete in %.1fs | phases=%s | warnings=%d | errors=%d",
        context.run_id,
        elapsed,
        len(completed_phases),
        len(context.warnings),
        len(context.errors),
    )
    if context.errors:
        for err in context.errors:
            logger.error("Pipeline error: %s", err)
