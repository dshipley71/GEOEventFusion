"""Pipeline orchestration data models for GEOEventFusion.

Defines PipelineContext (shared state object), AgentResult (base result type),
and PhaseRecord (per-phase timing log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from config.settings import PipelineConfig

if TYPE_CHECKING:
    from geoeventfusion.models.events import (
        CustomDatasetAgentResult,
        GDELTAgentResult,
        GroundTruthAgentResult,
        RSSAgentResult,
    )
    from geoeventfusion.models.export import ExportAgentResult
    from geoeventfusion.models.fusion import FusionAgentResult
    from geoeventfusion.models.storyboard import LLMExtractionAgentResult, StoryboardAgentResult
    from geoeventfusion.models.validation import ValidationAgentResult


@dataclass
class AgentResult:
    """Base result type for all agents. All typed results are subclasses or instances."""

    agent_name: str
    status: str = "OK"       # "OK", "PARTIAL", "CRITICAL"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class PhaseRecord:
    """Timing and status record for a single pipeline phase."""

    phase_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "OK"
    warnings: List[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        """Compute elapsed time in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class PipelineContext:
    """Shared state object threaded through all pipeline agents.

    Each agent reads from context fields populated by earlier agents and
    writes its own result to the appropriate context field. No agent reads
    directly from another agent's module — all communication flows here.
    """

    config: PipelineConfig
    run_id: str
    output_dir: Path

    # ── Agent results (populated progressively) ────────────────────────────────
    gdelt_result: Optional[Any] = None          # GDELTAgentResult
    rss_result: Optional[Any] = None            # RSSAgentResult
    ground_truth_result: Optional[Any] = None   # GroundTruthAgentResult
    custom_dataset_result: Optional[Any] = None # CustomDatasetAgentResult
    llm_result: Optional[Any] = None            # LLMExtractionAgentResult
    fusion_result: Optional[Any] = None         # FusionAgentResult
    storyboard_result: Optional[Any] = None     # StoryboardAgentResult
    validation_result: Optional[Any] = None     # ValidationAgentResult
    export_result: Optional[Any] = None         # ExportAgentResult

    # ── Pipeline metadata ──────────────────────────────────────────────────────
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    phase_log: List[PhaseRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def log_phase_start(self, phase_name: str) -> PhaseRecord:
        """Record the start of a pipeline phase."""
        record = PhaseRecord(phase_name=phase_name, start_time=datetime.utcnow())
        self.phase_log.append(record)
        return record

    def log_phase_end(self, record: PhaseRecord, status: str = "OK") -> None:
        """Record the end of a pipeline phase."""
        record.end_time = datetime.utcnow()
        record.status = status

    def add_warning(self, warning: str) -> None:
        """Append a warning to the global pipeline warning list."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Append an error to the global pipeline error list."""
        self.errors.append(error)
