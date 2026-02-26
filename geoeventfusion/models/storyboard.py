"""Storyboard and LLM extraction data models for GEOEventFusion.

Defines typed schemas for storyboard panels, hypotheses, timeline phases,
and all structured outputs from the LLMExtractionAgent and StoryboardAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimelineEntry:
    """A single structured event extracted from article text by the LLM."""

    event_type: str
    datetime: str          # YYYY-MM-DD
    country: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    actors: List[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    source_url: str = ""
    source_title: str = ""


@dataclass
class TurningPoint:
    """A phase transition or decisive moment identified in the timeline analysis."""

    date: str
    description: str
    evidence_title: str = ""
    evidence_url: str = ""


@dataclass
class TimelinePhase:
    """A named phase within the overall analysis timeline."""

    label: str
    date_range: Dict[str, str] = field(default_factory=dict)   # {"start": ..., "end": ...}
    description: str = ""
    key_events: List[str] = field(default_factory=list)
    tone_shift: str = ""
    actor_changes: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """A single adversarial hypothesis from the 4-round debate process."""

    id: int
    dimension: str
    claim: str
    supporting_evidence: List[str] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    stress_test_result: str = ""


@dataclass
class PanelKeyEvent:
    """A single verified (or flagged) key event within a storyboard panel."""

    date: str
    description: str
    source_title: str = ""
    source_url: str = ""
    verified: bool = False


@dataclass
class PanelActor:
    """An actor appearing in a storyboard panel with their role and centrality."""

    name: str
    role: str = "Peripheral"   # "Hub", "Broker", "Peripheral"
    centrality_score: float = 0.0


@dataclass
class StoryboardPanel:
    """A single narrative intelligence panel covering a date range and event cluster."""

    panel_id: str
    date_range: Dict[str, str] = field(default_factory=dict)
    headline: str = ""
    key_events: List[PanelKeyEvent] = field(default_factory=list)
    actors: List[PanelActor] = field(default_factory=list)
    narrative_summary: str = ""
    confidence: float = 0.0
    grounded_sources: List[str] = field(default_factory=list)
    unverified_elements: List[str] = field(default_factory=list)
    recommended_followup: List[str] = field(default_factory=list)


@dataclass
class StoryboardAgentResult:
    """Complete output from the StoryboardAgent."""

    query: str = ""
    date_range: Dict[str, str] = field(default_factory=dict)
    panels: List[StoryboardPanel] = field(default_factory=list)
    overall_confidence: float = 0.0
    escalation_risk: float = 0.0
    max_confidence_cap: float = 0.82
    recommended_followup: List[str] = field(default_factory=list)
    generation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    status: str = "OK"   # "OK", "LOW_CONFIDENCE", "CRITICAL"


@dataclass
class LLMExtractionAgentResult:
    """Complete output from the LLMExtractionAgent across all extraction phases."""

    # Phase 3 — Structured timeline
    timeline_events: List[TimelineEntry] = field(default_factory=list)
    timeline_phases: List[TimelinePhase] = field(default_factory=list)
    turning_points: List[TurningPoint] = field(default_factory=list)
    timeline_summary: str = ""
    timeline_confidence: float = 0.0

    # Phase 4 — Adversarial hypotheses
    hypotheses: List[Hypothesis] = field(default_factory=list)

    # Phase 6 — Follow-up enrichment briefs
    followup_briefs: List[Dict[str, Any]] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)
