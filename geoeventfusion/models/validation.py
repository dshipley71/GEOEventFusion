"""Validation agent data models for GEOEventFusion.

Defines typed schemas for grounding scores, verification flags,
and the full validation report produced by the ValidationAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerificationFlag:
    """A specific validation issue detected during evidence-grounding checks."""

    flag_type: str    # "UNVERIFIED_CLAIM", "TIMESTAMP_MISMATCH", "DEAD_URL", "LOW_CORROBORATION"
    severity: str     # "WARNING", "ERROR", "CRITICAL"
    detail: str = ""
    panel_id: Optional[str] = None
    event_description: Optional[str] = None


@dataclass
class GroundingFlag:
    """Grounding check result for a single claim or event."""

    claim: str
    source_url: str = ""
    source_title: str = ""
    title_similarity: float = 0.0
    url_reachable: Optional[bool] = None
    timestamp_consistent: Optional[bool] = None
    corroboration_count: int = 0
    ground_truth_match: Optional[bool] = None
    custom_dataset_match: Optional[bool] = None
    grounded: bool = False


@dataclass
class UrlCheckResult:
    """Result of an HTTP HEAD check on a cited article URL."""

    url: str
    status_code: Optional[int] = None
    reachable: bool = False
    error: Optional[str] = None


@dataclass
class ValidationAgentResult:
    """Complete output from the ValidationAgent."""

    grounding_score: float = 0.0
    verification_percentage: float = 0.0
    verified_events: List[Any] = field(default_factory=list)
    unverified_events: List[Any] = field(default_factory=list)
    url_check_results: Dict[str, UrlCheckResult] = field(default_factory=dict)
    cross_source_corroboration: Dict[str, int] = field(default_factory=dict)
    ground_truth_matches: List[Dict[str, Any]] = field(default_factory=list)
    grounding_flags: List[GroundingFlag] = field(default_factory=list)
    flags: List[VerificationFlag] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "OK"   # "OK", "LOW_GROUNDING", "CRITICAL"
