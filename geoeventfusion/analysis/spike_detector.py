"""Coverage spike detection for GEOEventFusion.

Detects anomalous coverage volume spikes from GDELT TimelineVolInfo data
using Z-score analysis. Pure functions — no I/O or external calls.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from geoeventfusion.models.events import SpikeWindow, TimelineStep

logger = logging.getLogger(__name__)


def detect_spikes(
    timeline: List[TimelineStep],
    z_threshold: float = 1.5,
    query: str = "",
) -> List[SpikeWindow]:
    """Detect coverage spikes from a GDELT TimelineVolInfo series.

    Computes Z-scores over the timeline volume values and returns time steps
    that exceed the threshold, sorted by Z-score descending.

    Zero-variance timelines (all identical values) are handled gracefully —
    no spikes are returned when there is no variation to detect.

    Args:
        timeline: List of TimelineStep objects from GDELT TimelineVolInfo.
        z_threshold: Z-score threshold above which a step is a spike (default: 1.5).
        query: The query string that produced this timeline (stored in spike metadata).

    Returns:
        List of SpikeWindow objects, sorted by Z-score descending.
    """
    if not timeline:
        logger.debug("Spike detection: empty timeline — returning no spikes")
        return []

    values = [step.value for step in timeline]

    # Compute mean and standard deviation
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0.0:
        logger.info("Spike detection: zero-variance timeline — no spikes (query=%.60s)", query)
        return []

    spikes: List[SpikeWindow] = []
    for step in timeline:
        z = (step.value - mean) / std_dev
        if z >= z_threshold:
            spikes.append(
                SpikeWindow(
                    date=step.date,
                    z_score=round(z, 4),
                    volume=step.value,
                    rank=0,  # Will be set after sorting
                    query=query,
                )
            )

    return rank_spikes(spikes)


def rank_spikes(spikes: List[SpikeWindow]) -> List[SpikeWindow]:
    """Sort spikes by Z-score descending and assign rank.

    Args:
        spikes: List of SpikeWindow objects.

    Returns:
        Same list sorted by Z-score descending with rank field set (1 = highest).
    """
    spikes_sorted = sorted(spikes, key=lambda s: s.z_score, reverse=True)
    for i, spike in enumerate(spikes_sorted):
        spike.rank = i + 1
    return spikes_sorted


def compute_vol_ratio(timeline_volraw: list) -> float:
    """Compute vol_ratio from TimelineVolRaw data.

    vol_ratio = mean(volume) / mean(norm) — measures the story's share
    of total GDELT-monitored news space across the analysis window.

    The norm field must NOT be smoothed — always use the raw norm value
    as the denominator (per CLAUDE.md known gotcha #10.1).

    Args:
        timeline_volraw: List of TimelineStepRaw objects.

    Returns:
        vol_ratio float, or 0.0 if data is insufficient.
    """
    if not timeline_volraw:
        return 0.0

    valid_steps = [
        step for step in timeline_volraw
        if hasattr(step, "norm") and step.norm > 0
    ]
    if not valid_steps:
        return 0.0

    mean_volume = sum(s.volume for s in valid_steps) / len(valid_steps)
    mean_norm = sum(s.norm for s in valid_steps) / len(valid_steps)

    if mean_norm == 0:
        return 0.0

    return round(mean_volume / mean_norm, 6)


def find_phase_boundaries(
    spikes: List[SpikeWindow],
    reorganization_score: Optional[float] = None,
    phase_boundary_date: Optional[str] = None,
) -> List[str]:
    """Derive phase boundary candidate dates from spike dates and community reorganization.

    Args:
        spikes: Ranked spike windows.
        reorganization_score: Community Jaccard reorganization score (optional).
        phase_boundary_date: Date candidate from community detection (optional).

    Returns:
        List of phase boundary date strings, sorted chronologically.
    """
    boundary_dates: List[str] = []

    # Top-3 spikes are candidate phase boundaries
    for spike in spikes[:3]:
        if spike.date and spike.date not in boundary_dates:
            boundary_dates.append(spike.date)

    # Community reorganization boundary
    if (
        reorganization_score is not None
        and reorganization_score > 0.4
        and phase_boundary_date
        and phase_boundary_date not in boundary_dates
    ):
        boundary_dates.append(phase_boundary_date)

    return sorted(boundary_dates)
