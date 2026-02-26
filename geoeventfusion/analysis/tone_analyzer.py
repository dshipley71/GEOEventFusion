"""Tone analysis functions for GEOEventFusion.

Analyzes GDELT ToneChart histogram data, timeline tone trends,
and language/country coverage statistics. Pure functions — no I/O.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import List

from geoeventfusion.models.events import (
    CountryStats,
    LanguageStats,
    TimelineStep,
    ToneChartBin,
    ToneStats,
)

logger = logging.getLogger(__name__)


def analyze_tone_distribution(tonechart: List[ToneChartBin]) -> ToneStats:
    """Analyze a GDELT ToneChart histogram to produce summary tone statistics.

    Args:
        tonechart: List of ToneChartBin objects from GDELT ToneChart mode.

    Returns:
        ToneStats with modal_tone, mean_tone, std_dev, polarity_ratio, total_articles.
    """
    if not tonechart:
        return ToneStats(
            modal_tone=0.0,
            mean_tone=0.0,
            std_dev=0.0,
            polarity_ratio=0.0,
            total_articles=0,
        )

    total = sum(b.count for b in tonechart)
    if total == 0:
        return ToneStats(
            modal_tone=0.0,
            mean_tone=0.0,
            std_dev=0.0,
            polarity_ratio=0.0,
            total_articles=0,
        )

    # Modal tone: bin with highest count
    modal_bin = max(tonechart, key=lambda b: b.count)
    modal_tone = modal_bin.tone_value

    # Weighted mean tone
    mean_tone = sum(b.tone_value * b.count for b in tonechart) / total

    # Weighted standard deviation
    variance = sum(b.count * (b.tone_value - mean_tone) ** 2 for b in tonechart) / total
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    # Polarity ratio: negative / total
    negative_count = sum(b.count for b in tonechart if b.tone_value < 0)
    polarity_ratio = negative_count / total

    return ToneStats(
        modal_tone=round(modal_tone, 2),
        mean_tone=round(mean_tone, 4),
        std_dev=round(std_dev, 4),
        polarity_ratio=round(polarity_ratio, 4),
        total_articles=total,
    )


def compute_language_stats(timeline_lang: List[TimelineStep]) -> LanguageStats:
    """Compute language coverage statistics from GDELT TimelineLang data.

    Args:
        timeline_lang: List of TimelineStep objects from TimelineLang mode.
            Each step's label is a language name and value is the coverage volume.

    Returns:
        LanguageStats with top_languages list and Shannon diversity index.
    """
    if not timeline_lang:
        return LanguageStats(top_languages=[], diversity_index=0.0)

    # Aggregate volume by language
    lang_volumes: dict = defaultdict(float)
    for step in timeline_lang:
        if step.label:
            lang_volumes[step.label] += step.value

    total = sum(lang_volumes.values())
    if total == 0:
        return LanguageStats(top_languages=[], diversity_index=0.0)

    # Sort by volume descending
    sorted_langs = sorted(lang_volumes.items(), key=lambda x: x[1], reverse=True)
    top_languages = [
        {"language": lang, "volume": round(vol, 4), "share": round(vol / total, 4)}
        for lang, vol in sorted_langs[:20]
    ]

    # Shannon diversity index
    diversity = _shannon_index(lang_volumes)

    return LanguageStats(top_languages=top_languages, diversity_index=round(diversity, 4))


def compute_country_stats(timeline_country: List[TimelineStep]) -> CountryStats:
    """Compute source country distribution from GDELT TimelineSourceCountry data.

    Args:
        timeline_country: List of TimelineStep objects from TimelineSourceCountry mode.

    Returns:
        CountryStats with top_countries list and Shannon diversity index.
    """
    if not timeline_country:
        return CountryStats(top_countries=[], diversity_index=0.0)

    country_volumes: dict = defaultdict(float)
    for step in timeline_country:
        if step.label:
            country_volumes[step.label] += step.value

    total = sum(country_volumes.values())
    if total == 0:
        return CountryStats(top_countries=[], diversity_index=0.0)

    sorted_countries = sorted(country_volumes.items(), key=lambda x: x[1], reverse=True)
    top_countries = [
        {"country": country, "volume": round(vol, 4), "share": round(vol / total, 4)}
        for country, vol in sorted_countries[:30]
    ]

    diversity = _shannon_index(country_volumes)

    return CountryStats(top_countries=top_countries, diversity_index=round(diversity, 4))


def _shannon_index(distribution: dict) -> float:
    """Compute the Shannon diversity index for a frequency distribution.

    Args:
        distribution: Dict mapping category label to count/volume.

    Returns:
        Shannon diversity index H = -sum(p * log(p)).
    """
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            h -= p * math.log(p)
    return h


def compute_tone_trend(timeline_tone: List[TimelineStep]) -> dict:
    """Summarize the tone trend across the analysis window.

    Args:
        timeline_tone: List of TimelineStep objects from TimelineTone mode.

    Returns:
        Dict with early_mean, late_mean, trend_direction, and trend_delta.
    """
    if not timeline_tone:
        return {
            "early_mean": 0.0,
            "late_mean": 0.0,
            "trend_direction": "stable",
            "trend_delta": 0.0,
        }

    n = len(timeline_tone)
    third = max(n // 3, 1)

    early_values = [s.value for s in timeline_tone[:third]]
    late_values = [s.value for s in timeline_tone[-third:]]

    early_mean = sum(early_values) / len(early_values) if early_values else 0.0
    late_mean = sum(late_values) / len(late_values) if late_values else 0.0
    delta = late_mean - early_mean

    if delta < -0.5:
        direction = "deteriorating"
    elif delta > 0.5:
        direction = "improving"
    else:
        direction = "stable"

    return {
        "early_mean": round(early_mean, 3),
        "late_mean": round(late_mean, 3),
        "trend_direction": direction,
        "trend_delta": round(delta, 3),
    }
