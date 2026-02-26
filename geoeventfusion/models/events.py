"""Event and article data models for GEOEventFusion.

Defines the core data structures produced by the GDELTAgent and GroundTruthAgent.
All fields are typed; no raw dicts are returned from agent code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Article:
    """A single news article from any source (GDELT, RSS, authoritative pool, etc.)."""

    url: str
    title: str
    published_at: str  # ISO 8601 date string, normalized via date_utils
    source: str
    full_text: str = ""
    tone: Optional[float] = None
    domain: str = ""
    language: str = ""
    source_country: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Article):
            return NotImplemented
        return self.url == other.url


@dataclass
class TimelineStep:
    """A single time step from GDELT TimelineVolInfo, TimelineTone, TimelineLang,
    or TimelineSourceCountry modes."""

    date: str          # ISO date string
    value: float       # Volume percentage, tone value, or category-specific value
    label: str = ""    # Category label for multi-series modes (language, country)
    articles: List[Article] = field(default_factory=list)  # Top articles (VolInfo only)


@dataclass
class TimelineStepRaw:
    """A single time step from GDELT TimelineVolRaw mode.

    Provides absolute article counts alongside the norm field (total monitored
    articles per interval) for computing vol_ratio.
    """

    date: str
    volume: int        # Absolute article count
    norm: float        # Total monitored articles in this interval (denominator for vol_ratio)


@dataclass
class ToneChartBin:
    """A single bin from the GDELT ToneChart histogram (−100 to +100)."""

    tone_value: float   # Bin center value
    count: int          # Article count in this bin


@dataclass
class SpikeWindow:
    """A detected coverage spike from Z-score analysis of TimelineVolInfo data."""

    date: str
    z_score: float
    volume: float       # Normalized volume value at the spike date
    rank: int           # 1 = highest Z-score spike
    query: str = ""     # The query that produced this spike


@dataclass
class ToneStats:
    """Aggregated tone statistics from ToneChart analysis."""

    modal_tone: float
    mean_tone: float
    std_dev: float
    polarity_ratio: float   # Negative articles / total articles
    total_articles: int


@dataclass
class LanguageStats:
    """Language coverage statistics from TimelineLang analysis."""

    top_languages: List[Dict[str, Any]]   # [{"language": str, "volume": float}]
    diversity_index: float                 # Shannon diversity index over language distribution


@dataclass
class CountryStats:
    """Source country distribution statistics from TimelineSourceCountry analysis."""

    top_countries: List[Dict[str, Any]]   # [{"country": str, "volume": float}]
    diversity_index: float


@dataclass
class RunMetadata:
    """Metadata captured at pipeline run time for audit and reproducibility."""

    query: str
    final_query: str
    days_back: int
    start_date: str
    end_date: str
    record_counts: Dict[str, int] = field(default_factory=dict)
    active_fetch_modes: List[str] = field(default_factory=list)
    run_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class GroundTruthEvent:
    """A conflict event from a validated ground-truth dataset (ACLED or ICEWS)."""

    event_id: str
    source: str          # "acled" or "icews"
    event_type: str
    date: str
    country: str
    region: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    actors: List[str] = field(default_factory=list)
    fatalities: int = 0
    notes: str = ""
    confidence: float = 1.0


@dataclass
class ImageTopicTag:
    """A visual topic tag from GDELT WordCloudImageTags mode."""

    tag: str
    count: int
    normalized_count: float = 0.0


@dataclass
class GDELTAgentResult:
    """Complete output from the GDELTAgent — all article pools, timelines, and analysis."""

    # ── Core article pools (Group A) ───────────────────────────────────────────
    articles_recent: List[Article] = field(default_factory=list)
    articles_negative: List[Article] = field(default_factory=list)
    articles_positive: List[Article] = field(default_factory=list)
    articles_relevant: List[Article] = field(default_factory=list)
    articles_high_neg: List[Article] = field(default_factory=list)
    articles_high_emotion: List[Article] = field(default_factory=list)

    # ── Conditional source pools (Group C) ────────────────────────────────────
    articles_source_country: List[Article] = field(default_factory=list)
    articles_source_lang: List[Article] = field(default_factory=list)
    articles_authoritative: List[Article] = field(default_factory=list)

    # ── Timeline and signal modes (Group B) ───────────────────────────────────
    timeline_volinfo: List[TimelineStep] = field(default_factory=list)
    timeline_volraw: List[TimelineStepRaw] = field(default_factory=list)
    timeline_tone: List[TimelineStep] = field(default_factory=list)
    timeline_lang: List[TimelineStep] = field(default_factory=list)
    timeline_country: List[TimelineStep] = field(default_factory=list)
    tonechart: List[ToneChartBin] = field(default_factory=list)

    # ── Derived analysis ───────────────────────────────────────────────────────
    spikes: List[SpikeWindow] = field(default_factory=list)
    spike_articles: Dict[str, List[Article]] = field(default_factory=dict)
    title_url_map: Dict[str, List[Article]] = field(default_factory=dict)
    tone_stats: Optional[ToneStats] = None
    language_stats: Optional[LanguageStats] = None
    country_stats: Optional[CountryStats] = None
    vol_ratio: float = 0.0

    # ── Visual intelligence (Group D — populated only when enable_visual_intel=True) ──
    visual_images: List[Any] = field(default_factory=list)   # List[VisualImage]
    image_topics: List[ImageTopicTag] = field(default_factory=list)

    # ── Actor graph ────────────────────────────────────────────────────────────
    actor_graph: Optional[Any] = None   # ActorGraph — avoids circular import

    # ── Run metadata ───────────────────────────────────────────────────────────
    run_metadata: Optional[RunMetadata] = None

    # ── Status ─────────────────────────────────────────────────────────────────
    status: str = "OK"   # "OK", "PARTIAL", "CRITICAL"
    warnings: List[str] = field(default_factory=list)

    def all_articles(self) -> List[Article]:
        """Return a deduplicated union of all article pools."""
        seen: set = set()
        result: List[Article] = []
        for pool in [
            self.articles_recent,
            self.articles_relevant,
            self.articles_negative,
            self.articles_positive,
            self.articles_high_neg,
            self.articles_high_emotion,
            self.articles_source_country,
            self.articles_source_lang,
            self.articles_authoritative,
        ]:
            for article in pool:
                if article.url not in seen:
                    seen.add(article.url)
                    result.append(article)
        return result


@dataclass
class RSSAgentResult:
    """Output from the RSSAgent — enriched full-text articles from RSS/Atom feeds."""

    articles: List[Article] = field(default_factory=list)
    feed_counts: Dict[str, int] = field(default_factory=dict)   # feed_url -> article count
    warnings: List[str] = field(default_factory=list)


@dataclass
class GroundTruthAgentResult:
    """Output from the GroundTruthAgent — validated conflict event records."""

    events: List[GroundTruthEvent] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CustomDatasetMatch:
    """A single match between a GDELT article and a custom dataset record."""

    article: Article
    custom_record: Dict[str, Any]
    match_confidence: float
    match_dimensions: List[str] = field(default_factory=list)


@dataclass
class CustomDatasetAgentResult:
    """Output from the CustomDatasetAgent — matched and unmatched custom records."""

    matches: List[CustomDatasetMatch] = field(default_factory=list)
    unmatched_records: List[Dict[str, Any]] = field(default_factory=list)
    match_rate: float = 0.0
    warnings: List[str] = field(default_factory=list)
