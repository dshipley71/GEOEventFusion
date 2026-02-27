"""GEOEventFusion — PipelineConfig and environment-based configuration loading.

All runtime configuration flows through PipelineConfig. No module-level globals,
no hard-coded values. API keys come exclusively from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

from config.defaults import (
    ACTOR_BROKER_RATIO_THRESHOLD,
    ACTOR_HUB_TOP_N,
    ACTOR_PAGERANK_MAX_ITER,
    ANTHROPIC_MODEL,
    DAYS_BACK,
    DEFAULT_LOG_LEVEL,
    DOMAIN_CAP_PCT,
    FUSION_GEOGRAPHIC_THRESHOLD_KM,
    FUSION_TEMPORAL_WINDOW_HOURS,
    FUSION_WEIGHT_ACTOR,
    FUSION_WEIGHT_EVENT_TYPE,
    FUSION_WEIGHT_GEOGRAPHIC,
    FUSION_WEIGHT_SEMANTIC,
    FUSION_WEIGHT_TEMPORAL,
    GDELT_BACKOFF_BASE,
    GDELT_MAX_RETRIES,
    GDELT_MAX_WORKERS,
    GDELT_REQUEST_TIMEOUT,
    GDELT_STAGGER_SECONDS,
    LLM_BACKEND,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_MIN_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_CONFIDENCE,
    MAX_RECORDS,
    MAX_SPIKES,
    MIN_CITATIONS,
    MIN_PANEL_CONFIDENCE,
    NEAR_MIN_TERM_LENGTH,
    NEAR_WINDOW,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OUTPUT_ROOT,
    REPEAT_THRESHOLD,
    RSS_DEDUP_THRESHOLD,
    RSS_MAX_ARTICLES_PER_SPIKE,
    RSS_REQUEST_TIMEOUT,
    RSS_TIME_WINDOW_HOURS,
    SPIKE_BACKFILL_HOURS,
    SPIKE_Z_THRESHOLD,
    TIMELINE_SMOOTH,
    TONEABS_THRESHOLD,
    TONE_NEGATIVE_THRESHOLD,
    VALIDATION_CUSTOM_MATCH_THRESHOLD,
    VALIDATION_DATE_DELTA_DAYS,
    VALIDATION_GROUND_TRUTH_SIMILARITY_THRESHOLD,
    VALIDATION_MIN_CORROBORATION,
    VALIDATION_TITLE_SIMILARITY_THRESHOLD,
    VALIDATION_URL_TIMEOUT,
    VISUAL_STALENESS_HOURS,
)

# Load .env file if present; silently skip if missing
load_dotenv()


@dataclass
class FusionWeights:
    """Configurable weights for multi-dimensional event fusion scoring."""

    temporal: float = FUSION_WEIGHT_TEMPORAL
    geographic: float = FUSION_WEIGHT_GEOGRAPHIC
    actor: float = FUSION_WEIGHT_ACTOR
    semantic: float = FUSION_WEIGHT_SEMANTIC
    event_type: float = FUSION_WEIGHT_EVENT_TYPE

    def __post_init__(self) -> None:
        total = self.temporal + self.geographic + self.actor + self.semantic + self.event_type
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FusionWeights must sum to 1.0, got {total:.4f}")


@dataclass
class PipelineConfig:
    """Single configuration object threaded through all pipeline agents.

    All tuneable thresholds, API keys, model names, and file paths live here.
    Never use module-level globals or hard-coded values in agent code.
    """

    # ── Core query parameters ──────────────────────────────────────────────────
    query: str = ""
    days_back: int = DAYS_BACK
    max_records: int = MAX_RECORDS

    # ── LLM backend ───────────────────────────────────────────────────────────
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", LLM_BACKEND))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL)
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", OLLAMA_MODEL))
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    )
    ollama_api_key: str = field(
        default_factory=lambda: os.getenv("OLLAMA_API_KEY", OLLAMA_API_KEY)
    )
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens: int = LLM_DEFAULT_MAX_TOKENS
    llm_min_max_tokens: int = LLM_MIN_MAX_TOKENS

    # ── API credentials (from environment only) ────────────────────────────────
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    acled_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ACLED_API_KEY"))
    acled_email: Optional[str] = field(default_factory=lambda: os.getenv("ACLED_EMAIL"))

    # ── Spike detection ────────────────────────────────────────────────────────
    spike_z_threshold: float = SPIKE_Z_THRESHOLD
    max_spikes: int = MAX_SPIKES
    spike_backfill_hours: int = SPIKE_BACKFILL_HOURS

    # ── GDELT fetch configuration ──────────────────────────────────────────────
    domain_cap_pct: float = DOMAIN_CAP_PCT
    timeline_smooth: int = TIMELINE_SMOOTH
    repeat_threshold: int = REPEAT_THRESHOLD
    near_window: int = NEAR_WINDOW
    near_min_term_length: int = NEAR_MIN_TERM_LENGTH
    tone_negative_threshold: float = TONE_NEGATIVE_THRESHOLD
    toneabs_threshold: float = TONEABS_THRESHOLD
    gdelt_stagger_seconds: float = GDELT_STAGGER_SECONDS
    gdelt_max_workers: int = GDELT_MAX_WORKERS
    gdelt_max_retries: int = GDELT_MAX_RETRIES
    gdelt_backoff_base: float = GDELT_BACKOFF_BASE
    gdelt_request_timeout: int = GDELT_REQUEST_TIMEOUT

    # ── Conditional GDELT fetch pools ─────────────────────────────────────────
    source_country_filter: Optional[str] = None   # FIPS country code
    source_lang_filter: Optional[str] = None       # ISO 3-char language code
    authoritative_domains: List[str] = field(default_factory=list)
    visual_imagetags: List[str] = field(default_factory=list)
    enable_visual_intel: bool = False
    enable_word_clouds: bool = False

    # ── RSS feed configuration ─────────────────────────────────────────────────
    rss_feed_list: List[str] = field(default_factory=list)
    rss_max_articles_per_spike: int = RSS_MAX_ARTICLES_PER_SPIKE
    rss_time_window_hours: int = RSS_TIME_WINDOW_HOURS
    rss_request_timeout: int = RSS_REQUEST_TIMEOUT
    rss_dedup_threshold: float = RSS_DEDUP_THRESHOLD

    # ── Ground truth datasets ──────────────────────────────────────────────────
    ground_truth_sources: List[str] = field(default_factory=list)
    ground_truth_country_filter: List[str] = field(default_factory=list)
    ground_truth_event_types: List[str] = field(default_factory=list)

    # ── Custom dataset ─────────────────────────────────────────────────────────
    custom_dataset_path: Optional[str] = None
    custom_dataset_format: str = "csv"   # csv, json, sql, api

    # ── Fusion parameters ──────────────────────────────────────────────────────
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    fusion_temporal_window_hours: int = FUSION_TEMPORAL_WINDOW_HOURS
    fusion_geographic_threshold_km: float = FUSION_GEOGRAPHIC_THRESHOLD_KM

    # ── Confidence and citation ────────────────────────────────────────────────
    max_confidence: float = MAX_CONFIDENCE
    min_citations: int = MIN_CITATIONS
    min_panel_confidence: float = MIN_PANEL_CONFIDENCE

    # ── Validation thresholds ──────────────────────────────────────────────────
    validation_title_similarity_threshold: float = VALIDATION_TITLE_SIMILARITY_THRESHOLD
    validation_ground_truth_similarity_threshold: float = (
        VALIDATION_GROUND_TRUTH_SIMILARITY_THRESHOLD
    )
    validation_custom_match_threshold: float = VALIDATION_CUSTOM_MATCH_THRESHOLD
    validation_date_delta_days: int = VALIDATION_DATE_DELTA_DAYS
    validation_url_timeout: int = VALIDATION_URL_TIMEOUT
    validation_min_corroboration: int = VALIDATION_MIN_CORROBORATION

    # ── Actor graph ────────────────────────────────────────────────────────────
    actor_hub_top_n: int = ACTOR_HUB_TOP_N
    actor_broker_ratio_threshold: float = ACTOR_BROKER_RATIO_THRESHOLD
    actor_pagerank_max_iter: int = ACTOR_PAGERANK_MAX_ITER

    # ── Visual intelligence ────────────────────────────────────────────────────
    visual_staleness_hours: int = VISUAL_STALENESS_HOURS

    # ── Output and logging ─────────────────────────────────────────────────────
    output_root: str = field(default_factory=lambda: os.getenv("OUTPUT_ROOT", OUTPUT_ROOT))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL))

    # ── Test mode ─────────────────────────────────────────────────────────────
    # When True, agents use fixture data and make no real API calls
    test_mode: bool = False

    def __post_init__(self) -> None:
        # Clamp max_confidence to hard ceiling
        if self.max_confidence > MAX_CONFIDENCE:
            self.max_confidence = MAX_CONFIDENCE
