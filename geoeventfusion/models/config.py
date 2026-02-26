"""Configuration data models for GEOEventFusion.

Re-exports PipelineConfig from config.settings and defines lightweight
sub-config dataclasses (LLMConfig, GDELTConfig) for typed grouping of
related configuration fields.

These are convenience wrappers — the canonical source of all defaults
is config/defaults.py and the full pipeline config lives in config/settings.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config.defaults import (
    ANTHROPIC_MODEL,
    GDELT_BACKOFF_BASE,
    GDELT_MAX_RETRIES,
    GDELT_MAX_WORKERS,
    GDELT_REQUEST_TIMEOUT,
    GDELT_STAGGER_SECONDS,
    LLM_BACKEND,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_MIN_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_RECORDS,
    OLLAMA_MODEL,
    REPEAT_THRESHOLD,
    TIMELINE_SMOOTH,
    TONEABS_THRESHOLD,
    TONE_NEGATIVE_THRESHOLD,
    VISUAL_STALENESS_HOURS,
)

# Re-export the primary config object so downstream code can import from either location
from config.settings import FusionWeights, PipelineConfig

__all__ = [
    "PipelineConfig",
    "FusionWeights",
    "LLMConfig",
    "GDELTConfig",
]


@dataclass
class LLMConfig:
    """Typed sub-config grouping all LLM-related parameters.

    Can be used to construct LLMClient without passing a full PipelineConfig.
    """

    backend: str = LLM_BACKEND
    anthropic_model: str = ANTHROPIC_MODEL
    ollama_model: str = OLLAMA_MODEL
    ollama_host: str = "http://localhost:11434"
    anthropic_api_key: Optional[str] = None
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS
    min_max_tokens: int = LLM_MIN_MAX_TOKENS


@dataclass
class GDELTConfig:
    """Typed sub-config grouping all GDELT fetch parameters.

    Can be used to construct GDELTClient without passing a full PipelineConfig.
    """

    max_records: int = MAX_RECORDS
    timeline_smooth: int = TIMELINE_SMOOTH
    repeat_threshold: int = REPEAT_THRESHOLD
    tone_negative_threshold: float = TONE_NEGATIVE_THRESHOLD
    toneabs_threshold: float = TONEABS_THRESHOLD
    stagger_seconds: float = GDELT_STAGGER_SECONDS
    max_workers: int = GDELT_MAX_WORKERS
    max_retries: int = GDELT_MAX_RETRIES
    backoff_base: float = GDELT_BACKOFF_BASE
    request_timeout: int = GDELT_REQUEST_TIMEOUT
    source_country_filter: Optional[str] = None
    source_lang_filter: Optional[str] = None
    authoritative_domains: List[str] = field(default_factory=list)
    visual_imagetags: List[str] = field(default_factory=list)
    enable_visual_intel: bool = False
    enable_word_clouds: bool = False
    visual_staleness_hours: int = VISUAL_STALENESS_HOURS
