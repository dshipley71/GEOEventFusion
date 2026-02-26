"""GEOEventFusion configuration package."""

from config.defaults import (
    ANTHROPIC_MODEL,
    DAYS_BACK,
    DOMAIN_CAP_PCT,
    LLM_BACKEND,
    MAX_CONFIDENCE,
    MAX_RECORDS,
    MIN_CITATIONS,
    OLLAMA_MODEL,
    SPIKE_Z_THRESHOLD,
    TIMELINE_SMOOTH,
    TONE_NEGATIVE_THRESHOLD,
    TONEABS_THRESHOLD,
)
from config.settings import PipelineConfig

__all__ = [
    "PipelineConfig",
    "MAX_CONFIDENCE",
    "MIN_CITATIONS",
    "SPIKE_Z_THRESHOLD",
    "DOMAIN_CAP_PCT",
    "TIMELINE_SMOOTH",
    "TONE_NEGATIVE_THRESHOLD",
    "TONEABS_THRESHOLD",
    "MAX_RECORDS",
    "DAYS_BACK",
    "LLM_BACKEND",
    "ANTHROPIC_MODEL",
    "OLLAMA_MODEL",
]
