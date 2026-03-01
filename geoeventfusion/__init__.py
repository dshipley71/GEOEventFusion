"""GEOEventFusion — Modular multi-agent geopolitical intelligence pipeline.

Public API surface:
    - PipelineConfig: Runtime configuration
    - PipelineContext: Shared state threaded through all agents
    - run_pipeline: Convenience entry point for the full pipeline
"""

__version__ = "1.0.0"
__author__ = "GEOEventFusion Contributors"

from config.settings import PipelineConfig
from geoeventfusion.pipeline import run, run_pipeline

__all__ = [
    "__version__",
    "PipelineConfig",
    "run",
    "run_pipeline",
]
