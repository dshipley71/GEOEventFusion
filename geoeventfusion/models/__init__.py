"""GEOEventFusion data models package.

All agent input/output schemas are defined here as typed dataclasses.
Never return raw Dict from agent code — always use the typed models.
"""

from geoeventfusion.models.actors import (
    ActorEdge,
    ActorGraph,
    ActorNode,
    CentralityResult,
)
from geoeventfusion.models.events import (
    Article,
    GDELTAgentResult,
    GroundTruthEvent,
    ImageTopicTag,
    LanguageStats,
    RunMetadata,
    SpikeWindow,
    TimelineStep,
    TimelineStepRaw,
    ToneChartBin,
    ToneStats,
    CountryStats,
)
from geoeventfusion.models.export import ArtifactRecord, ExportAgentResult, ExportManifest
from geoeventfusion.models.fusion import (
    ContradictionFlag,
    FusionAgentResult,
    FusionCluster,
    FusionStats,
)
from geoeventfusion.models.pipeline import AgentResult, PhaseRecord, PipelineContext
from geoeventfusion.models.storyboard import (
    Hypothesis,
    LLMExtractionAgentResult,
    StoryboardAgentResult,
    StoryboardPanel,
    TimelineEntry,
    TimelinePhase,
    TurningPoint,
)
from geoeventfusion.models.validation import (
    GroundingFlag,
    ValidationAgentResult,
    VerificationFlag,
)
from geoeventfusion.models.visual import VisualImage

__all__ = [
    # events
    "Article",
    "GDELTAgentResult",
    "GroundTruthEvent",
    "ImageTopicTag",
    "LanguageStats",
    "RunMetadata",
    "SpikeWindow",
    "TimelineStep",
    "TimelineStepRaw",
    "ToneChartBin",
    "ToneStats",
    "CountryStats",
    # actors
    "ActorNode",
    "ActorEdge",
    "ActorGraph",
    "CentralityResult",
    # fusion
    "FusionCluster",
    "FusionAgentResult",
    "ContradictionFlag",
    "FusionStats",
    # storyboard
    "StoryboardPanel",
    "StoryboardAgentResult",
    "Hypothesis",
    "LLMExtractionAgentResult",
    "TimelineEntry",
    "TimelinePhase",
    "TurningPoint",
    # validation
    "ValidationAgentResult",
    "GroundingFlag",
    "VerificationFlag",
    # visual
    "VisualImage",
    # export
    "ExportManifest",
    "ArtifactRecord",
    "ExportAgentResult",
    # pipeline
    "PipelineContext",
    "AgentResult",
    "PhaseRecord",
]
