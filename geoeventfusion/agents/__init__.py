"""GEOEventFusion agents package.

All agents inherit from BaseAgent and operate on PipelineContext.
Agents do not import from each other — all communication flows through context.
"""

from geoeventfusion.agents.base import AgentStatus, BaseAgent

__all__ = [
    "BaseAgent",
    "AgentStatus",
]
