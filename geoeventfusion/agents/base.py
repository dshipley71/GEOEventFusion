"""BaseAgent ABC and AgentStatus constants for GEOEventFusion.

All pipeline agents inherit from BaseAgent and implement the run() method.
The base class enforces the standard interface: run, validate_output, reset.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geoeventfusion.models.pipeline import PipelineContext

logger = logging.getLogger(__name__)


class AgentStatus:
    """Status codes used in AgentResult.status and GDELTAgentResult.status."""

    OK = "OK"
    PARTIAL = "PARTIAL"
    CRITICAL = "CRITICAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_GROUNDING = "LOW_GROUNDING"
    FAILED = "FAILED"


class BaseAgent(ABC):
    """Abstract base class for all GEOEventFusion pipeline agents.

    Every agent must implement run(). Agents are stateless with respect to
    pipeline data — all state flows through PipelineContext. Internal caches
    or connection pools are reset via reset().

    Args:
        name: Unique agent identifier (e.g. "GDELTAgent").
        version: Semantic version of this agent implementation.
    """

    name: str = "BaseAgent"
    version: str = "1.0.0"

    @abstractmethod
    def run(self, context: "PipelineContext") -> Any:
        """Execute the agent and return a typed result.

        The result should be stored on the context object by the caller
        (pipeline orchestrator) after this method returns.

        Args:
            context: Shared pipeline context with configuration and upstream results.

        Returns:
            A typed agent result dataclass (subclass-specific).
        """

    def validate_output(self, result: Any) -> bool:
        """Post-run validation of structured output.

        Override this method to add agent-specific output checks.

        Args:
            result: The typed result produced by run().

        Returns:
            True if output is valid, False if validation failed.
        """
        return result is not None

    def reset(self) -> None:
        """Clear internal state for re-use or retry.

        Override if the agent maintains connection pools, caches, or
        any mutable state that should be cleared between runs.
        """

    def _run_timed(self, context: "PipelineContext") -> Any:
        """Execute run() and log elapsed time.

        Args:
            context: Shared pipeline context.

        Returns:
            Result from run().
        """
        start = time.monotonic()
        try:
            result = self.run(context)
            elapsed = time.monotonic() - start
            logger.info(
                "Agent %s completed in %.2fs (status=%s)",
                self.name,
                elapsed,
                getattr(result, "status", "?"),
            )
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "Agent %s failed after %.2fs: %s",
                self.name,
                elapsed,
                exc,
                exc_info=True,
            )
            raise
