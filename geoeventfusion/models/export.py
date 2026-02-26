"""Export agent data models for GEOEventFusion.

Defines the export manifest and artifact record schemas produced by the ExportAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArtifactRecord:
    """Metadata record for a single exported artifact file."""

    filename: str
    path: str
    format: str       # "json", "html", "png", "gexf", "csv"
    size_bytes: int = 0
    checksum: str = ""   # SHA-256 hex digest
    description: str = ""


@dataclass
class ExportManifest:
    """Index of all artifacts produced by a pipeline run."""

    run_id: str
    query: str
    output_dir: str
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    total_size_bytes: int = 0
    generation_timestamp: str = ""
    pipeline_version: str = ""

    def artifact_by_format(self, fmt: str) -> List[ArtifactRecord]:
        """Return all artifacts matching the given format string."""
        return [a for a in self.artifacts if a.format == fmt]


@dataclass
class ExportAgentResult:
    """Complete output from the ExportAgent."""

    manifest: Optional[ExportManifest] = None
    exported_paths: Dict[str, str] = field(default_factory=dict)   # artifact_name -> path
    warnings: List[str] = field(default_factory=list)
    status: str = "OK"   # "OK", "PARTIAL", "FAILED"
