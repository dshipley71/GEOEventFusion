"""Export utility functions for GEOEventFusion.

Handles JSON serialization, HTML file writing, and NetworkX GEXF graph export.
No business logic — file I/O and format conversion only.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def export_json_artifact(data: Any, output_path: str | Path) -> bool:
    """Write a data object to a JSON file via the persistence layer.

    Args:
        data: Data to export (dict, list, or dataclass).
        output_path: Destination file path.

    Returns:
        True on success, False on error.
    """
    from geoeventfusion.io.persistence import save_json

    try:
        save_json(data, output_path)
        logger.info("Exported JSON: %s", output_path)
        return True
    except Exception as exc:
        logger.error("Failed to export JSON to %s: %s", output_path, exc)
        return False


def export_html_artifact(html_content: str, output_path: str | Path) -> bool:
    """Write an HTML string to a file.

    Args:
        html_content: Full HTML document string.
        output_path: Destination file path.

    Returns:
        True on success, False on error.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("Exported HTML: %s", output_path)
        return True
    except OSError as exc:
        logger.error("Failed to export HTML to %s: %s", output_path, exc)
        return False


def export_gexf_graph(graph: Any, output_path: str | Path) -> bool:
    """Export a NetworkX graph to GEXF format for Gephi or other graph tools.

    Args:
        graph: NetworkX Graph object.
        output_path: Destination .gexf file path.

    Returns:
        True on success, False on error.
    """
    try:
        import networkx as nx  # type: ignore[import]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(graph, str(output_path))
        logger.info("Exported GEXF: %s", output_path)
        return True
    except ImportError:
        logger.error("networkx is required for GEXF export")
        return False
    except Exception as exc:
        logger.error("Failed to export GEXF to %s: %s", output_path, exc)
        return False


def build_run_manifest(
    run_id: str,
    query: str,
    output_dir: str | Path,
    pipeline_version: str = "1.0.0",
) -> dict:
    """Build an export manifest dict for all artifacts in a run output directory.

    Scans the output directory for known artifact files and computes their metadata.

    Args:
        run_id: Pipeline run identifier.
        query: The search query string.
        output_dir: Run output directory path.
        pipeline_version: Pipeline version string.

    Returns:
        Dict conforming to the ExportManifest schema.
    """
    from datetime import datetime

    from geoeventfusion.io.persistence import file_checksum

    output_dir = Path(output_dir)
    artifacts = []

    # Known artifact names and their formats
    known_artifacts = [
        ("run_metadata.json", "json", "Pipeline run metadata and timing"),
        ("storyboard.json", "json", "Storyboard panels with evidence citations"),
        ("timeline.json", "json", "Phase-structured timeline with turning points"),
        ("hypotheses.json", "json", "4-round hypothesis debate results"),
        ("validation_report.json", "json", "Grounding scores and verification flags"),
        ("storyboard_report.html", "html", "Full dark-theme HTML storyboard report"),
        ("actor_network.gexf", "gexf", "NetworkX actor graph for Gephi"),
        ("charts/event_timeline_annotated.png", "png", "Coverage volume timeline chart"),
        ("charts/tone_distribution.png", "png", "Tone distribution histogram"),
        ("charts/timeline_language.png", "png", "Language stacked area chart"),
        ("charts/actor_network.png", "png", "Actor network visualization"),
        ("charts/source_country_map.html", "html", "Source country choropleth map"),
    ]

    total_bytes = 0
    for filename, fmt, description in known_artifacts:
        full_path = output_dir / filename
        if full_path.exists():
            size = full_path.stat().st_size
            total_bytes += size
            checksum = file_checksum(full_path)
            artifacts.append(
                {
                    "filename": filename,
                    "path": str(full_path),
                    "format": fmt,
                    "size_bytes": size,
                    "checksum": checksum,
                    "description": description,
                }
            )

    return {
        "run_id": run_id,
        "query": query,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "total_size_bytes": total_bytes,
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_version": pipeline_version,
    }
