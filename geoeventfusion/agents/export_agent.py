"""ExportAgent — Package and export all intelligence artifacts.

Implements AGENTS.md §2.9 specification:
- JSON export for all structured data
- HTML dark-theme storyboard report
- PNG chart generation (5 visualizations)
- Actor network GEXF export
- Run manifest with artifact index and checksums

Output structure:
  outputs/runs/<YYYYMMDD_HHMMSS>_<sanitized_query>/
  ├── run_metadata.json
  ├── storyboard.json
  ├── timeline.json
  ├── hypotheses.json
  ├── validation_report.json
  ├── storyboard_report.html
  ├── actor_network.gexf
  └── charts/
      ├── event_timeline_annotated.png
      ├── tone_distribution.png
      ├── timeline_language.png
      ├── actor_network.png
      └── source_country_map.html
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from geoeventfusion.agents.base import AgentStatus, BaseAgent
from geoeventfusion.io.exporters import (
    build_run_manifest,
    export_gexf_graph,
    export_html_artifact,
    export_json_artifact,
)
from geoeventfusion.models.export import ArtifactRecord, ExportAgentResult, ExportManifest

logger = logging.getLogger(__name__)


class ExportAgent(BaseAgent):
    """Package and export all intelligence artifacts for the completed pipeline run.

    Writes JSON, HTML, PNG charts, and GEXF graph files to the run output directory.
    Individual failures are logged and skipped — the agent continues to export
    whatever it can.
    """

    name = "ExportAgent"
    version = "1.0.0"

    def run(self, context: Any) -> ExportAgentResult:
        """Export all pipeline artifacts.

        Args:
            context: PipelineContext with config, run_id, output_dir, and all results.

        Returns:
            ExportAgentResult with manifest and exported path map.
        """
        cfg = context.config
        output_dir = context.output_dir
        result = ExportAgentResult()
        exported_paths: Dict[str, str] = {}

        # Ensure output directories exist
        output_dir = Path(output_dir)
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. run_metadata.json ──────────────────────────────────────────────
        try:
            meta = _build_run_metadata(context)
            if export_json_artifact(meta, output_dir / "run_metadata.json"):
                exported_paths["run_metadata"] = str(output_dir / "run_metadata.json")
        except Exception as exc:
            logger.error("ExportAgent: run_metadata.json failed: %s", exc)
            result.warnings.append(f"run_metadata.json export failed: {exc}")

        # ── 2. storyboard.json ────────────────────────────────────────────────
        if context.storyboard_result:
            try:
                storyboard_data = dataclasses.asdict(context.storyboard_result)
                if export_json_artifact(storyboard_data, output_dir / "storyboard.json"):
                    exported_paths["storyboard"] = str(output_dir / "storyboard.json")
            except Exception as exc:
                logger.error("ExportAgent: storyboard.json failed: %s", exc)
                result.warnings.append(f"storyboard.json export failed: {exc}")

        # ── 3. timeline.json ──────────────────────────────────────────────────
        if context.llm_result:
            try:
                timeline_data = {
                    "events": [
                        dataclasses.asdict(e) for e in context.llm_result.timeline_events
                    ],
                    "phases": [
                        dataclasses.asdict(p) for p in context.llm_result.timeline_phases
                    ],
                    "turning_points": [
                        dataclasses.asdict(tp) for tp in context.llm_result.turning_points
                    ],
                    "summary": context.llm_result.timeline_summary,
                    "confidence": context.llm_result.timeline_confidence,
                }
                if export_json_artifact(timeline_data, output_dir / "timeline.json"):
                    exported_paths["timeline"] = str(output_dir / "timeline.json")
            except Exception as exc:
                logger.error("ExportAgent: timeline.json failed: %s", exc)
                result.warnings.append(f"timeline.json export failed: {exc}")

        # ── 4. hypotheses.json ────────────────────────────────────────────────
        if context.llm_result and context.llm_result.hypotheses:
            try:
                hyp_data = {
                    "hypotheses": [
                        dataclasses.asdict(h) for h in context.llm_result.hypotheses
                    ]
                }
                if export_json_artifact(hyp_data, output_dir / "hypotheses.json"):
                    exported_paths["hypotheses"] = str(output_dir / "hypotheses.json")
            except Exception as exc:
                logger.error("ExportAgent: hypotheses.json failed: %s", exc)
                result.warnings.append(f"hypotheses.json export failed: {exc}")

        # ── 5. validation_report.json ─────────────────────────────────────────
        if context.validation_result:
            try:
                validation_data = dataclasses.asdict(context.validation_result)
                if export_json_artifact(validation_data, output_dir / "validation_report.json"):
                    exported_paths["validation_report"] = str(
                        output_dir / "validation_report.json"
                    )
            except Exception as exc:
                logger.error("ExportAgent: validation_report.json failed: %s", exc)
                result.warnings.append(f"validation_report.json export failed: {exc}")

        # ── 6. storyboard_report.html ─────────────────────────────────────────
        if context.storyboard_result:
            try:
                html_path = output_dir / "storyboard_report.html"
                html_content = _render_html_report(context)
                if export_html_artifact(html_content, html_path):
                    exported_paths["storyboard_report"] = str(html_path)
            except Exception as exc:
                logger.error("ExportAgent: storyboard_report.html failed: %s", exc)
                result.warnings.append(f"storyboard_report.html export failed: {exc}")

        # ── 7. actor_network.gexf ─────────────────────────────────────────────
        if context.gdelt_result and context.gdelt_result.actor_graph:
            try:
                gexf_path = output_dir / "actor_network.gexf"
                nx_graph = _build_networkx_graph(context.gdelt_result.actor_graph)
                if nx_graph is not None and export_gexf_graph(nx_graph, gexf_path):
                    exported_paths["actor_network"] = str(gexf_path)
            except Exception as exc:
                logger.error("ExportAgent: actor_network.gexf failed: %s", exc)
                result.warnings.append(f"actor_network.gexf export failed: {exc}")

        # ── 8. Charts ─────────────────────────────────────────────────────────
        chart_paths = _export_charts(context, charts_dir)
        exported_paths.update(chart_paths)
        if not chart_paths:
            result.warnings.append("No charts generated (visualization modules may be unavailable)")

        # ── 9. Build run manifest ─────────────────────────────────────────────
        try:
            manifest_dict = build_run_manifest(
                run_id=context.run_id,
                query=cfg.query,
                output_dir=output_dir,
                pipeline_version="2.0.0",
            )
            artifacts: List[ArtifactRecord] = [
                ArtifactRecord(
                    filename=a["filename"],
                    path=a["path"],
                    format=a["format"],
                    size_bytes=a["size_bytes"],
                    checksum=a["checksum"],
                    description=a["description"],
                )
                for a in manifest_dict.get("artifacts", [])
            ]
            manifest = ExportManifest(
                run_id=manifest_dict.get("run_id", context.run_id),
                query=manifest_dict.get("query", cfg.query),
                output_dir=manifest_dict.get("output_dir", str(output_dir)),
                artifacts=artifacts,
                total_size_bytes=manifest_dict.get("total_size_bytes", 0),
                generation_timestamp=manifest_dict.get("generation_timestamp", ""),
                pipeline_version=manifest_dict.get("pipeline_version", "2.0.0"),
            )
            # Save the manifest
            export_json_artifact(manifest_dict, output_dir / "run_manifest.json")
            result.manifest = manifest
        except Exception as exc:
            logger.error("ExportAgent: manifest build failed: %s", exc)
            result.warnings.append(f"Manifest build failed: {exc}")

        result.exported_paths = exported_paths

        # ── Determine status ──────────────────────────────────────────────────
        if not exported_paths:
            result.status = AgentStatus.FAILED
        elif result.warnings:
            result.status = AgentStatus.PARTIAL
        else:
            result.status = AgentStatus.OK

        logger.info(
            "ExportAgent: exported %d artifacts to %s | status=%s",
            len(exported_paths),
            output_dir,
            result.status,
        )
        return result


# ── Private helpers ──────────────────────────────────────────────────────────────


def _build_run_metadata(context: Any) -> Dict[str, Any]:
    """Build run metadata dict from PipelineContext.

    Args:
        context: PipelineContext.

    Returns:
        Metadata dict.
    """
    cfg = context.config
    meta: Dict[str, Any] = {
        "run_id": context.run_id,
        "query": cfg.query,
        "days_back": cfg.days_back,
        "llm_backend": cfg.llm_backend,
        "start_time": (
            context.start_time.isoformat() + "Z" if context.start_time else None
        ),
        "end_time": (
            context.end_time.isoformat() + "Z" if context.end_time else None
        ),
        "warnings": context.warnings,
        "errors": context.errors,
        "phase_log": [
            {
                "phase": p.phase_name,
                "status": p.status,
                "elapsed_seconds": p.elapsed_seconds,
            }
            for p in context.phase_log
        ],
    }

    if context.gdelt_result and context.gdelt_result.run_metadata:
        gdelt_meta = context.gdelt_result.run_metadata
        meta.update({
            "final_query": gdelt_meta.final_query,
            "start_date": gdelt_meta.start_date,
            "end_date": gdelt_meta.end_date,
            "record_counts": gdelt_meta.record_counts,
            "active_fetch_modes": gdelt_meta.active_fetch_modes,
        })

    return meta


def _render_html_report(context: Any) -> str:
    """Render the full HTML storyboard report.

    Attempts to use the visualization.html_report module; falls back to a
    minimal HTML template if the module is unavailable or raises.

    Args:
        context: PipelineContext.

    Returns:
        HTML string.
    """
    try:
        from geoeventfusion.visualization.html_report import render_html_report

        html = render_html_report(
            storyboard_result=context.storyboard_result,
            validation_result=context.validation_result,
            run_metadata=_build_run_metadata(context),
        )
        if html is not None:
            return html
    except (ImportError, AttributeError) as exc:
        logger.warning("HTML report renderer unavailable (%s) — using minimal fallback", exc)
    except Exception as exc:
        logger.warning("HTML report render failed: %s — using minimal fallback", exc)

    # Minimal fallback HTML
    storyboard = context.storyboard_result
    query = context.config.query
    panels_html = ""
    if storyboard:
        for panel in storyboard.panels:
            events_html = "".join(
                f"<li>{ke.date}: {ke.description} "
                f'(<a href="{ke.source_url}">{ke.source_title or ke.source_url}</a>)</li>'
                for ke in panel.key_events
            )
            panels_html += (
                f"<section><h2>{panel.headline}</h2>"
                f"<p>{panel.narrative_summary}</p>"
                f"<ul>{events_html}</ul>"
                f"<p>Confidence: {panel.confidence:.2f}</p></section>"
            )

    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>GEOEventFusion: {query}</title>"
        "<style>body{{background:#0A0E17;color:#E5E7EB;font-family:sans-serif;"
        "max-width:900px;margin:auto;padding:2em}}"
        "h1,h2{{color:#60A5FA}}section{{background:#111827;padding:1.5em;"
        "margin:1em 0;border-radius:8px}}a{{color:#60A5FA}}</style></head>"
        f"<body><h1>GEOEventFusion Intelligence Report</h1>"
        f"<h2>Query: {query}</h2>"
        f"{panels_html}"
        "</body></html>"
    )


def _build_networkx_graph(actor_graph: Any) -> Optional[Any]:
    """Reconstruct a NetworkX graph from an ActorGraph model for GEXF export.

    Args:
        actor_graph: ActorGraph dataclass.

    Returns:
        NetworkX Graph, or None if networkx is unavailable.
    """
    try:
        import networkx as nx

        G = nx.Graph()
        for node in actor_graph.nodes:
            G.add_node(
                node.name,
                role=node.role,
                pagerank=node.pagerank,
                degree=node.degree_centrality,
                community=node.community_id if node.community_id is not None else -1,
            )
        for edge in actor_graph.edges:
            G.add_edge(edge.actor_a, edge.actor_b, weight=edge.weight)
        return G
    except ImportError:
        logger.warning("networkx not available — GEXF export skipped")
        return None
    except Exception as exc:
        logger.warning("Failed to build NetworkX graph: %s", exc)
        return None


def _export_charts(context: Any, charts_dir: Path) -> Dict[str, str]:
    """Export all visualization charts.

    Attempts each chart independently; failures are logged and skipped.

    Args:
        context: PipelineContext.
        charts_dir: Path to the charts output directory.

    Returns:
        Dict mapping chart name → file path for successfully exported charts.
    """
    paths: Dict[str, str] = {}

    gdelt = context.gdelt_result

    # Timeline chart
    if gdelt and gdelt.timeline_volinfo:
        try:
            from geoeventfusion.visualization.timeline_chart import render_timeline_chart

            chart_path = charts_dir / "event_timeline_annotated.png"
            result_path = render_timeline_chart(
                timeline_volinfo=gdelt.timeline_volinfo,
                timeline_volraw=gdelt.timeline_volraw if gdelt else None,
                spikes=gdelt.spikes if gdelt else None,
                phase_boundaries=None,
                output_path=chart_path,
            )
            if result_path:
                paths["timeline_chart"] = str(result_path)
        except (ImportError, AttributeError) as exc:
            logger.debug("Timeline chart skipped: %s", exc)
        except Exception as exc:
            logger.warning("Timeline chart failed: %s", exc)

    # Tone distribution chart
    if gdelt and gdelt.tonechart:
        try:
            from geoeventfusion.visualization.tone_chart import render_tone_chart

            chart_path = charts_dir / "tone_distribution.png"
            result_path = render_tone_chart(
                tonechart=gdelt.tonechart,
                tone_stats=gdelt.tone_stats if gdelt else None,
                output_path=chart_path,
            )
            if result_path:
                paths["tone_chart"] = str(result_path)
        except (ImportError, AttributeError) as exc:
            logger.debug("Tone chart skipped: %s", exc)
        except Exception as exc:
            logger.warning("Tone chart failed: %s", exc)

    # Language chart
    if gdelt and gdelt.timeline_lang:
        try:
            from geoeventfusion.visualization.language_chart import render_language_chart

            chart_path = charts_dir / "timeline_language.png"
            result_path = render_language_chart(
                timeline_lang=gdelt.timeline_lang,
                output_path=chart_path,
            )
            if result_path:
                paths["language_chart"] = str(result_path)
        except (ImportError, AttributeError) as exc:
            logger.debug("Language chart skipped: %s", exc)
        except Exception as exc:
            logger.warning("Language chart failed: %s", exc)

    # Actor network chart
    if gdelt and gdelt.actor_graph:
        try:
            from geoeventfusion.visualization.actor_network_chart import render_actor_network_chart

            chart_path = charts_dir / "actor_network.png"
            result_path = render_actor_network_chart(
                actor_graph=gdelt.actor_graph,
                output_path=chart_path,
            )
            if result_path:
                paths["actor_network_chart"] = str(result_path)
        except (ImportError, AttributeError) as exc:
            logger.debug("Actor network chart skipped: %s", exc)
        except Exception as exc:
            logger.warning("Actor network chart failed: %s", exc)

    # Choropleth / source country map
    if gdelt and gdelt.country_stats:
        try:
            from geoeventfusion.visualization.choropleth_chart import render_choropleth_map

            chart_path = charts_dir / "source_country_map.html"
            result_path = render_choropleth_map(
                country_stats=gdelt.country_stats,
                output_path=chart_path,
            )
            if result_path:
                paths["choropleth_chart"] = str(result_path)
        except (ImportError, AttributeError) as exc:
            logger.debug("Choropleth chart skipped: %s", exc)
        except Exception as exc:
            logger.warning("Choropleth chart failed: %s", exc)

    return paths
