"""HTML storyboard report renderer for GEOEventFusion.

Generates a full dark-theme HTML storyboard report with embedded evidence,
confidence indicators, and panel summaries.
Rendering only — no data transformation.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def render_html_report(
    storyboard_result: Any,
    validation_result: Optional[Any] = None,
    run_metadata: Optional[Any] = None,
    output_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Generate the full HTML storyboard intelligence report.

    Args:
        storyboard_result: StoryboardAgentResult containing panels.
        validation_result: Optional ValidationAgentResult for grounding scores.
        run_metadata: Optional RunMetadata for report header.
        output_path: If provided, save the HTML to this path.

    Returns:
        HTML string if generated successfully, None on error.
    """
    try:
        from geoeventfusion.visualization.theme import (
            THEME_BACKGROUND,
            THEME_PANEL,
            THEME_TEXT,
            THEME_ACCENT,
            THEME_SPIKE,
            THEME_BORDER,
            THEME_MUTED,
            escalation_color,
        )

        if not storyboard_result:
            logger.warning("HTML report: no storyboard result — skipping")
            return None

        panels = getattr(storyboard_result, "panels", [])
        query = getattr(storyboard_result, "query", "")
        overall_confidence = getattr(storyboard_result, "overall_confidence", 0.0)
        escalation_risk = getattr(storyboard_result, "escalation_risk", 0.0)
        generation_timestamp = getattr(storyboard_result, "generation_timestamp", "")
        grounding_score = getattr(validation_result, "grounding_score", None) if validation_result else None
        escalation_clr = escalation_color(escalation_risk)

        # ── Document header ────────────────────────────────────────────────────
        html_parts = [
            f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GEOEventFusion — {html.escape(query)}</title>
<style>
  :root {{
    --bg: {THEME_BACKGROUND};
    --panel: {THEME_PANEL};
    --text: {THEME_TEXT};
    --accent: {THEME_ACCENT};
    --spike: {THEME_SPIKE};
    --border: {THEME_BORDER};
    --muted: {THEME_MUTED};
    --escalation: {escalation_clr};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    padding: 24px;
  }}
  .report-header {{
    border-bottom: 2px solid var(--accent);
    padding-bottom: 20px;
    margin-bottom: 28px;
  }}
  .report-title {{
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 8px;
  }}
  .report-meta {{
    color: var(--muted);
    font-size: 12px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }}
  .metric-badge {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
  }}
  .metric-badge span {{
    font-weight: 600;
  }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
  }}
  .panel-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 12px;
    gap: 16px;
  }}
  .panel-headline {{
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    flex: 1;
  }}
  .panel-date {{
    color: var(--muted);
    font-size: 11px;
    white-space: nowrap;
  }}
  .confidence-bar-wrap {{
    margin: 8px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .confidence-bar {{
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    flex: 1;
    overflow: hidden;
  }}
  .confidence-fill {{
    height: 100%;
    border-radius: 3px;
    background: var(--accent);
  }}
  .confidence-label {{
    font-size: 11px;
    color: var(--muted);
    width: 40px;
    text-align: right;
  }}
  .narrative {{
    margin: 10px 0;
    color: var(--text);
    line-height: 1.7;
  }}
  .key-events {{
    margin: 12px 0;
    border-left: 3px solid var(--accent);
    padding-left: 14px;
  }}
  .key-event {{
    margin-bottom: 8px;
    font-size: 13px;
  }}
  .key-event-date {{
    color: var(--spike);
    font-weight: 600;
    margin-right: 6px;
  }}
  .key-event-verified {{
    color: #10B981;
    font-size: 10px;
    margin-left: 6px;
  }}
  .key-event-unverified {{
    color: #6B7280;
    font-size: 10px;
    margin-left: 6px;
  }}
  .source-link {{
    color: var(--muted);
    font-size: 11px;
    text-decoration: none;
    margin-left: 6px;
  }}
  .source-link:hover {{ color: var(--accent); }}
  .actors-section {{
    margin: 12px 0;
  }}
  .actors-header {{
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }}
  .actor-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    margin: 2px;
    font-weight: 500;
  }}
  .actor-hub {{ background: rgba(239,68,68,0.2); color: #FCA5A5; border: 1px solid #EF4444; }}
  .actor-broker {{ background: rgba(245,158,11,0.2); color: #FCD34D; border: 1px solid #F59E0B; }}
  .actor-peripheral {{ background: rgba(107,114,128,0.2); color: #9CA3AF; border: 1px solid #6B7280; }}
  .unverified-flags {{
    margin: 10px 0;
    padding: 8px 12px;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 4px;
    font-size: 12px;
    color: #FCD34D;
  }}
  .section-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 6px;
    margin-top: 14px;
  }}
  .escalation-indicator {{
    font-size: 12px;
    font-weight: 600;
    color: var(--escalation);
    padding: 3px 10px;
    border: 1px solid var(--escalation);
    border-radius: 4px;
  }}
  footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
    text-align: center;
  }}
</style>
</head>
<body>
""",
            # ── Report header ──────────────────────────────────────────────────
            f"""<div class="report-header">
  <div class="report-title">GEOEventFusion Intelligence Report</div>
  <div style="font-size:14px;color:{THEME_TEXT};margin-bottom:12px;">{html.escape(query)}</div>
  <div class="report-meta">
    <div class="metric-badge">Overall Confidence: <span>{overall_confidence:.0%}</span></div>
    <div class="metric-badge" style="border-color:{escalation_clr};">
      Escalation Risk: <span style="color:{escalation_clr};">{escalation_risk:.0%}</span>
    </div>
""",
        ]

        if grounding_score is not None:
            html_parts.append(
                f'    <div class="metric-badge">Grounding Score: <span>{grounding_score:.0%}</span></div>\n'
            )

        if generation_timestamp:
            html_parts.append(
                f'    <div class="metric-badge">Generated: <span>{html.escape(generation_timestamp[:19])}</span></div>\n'
            )

        html_parts.append("  </div>\n</div>\n")

        # ── Panels ─────────────────────────────────────────────────────────────
        for panel in panels:
            panel_id = getattr(panel, "panel_id", "")
            headline = getattr(panel, "headline", "")
            date_range = getattr(panel, "date_range", {})
            confidence = getattr(panel, "confidence", 0.0)
            narrative = getattr(panel, "narrative_summary", "")
            key_events = getattr(panel, "key_events", [])
            actors = getattr(panel, "actors", [])
            unverified = getattr(panel, "unverified_elements", [])
            date_str = f"{date_range.get('start', '')} → {date_range.get('end', '')}"

            html_parts.append(
                f'<div class="panel" id="{html.escape(panel_id)}">\n'
                f'  <div class="panel-header">\n'
                f'    <div class="panel-headline">{html.escape(headline)}</div>\n'
                f'    <div class="panel-date">{html.escape(date_str)}</div>\n'
                f'  </div>\n'
                f'  <div class="confidence-bar-wrap">\n'
                f'    <div class="confidence-bar">'
                f'<div class="confidence-fill" style="width:{confidence*100:.0f}%;"></div></div>\n'
                f'    <div class="confidence-label">{confidence:.0%}</div>\n'
                f'  </div>\n'
            )

            if narrative:
                html_parts.append(
                    f'  <div class="narrative">{html.escape(narrative)}</div>\n'
                )

            if key_events:
                html_parts.append('  <div class="section-label">Key Events</div>\n')
                html_parts.append('  <div class="key-events">\n')
                for event in key_events:
                    ev_date = getattr(event, "date", "")
                    ev_desc = getattr(event, "description", "")
                    ev_verified = getattr(event, "verified", False)
                    ev_url = getattr(event, "source_url", "")
                    ev_title = getattr(event, "source_title", "")
                    verified_badge = (
                        '<span class="key-event-verified">✓ verified</span>'
                        if ev_verified
                        else '<span class="key-event-unverified">unverified</span>'
                    )
                    source_link = (
                        f'<a href="{html.escape(ev_url)}" class="source-link" target="_blank">'
                        f'{html.escape(ev_title or ev_url[:40])}</a>'
                        if ev_url
                        else ""
                    )
                    html_parts.append(
                        f'    <div class="key-event">'
                        f'<span class="key-event-date">{html.escape(ev_date)}</span>'
                        f'{html.escape(ev_desc)}'
                        f'{verified_badge}'
                        f'{source_link}'
                        f'</div>\n'
                    )
                html_parts.append("  </div>\n")

            if actors:
                html_parts.append('  <div class="actors-section">\n')
                html_parts.append('    <div class="actors-header">Key Actors</div>\n')
                for actor in actors:
                    a_name = getattr(actor, "name", "")
                    a_role = getattr(actor, "role", "Peripheral")
                    role_class = {
                        "Hub": "actor-hub",
                        "Broker": "actor-broker",
                    }.get(a_role, "actor-peripheral")
                    html_parts.append(
                        f'    <span class="actor-badge {role_class}">'
                        f'{html.escape(a_name)}'
                        f'</span>\n'
                    )
                html_parts.append("  </div>\n")

            if unverified:
                items = "; ".join(html.escape(str(u)) for u in unverified[:3])
                html_parts.append(
                    f'  <div class="unverified-flags">⚠ Unverified elements: {items}</div>\n'
                )

            html_parts.append("</div>\n")

        # ── Footer ─────────────────────────────────────────────────────────────
        html_parts.append(
            f'<footer>GEOEventFusion v1.0 — {html.escape(generation_timestamp[:10] if generation_timestamp else "")}'
            f' — All confidence scores capped at 0.82 per epistemic discipline.</footer>\n'
        )
        html_parts.append("</body>\n</html>\n")

        html_content = "".join(html_parts)

        if output_path:
            from geoeventfusion.io.exporters import export_html_artifact

            output_path = Path(output_path)
            export_html_artifact(html_content, output_path)
            logger.info("Saved HTML report: %s", output_path)

        return html_content

    except Exception as exc:
        logger.error("HTML report rendering failed: %s", exc)
        return None
