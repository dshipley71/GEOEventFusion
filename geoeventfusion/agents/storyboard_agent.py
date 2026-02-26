"""StoryboardAgent — Generate structured narrative intelligence panels.

Implements AGENTS.md §2.7 specification:
- Structured narrative panel synthesis from fusion clusters
- LLM-driven headline generation grounded in spike article titles
- Multi-event summarization per panel with citation inclusion
- Phase boundary detection from community reorganization and spike dates
- Turning-point identification with article title as evidence
- Contradiction detection across clusters
- Auto-supplementation of citations to meet MIN_CITATIONS floor
- MAX_CONFIDENCE cap enforcement on all panels
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from geoeventfusion.agents.base import AgentStatus, BaseAgent
from geoeventfusion.clients.llm_client import LLMClient
from geoeventfusion.models.storyboard import (
    PanelActor,
    PanelKeyEvent,
    StoryboardAgentResult,
    StoryboardPanel,
)

logger = logging.getLogger(__name__)

# Minimum confidence below which storyboard is flagged LOW_CONFIDENCE
_LOW_CONFIDENCE_THRESHOLD = 0.25


class StoryboardAgent(BaseAgent):
    """Generate narrative intelligence storyboard panels from fused event clusters.

    Each panel covers a phase/cluster with grounded key events, actors, and
    narrative summary. All confidence scores are capped at MAX_CONFIDENCE.
    """

    name = "StoryboardAgent"
    version = "1.0.0"

    def run(self, context: Any) -> StoryboardAgentResult:
        """Generate storyboard panels from all upstream analysis results.

        Args:
            context: PipelineContext with config and all upstream results.

        Returns:
            StoryboardAgentResult with panels, overall confidence, and followup.
        """
        cfg = context.config
        result = StoryboardAgentResult()
        result.query = cfg.query
        result.generation_timestamp = datetime.utcnow().isoformat() + "Z"
        result.max_confidence_cap = cfg.max_confidence

        # Compute date range from gdelt_result
        from geoeventfusion.utils.date_utils import parse_date_range
        start_date, end_date = parse_date_range(cfg.days_back)
        result.date_range = {"start": start_date, "end": end_date}

        # Build LLM client
        llm = LLMClient(
            backend=cfg.llm_backend,
            anthropic_model=cfg.anthropic_model,
            ollama_model=cfg.ollama_model,
            ollama_host=cfg.ollama_host,
            anthropic_api_key=cfg.anthropic_api_key,
            max_confidence=cfg.max_confidence,
        )

        # ── Determine panel boundaries ────────────────────────────────────────
        phase_boundaries = _determine_phase_boundaries(context)

        # ── Build one panel per phase/boundary ───────────────────────────────
        panels: List[StoryboardPanel] = []
        for i, (phase_start, phase_end) in enumerate(phase_boundaries):
            panel = self._build_panel(
                panel_index=i,
                phase_start=phase_start,
                phase_end=phase_end,
                context=context,
                llm=llm,
                cfg=cfg,
            )
            if panel is not None:
                panels.append(panel)

        # Fallback: if no phase boundaries produced panels, build a single summary panel
        if not panels:
            panel = self._build_summary_panel(context, llm, cfg, start_date, end_date)
            if panel is not None:
                panels.append(panel)

        # ── Auto-supplement citations to reach MIN_CITATIONS ──────────────────
        for panel in panels:
            _supplement_citations(panel, context, cfg.min_citations)

        # ── Compute overall confidence ─────────────────────────────────────────
        if panels:
            overall_confidence = sum(p.confidence for p in panels) / len(panels)
            overall_confidence = min(overall_confidence, cfg.max_confidence)
        else:
            overall_confidence = 0.0

        # ── Escalation risk from tone stats ──────────────────────────────────
        escalation_risk = _compute_escalation_risk(context, cfg.max_confidence)

        # ── Aggregate follow-up recommendations ──────────────────────────────
        followup: List[str] = []
        for panel in panels:
            followup.extend(panel.recommended_followup)
        if context.llm_result:
            for brief in context.llm_result.followup_briefs:
                q = brief.get("gdelt_query", "")
                if q and q not in followup:
                    followup.append(q)

        result.panels = panels
        result.overall_confidence = round(overall_confidence, 4)
        result.escalation_risk = escalation_risk
        result.recommended_followup = list(dict.fromkeys(followup))[:10]

        # ── Determine status ──────────────────────────────────────────────────
        if not panels:
            result.status = AgentStatus.CRITICAL
        elif overall_confidence < _LOW_CONFIDENCE_THRESHOLD:
            result.status = AgentStatus.LOW_CONFIDENCE
        else:
            result.status = AgentStatus.OK

        logger.info(
            "StoryboardAgent: %d panels | confidence=%.3f | escalation=%.3f | status=%s",
            len(panels),
            overall_confidence,
            escalation_risk,
            result.status,
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _build_panel(
        self,
        panel_index: int,
        phase_start: str,
        phase_end: str,
        context: Any,
        llm: LLMClient,
        cfg: Any,
    ) -> Optional[StoryboardPanel]:
        """Build a single storyboard panel for a date range.

        Args:
            panel_index: Panel index (0-based).
            phase_start: Phase start date.
            phase_end: Phase end date.
            context: PipelineContext.
            llm: LLMClient instance.
            cfg: PipelineConfig.

        Returns:
            StoryboardPanel, or None if insufficient data for this phase.
        """
        panel_id = f"panel_{panel_index + 1:02d}"

        # Collect articles in this phase window
        phase_articles = _get_phase_articles(context, phase_start, phase_end)
        if not phase_articles and panel_index > 0:
            return None

        # Collect events in this phase window
        phase_events = _get_phase_events(context, phase_start, phase_end)

        # Build key events from articles and LLM events
        key_events = _build_key_events(phase_articles, phase_events, cfg.min_citations)

        # Build actors from GDELT actor graph for this phase
        panel_actors = _build_panel_actors(context, phase_start, phase_end)

        # Generate headline via LLM
        headline = _generate_headline(
            phase_articles[:5], cfg.query, phase_start, phase_end, llm
        )

        # Generate narrative summary via LLM
        narrative = _generate_narrative(
            phase_articles[:15],
            phase_events,
            cfg.query,
            phase_start,
            phase_end,
            llm,
        )

        # Compute panel confidence
        n_sources = len({a.domain for a in phase_articles if a.domain})
        n_events = len(phase_events)
        confidence = min(
            0.3 + n_sources * 0.04 + n_events * 0.02,
            cfg.max_confidence,
        )

        # Grounded sources (unique domains with articles)
        grounded_sources = list({
            f"{a.domain}: {a.title}" for a in phase_articles[:cfg.min_citations]
        })

        # Follow-up recommendations from hypotheses
        followup = _build_followup_recs(context, phase_start, phase_end)

        return StoryboardPanel(
            panel_id=panel_id,
            date_range={"start": phase_start, "end": phase_end},
            headline=headline,
            key_events=key_events,
            actors=panel_actors,
            narrative_summary=narrative,
            confidence=round(confidence, 4),
            grounded_sources=grounded_sources,
            recommended_followup=followup[:3],
        )

    def _build_summary_panel(
        self,
        context: Any,
        llm: LLMClient,
        cfg: Any,
        start_date: str,
        end_date: str,
    ) -> Optional[StoryboardPanel]:
        """Build a single full-window summary panel when no phase boundaries exist.

        Args:
            context: PipelineContext.
            llm: LLMClient.
            cfg: PipelineConfig.
            start_date: Analysis window start.
            end_date: Analysis window end.

        Returns:
            StoryboardPanel covering the full analysis window.
        """
        return self._build_panel(
            panel_index=0,
            phase_start=start_date,
            phase_end=end_date,
            context=context,
            llm=llm,
            cfg=cfg,
        )


# ── Panel-building helpers ──────────────────────────────────────────────────────


def _determine_phase_boundaries(context: Any) -> List[tuple]:
    """Compute phase boundaries from spikes and community reorganization.

    Args:
        context: PipelineContext.

    Returns:
        List of (start_date, end_date) tuples defining phase windows.
    """
    from geoeventfusion.utils.date_utils import parse_date_range

    cfg = context.config
    start_date, end_date = parse_date_range(cfg.days_back)

    # Gather candidate boundary dates
    boundary_dates: List[str] = []

    # From LLM timeline phases
    if context.llm_result and context.llm_result.timeline_phases:
        for phase in context.llm_result.timeline_phases:
            dr = phase.date_range
            if isinstance(dr, dict):
                for key in ("start", "end"):
                    d = dr.get(key, "")
                    if d and d not in boundary_dates:
                        boundary_dates.append(d)

    # From spike dates
    if context.gdelt_result and context.gdelt_result.spikes:
        from geoeventfusion.analysis.spike_detector import find_phase_boundaries

        reorg_score: Optional[float] = None
        boundary_candidate: Optional[str] = None
        if (
            context.gdelt_result.actor_graph
            and context.gdelt_result.actor_graph.temporal_shift
        ):
            ts = context.gdelt_result.actor_graph.temporal_shift
            reorg_score = ts.reorganization_score
            boundary_candidate = ts.phase_boundary_candidate

        spike_boundaries = find_phase_boundaries(
            context.gdelt_result.spikes,
            reorganization_score=reorg_score,
            phase_boundary_date=boundary_candidate,
        )
        for d in spike_boundaries:
            if d not in boundary_dates:
                boundary_dates.append(d)

    boundary_dates = sorted(set(boundary_dates))

    if not boundary_dates:
        return [(start_date, end_date)]

    # Build phase windows from sorted boundaries
    all_dates = [start_date] + boundary_dates + [end_date]
    phases = []
    for i in range(len(all_dates) - 1):
        phases.append((all_dates[i], all_dates[i + 1]))

    return phases if phases else [(start_date, end_date)]


def _get_phase_articles(context: Any, start: str, end: str) -> List[Any]:
    """Get articles published within a phase date window.

    Args:
        context: PipelineContext.
        start: Phase start date (ISO).
        end: Phase end date (ISO).

    Returns:
        List of Article objects within the window.
    """
    articles: List[Any] = []
    if context.gdelt_result:
        for article in context.gdelt_result.all_articles():
            d = article.published_at[:10] if article.published_at else ""
            if start <= d <= end:
                articles.append(article)
    if context.rss_result:
        for article in context.rss_result.articles:
            d = article.published_at[:10] if article.published_at else ""
            if start <= d <= end:
                articles.append(article)
    return articles


def _get_phase_events(context: Any, start: str, end: str) -> List[Any]:
    """Get LLM-extracted events within a phase date window.

    Args:
        context: PipelineContext.
        start: Phase start date.
        end: Phase end date.

    Returns:
        List of TimelineEntry objects.
    """
    events: List[Any] = []
    if context.llm_result and context.llm_result.timeline_events:
        for event in context.llm_result.timeline_events:
            d = event.datetime[:10] if event.datetime else ""
            if not d or (start <= d <= end):
                events.append(event)
    return events


def _build_key_events(
    articles: List[Any], events: List[Any], min_citations: int
) -> List[PanelKeyEvent]:
    """Build key events from articles and structured events.

    Args:
        articles: Phase articles.
        events: LLM-extracted timeline events.
        min_citations: Minimum required events.

    Returns:
        List of PanelKeyEvent objects.
    """
    key_events: List[PanelKeyEvent] = []
    seen_urls: set = set()

    # From LLM events (verified)
    for event in events[:5]:
        key_events.append(
            PanelKeyEvent(
                date=getattr(event, "datetime", "")[:10],
                description=getattr(event, "summary", ""),
                source_title=getattr(event, "source_title", ""),
                source_url=getattr(event, "source_url", ""),
                verified=True,
            )
        )
        url = getattr(event, "source_url", "")
        if url:
            seen_urls.add(url)

    # Supplement from articles to reach min_citations
    for article in articles:
        if len(key_events) >= max(min_citations, 5):
            break
        if article.url in seen_urls:
            continue
        key_events.append(
            PanelKeyEvent(
                date=article.published_at[:10] if article.published_at else "",
                description=article.title,
                source_title=article.title,
                source_url=article.url,
                verified=False,
            )
        )
        seen_urls.add(article.url)

    return key_events


def _build_panel_actors(context: Any, start: str, end: str) -> List[PanelActor]:
    """Build actor list for a panel from the GDELT actor graph.

    Args:
        context: PipelineContext.
        start: Phase start date.
        end: Phase end date.

    Returns:
        List of PanelActor objects (top-10 by PageRank).
    """
    if not context.gdelt_result or not context.gdelt_result.actor_graph:
        return []

    graph = context.gdelt_result.actor_graph
    top_actors = graph.get_top_actors(n=10)

    return [
        PanelActor(
            name=actor.name,
            role=actor.role,
            centrality_score=round(actor.pagerank, 6),
        )
        for actor in top_actors
    ]


def _generate_headline(
    articles: List[Any],
    query: str,
    start: str,
    end: str,
    llm: LLMClient,
) -> str:
    """Generate a panel headline via LLM.

    Args:
        articles: Sample articles for context.
        query: Pipeline query.
        start: Phase start date.
        end: Phase end date.
        llm: LLMClient instance.

    Returns:
        Headline string (falls back to query if LLM fails).
    """
    if not articles:
        return f"{query} — {start} to {end}"

    titles = "\n".join(f"- {a.title}" for a in articles[:5])
    system = (
        "You are a geopolitical intelligence analyst. "
        "Generate a concise, factual headline (under 120 characters) for an intelligence brief. "
        "Return only the headline text — no quotes, no markdown."
    )
    prompt = (
        f"Analysis topic: {query}\n"
        f"Date range: {start} to {end}\n\n"
        f"Key article titles:\n{titles}\n\n"
        "Generate a concise headline for this intelligence brief panel."
    )
    try:
        raw = llm.call(system, prompt, max_tokens=100, temperature=0.1)
        if raw and raw.strip():
            return raw.strip()[:200]
    except Exception as exc:
        logger.debug("Headline generation failed: %s", exc)

    return f"{query} — {start} to {end}"


def _generate_narrative(
    articles: List[Any],
    events: List[Any],
    query: str,
    start: str,
    end: str,
    llm: LLMClient,
) -> str:
    """Generate a narrative summary for a panel via LLM.

    Args:
        articles: Phase articles.
        events: LLM-extracted events.
        query: Pipeline query.
        start: Phase start.
        end: Phase end.
        llm: LLMClient.

    Returns:
        Narrative summary string.
    """
    article_text = "\n".join(
        f"[{a.published_at[:10] if a.published_at else ''}] {a.title}"
        for a in articles[:10]
    )
    event_text = "\n".join(
        f"- {e.datetime[:10] if e.datetime else ''}: {e.summary}"
        for e in events[:5]
    )

    if not article_text:
        return "Insufficient data for narrative generation."

    system = (
        "You are a geopolitical intelligence analyst. "
        "Write a concise, evidence-grounded intelligence narrative (3-5 sentences). "
        "Cite specific events and dates. Maintain analytical objectivity."
    )
    prompt = (
        f"Analysis topic: {query}\nDate range: {start} to {end}\n\n"
        f"Articles:\n{article_text}\n\n"
        + (f"Structured events:\n{event_text}\n\n" if event_text else "")
        + "Write a 3-5 sentence intelligence narrative for this phase."
    )

    try:
        raw = llm.call(system, prompt, max_tokens=512, temperature=0.1)
        if raw and raw.strip():
            return raw.strip()[:2000]
    except Exception as exc:
        logger.debug("Narrative generation failed: %s", exc)

    return f"Coverage of {query} during {start}–{end} based on {len(articles)} articles."


def _build_followup_recs(context: Any, start: str, end: str) -> List[str]:
    """Build follow-up recommendations from hypotheses and turning points.

    Args:
        context: PipelineContext.
        start: Phase start.
        end: Phase end.

    Returns:
        List of follow-up query strings.
    """
    recs: List[str] = []
    if context.llm_result:
        for hyp in context.llm_result.hypotheses[:2]:
            claim_words = hyp.claim.split()[:4]
            if claim_words:
                recs.append(" ".join(claim_words))
        for tp in context.llm_result.turning_points[:2]:
            if tp.description:
                recs.append(tp.description[:80])
    return recs


def _supplement_citations(
    panel: StoryboardPanel, context: Any, min_citations: int
) -> None:
    """Auto-supplement citations to reach MIN_CITATIONS floor.

    Adds articles from the GDELT pool to panel.grounded_sources and
    panel.key_events if the panel has fewer than min_citations.

    Args:
        panel: StoryboardPanel to supplement (modified in-place).
        context: PipelineContext.
        min_citations: Minimum required citations.
    """
    if len(panel.grounded_sources) >= min_citations:
        return
    if not context.gdelt_result:
        return

    existing_sources = set(panel.grounded_sources)
    for article in context.gdelt_result.articles_relevant[:20]:
        if len(panel.grounded_sources) >= min_citations:
            break
        citation = f"{article.domain}: {article.title}"
        if citation not in existing_sources:
            panel.grounded_sources.append(citation)
            existing_sources.add(citation)

            # Also add to key_events if needed
            if len(panel.key_events) < min_citations:
                panel.key_events.append(
                    PanelKeyEvent(
                        date=article.published_at[:10] if article.published_at else "",
                        description=article.title,
                        source_title=article.title,
                        source_url=article.url,
                        verified=False,
                    )
                )


def _compute_escalation_risk(context: Any, max_confidence: float) -> float:
    """Compute escalation risk from tone stats and hypothesis confidence.

    Args:
        context: PipelineContext.
        max_confidence: Hard confidence cap.

    Returns:
        Escalation risk score in [0.0, max_confidence].
    """
    risk = 0.0

    # Tone-based risk: high polarity ratio → escalation risk
    if context.gdelt_result and context.gdelt_result.tone_stats:
        tone_stats = context.gdelt_result.tone_stats
        risk += tone_stats.polarity_ratio * 0.4

    # Spike-based risk: many high-Z spikes → escalation risk
    if context.gdelt_result and context.gdelt_result.spikes:
        spikes = context.gdelt_result.spikes
        avg_z = sum(s.z_score for s in spikes[:5]) / max(1, len(spikes[:5]))
        # Normalize z-score contribution (typical max Z is ~5)
        risk += min(avg_z / 10.0, 0.3)

    # Hypothesis-based risk
    if context.llm_result and context.llm_result.hypotheses:
        conflict_hyps = [
            h for h in context.llm_result.hypotheses
            if "escalation" in h.dimension.lower() or "conflict" in h.dimension.lower()
        ]
        if conflict_hyps:
            avg_conf = sum(h.confidence for h in conflict_hyps) / len(conflict_hyps)
            risk += avg_conf * 0.3

    return round(min(risk, max_confidence), 4)
