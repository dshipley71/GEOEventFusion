"""LLMExtractionAgent — Convert unstructured text into structured event objects.

Implements AGENTS.md §2.5 specification:
- Backend-agnostic llm_call() interface (Anthropic or Ollama)
- Phase 3: Structured timeline generation with turning points
- Phase 4: 4-round adversarial hypothesis debate
- Phase 6: Follow-up GDELT enrichment brief generation
- JSON-only output enforcement with defensive parsing
- MAX_CONFIDENCE cap enforced after every LLM call
- Multi-event extraction from article pools
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geoeventfusion.agents.base import BaseAgent
from geoeventfusion.analysis.hypothesis_engine import generate_hypotheses
from geoeventfusion.clients.llm_client import LLMClient
from geoeventfusion.models.storyboard import (
    LLMExtractionAgentResult,
    TimelineEntry,
    TimelinePhase,
    TurningPoint,
)

logger = logging.getLogger(__name__)

# Maximum articles to include in a single extraction prompt
_MAX_PROMPT_ARTICLES = 30


class LLMExtractionAgent(BaseAgent):
    """Convert article text to structured events, timelines, and hypotheses.

    Runs three extraction phases:
    - Phase 3: Structured timeline with phase boundaries and turning points
    - Phase 4: 4-round adversarial hypothesis debate
    - Phase 6: Follow-up enrichment brief generation (optional)
    """

    name = "LLMExtractionAgent"
    version = "2.0.0"

    def run(self, context: Any) -> LLMExtractionAgentResult:
        """Execute all LLM extraction phases.

        Args:
            context: PipelineContext with config and upstream results.

        Returns:
            LLMExtractionAgentResult with timeline, hypotheses, and follow-up briefs.
        """
        cfg = context.config
        result = LLMExtractionAgentResult()

        # Build LLM client from config
        llm = LLMClient(
            backend=cfg.llm_backend,
            anthropic_model=cfg.anthropic_model,
            ollama_model=cfg.ollama_model,
            ollama_host=cfg.ollama_host,
            anthropic_api_key=cfg.anthropic_api_key,
            max_confidence=cfg.max_confidence,
        )

        # ── Gather article pools from GDELT result ──────────────────────────────
        articles_recent: List[Any] = []
        articles_negative: List[Any] = []
        articles_positive: List[Any] = []

        if context.gdelt_result:
            articles_recent = context.gdelt_result.articles_recent
            articles_negative = context.gdelt_result.articles_negative
            articles_positive = context.gdelt_result.articles_positive

        # Also include RSS articles if available
        if context.rss_result:
            articles_recent = list(articles_recent) + list(context.rss_result.articles)

        if not articles_recent and not articles_negative:
            logger.warning(
                "LLMExtractionAgent: no articles available — all phases skipped"
            )
            result.warnings.append("No articles available for LLM extraction")
            return result

        # ── Phase 3: Structured timeline ─────────────────────────────────────
        logger.info("LLMExtractionAgent: Phase 3 — structured timeline extraction")
        try:
            timeline_result = self._extract_timeline(
                articles_recent, cfg.query, llm, cfg.max_confidence
            )
            result.timeline_events = timeline_result.get("events", [])
            result.timeline_phases = timeline_result.get("phases", [])
            result.turning_points = timeline_result.get("turning_points", [])
            result.timeline_summary = timeline_result.get("summary", "")
            raw_confidence = timeline_result.get("confidence", 0.0)
            result.timeline_confidence = min(float(raw_confidence), cfg.max_confidence)
        except Exception as exc:
            logger.warning("LLMExtractionAgent: Phase 3 failed: %s", exc)
            result.warnings.append(f"Timeline extraction failed: {exc}")

        # ── Phase 4: Adversarial hypothesis debate ────────────────────────────
        logger.info("LLMExtractionAgent: Phase 4 — adversarial hypothesis debate")
        try:
            result.hypotheses = generate_hypotheses(
                articles_negative=articles_negative,
                articles_recent=articles_recent,
                articles_positive=articles_positive,
                query=cfg.query,
                llm_client=llm,
                max_confidence=cfg.max_confidence,
            )
        except Exception as exc:
            logger.warning("LLMExtractionAgent: Phase 4 failed: %s", exc)
            result.warnings.append(f"Hypothesis generation failed: {exc}")

        # ── Phase 6: Follow-up enrichment briefs ─────────────────────────────
        if result.timeline_phases or result.hypotheses:
            logger.info("LLMExtractionAgent: Phase 6 — follow-up enrichment briefs")
            try:
                result.followup_briefs = self._generate_followup_briefs(
                    result, cfg.query, llm
                )
            except Exception as exc:
                logger.warning("LLMExtractionAgent: Phase 6 failed: %s", exc)
                result.warnings.append(f"Follow-up brief generation failed: {exc}")

        logger.info(
            "LLMExtractionAgent: %d timeline events | %d phases | %d hypotheses | "
            "%d followup briefs",
            len(result.timeline_events),
            len(result.timeline_phases),
            len(result.hypotheses),
            len(result.followup_briefs),
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _extract_timeline(
        self,
        articles: List[Any],
        query: str,
        llm: LLMClient,
        max_confidence: float,
    ) -> Dict[str, Any]:
        """Extract structured timeline from article corpus.

        Args:
            articles: List of Article objects.
            query: Pipeline query string.
            llm: LLMClient instance.
            max_confidence: Hard confidence cap.

        Returns:
            Dict with events, phases, turning_points, summary, confidence fields.
        """
        article_text = _format_articles_for_prompt(articles, max_items=_MAX_PROMPT_ARTICLES)

        system = (
            "You are a geopolitical intelligence analyst specializing in timeline construction. "
            "Extract a structured event timeline from the provided news articles. "
            "Return only a JSON object with keys: events, phases, turning_points, summary, confidence."
        )
        prompt = (
            f"Analysis query: {query}\n\n"
            f"News articles:\n{article_text}\n\n"
            "Extract a structured timeline. Return a JSON object with:\n"
            "- events: array of event objects, each with: "
            "event_type, datetime (YYYY-MM-DD), country, lat, lon, actors (array), "
            "summary, confidence (0.0-0.82), source_url, source_title\n"
            "- phases: array of phase objects, each with: "
            "label, date_range ({start, end}), description, key_events (array of strings), "
            "tone_shift, actor_changes (array of strings)\n"
            "- turning_points: array with: date, description, evidence_title, evidence_url\n"
            "- summary: 2-3 sentence overall summary\n"
            "- confidence: float 0.0-0.82\n"
            "Return only a JSON object. Use 'return only a JSON array' for events. "
            "Ensure multi-event extraction — return all events found."
        )

        raw = llm.call_json(system, prompt, max_tokens=cfg_max_tokens(4096), temperature=0.1)
        if not isinstance(raw, dict):
            logger.warning("LLMExtractionAgent: timeline returned non-dict: %s", type(raw))
            return {}

        # Apply MAX_CONFIDENCE cap
        raw = llm.enforce_confidence_cap(raw)

        # Parse events
        events: List[TimelineEntry] = []
        for item in raw.get("events", []):
            if not isinstance(item, dict):
                continue
            try:
                entry = TimelineEntry(
                    event_type=str(item.get("event_type", "OTHER")),
                    datetime=str(item.get("datetime", "")),
                    country=str(item.get("country", "")),
                    lat=_safe_float(item.get("lat")),
                    lon=_safe_float(item.get("lon")),
                    actors=list(item.get("actors", [])),
                    summary=str(item.get("summary", "")),
                    confidence=min(float(item.get("confidence", 0.0) or 0.0), max_confidence),
                    source_url=str(item.get("source_url", "")),
                    source_title=str(item.get("source_title", "")),
                )
                events.append(entry)
            except (ValueError, TypeError) as exc:
                logger.debug("LLMExtractionAgent: failed to parse event: %s", exc)

        # Parse phases
        phases: List[TimelinePhase] = []
        for item in raw.get("phases", []):
            if not isinstance(item, dict):
                continue
            try:
                phase = TimelinePhase(
                    label=str(item.get("label", "")),
                    date_range=dict(item.get("date_range", {})),
                    description=str(item.get("description", "")),
                    key_events=list(item.get("key_events", [])),
                    tone_shift=str(item.get("tone_shift", "")),
                    actor_changes=list(item.get("actor_changes", [])),
                )
                phases.append(phase)
            except (ValueError, TypeError) as exc:
                logger.debug("LLMExtractionAgent: failed to parse phase: %s", exc)

        # Parse turning points
        turning_points: List[TurningPoint] = []
        for item in raw.get("turning_points", []):
            if not isinstance(item, dict):
                continue
            try:
                tp = TurningPoint(
                    date=str(item.get("date", "")),
                    description=str(item.get("description", "")),
                    evidence_title=str(item.get("evidence_title", "")),
                    evidence_url=str(item.get("evidence_url", "")),
                )
                turning_points.append(tp)
            except (ValueError, TypeError) as exc:
                logger.debug("LLMExtractionAgent: failed to parse turning point: %s", exc)

        return {
            "events": events,
            "phases": phases,
            "turning_points": turning_points,
            "summary": str(raw.get("summary", "")),
            "confidence": float(raw.get("confidence", 0.0) or 0.0),
        }

    def _generate_followup_briefs(
        self,
        result: LLMExtractionAgentResult,
        query: str,
        llm: LLMClient,
    ) -> List[Dict[str, Any]]:
        """Generate follow-up GDELT enrichment briefs from storyboard recommendations.

        Each brief includes a recommended near-operator GDELT query for further
        enrichment of a specific angle identified in the timeline or hypotheses.

        Args:
            result: Current LLMExtractionAgentResult with phases and hypotheses.
            query: Original pipeline query.
            llm: LLMClient instance.

        Returns:
            List of follow-up brief dicts with query and rationale.
        """
        # Build context from phases and hypotheses
        phase_labels = [p.label for p in result.timeline_phases[:3]]
        hyp_claims = [h.claim for h in result.hypotheses[:3]]

        if not phase_labels and not hyp_claims:
            return []

        system = (
            "You are a GDELT query specialist generating follow-up intelligence briefs. "
            "Convert analysis findings into targeted GDELT search queries for further enrichment. "
            "Return only a JSON array of brief objects."
        )
        prompt = (
            f"Original query: {query}\n\n"
            f"Timeline phases identified: {', '.join(phase_labels)}\n"
            f"Key hypotheses: {'; '.join(hyp_claims)}\n\n"
            "Generate 3-5 follow-up enrichment briefs. Each brief must include:\n"
            "- angle: what specific aspect to investigate further\n"
            "- gdelt_query: a targeted GDELT query string (may use near<N>: or theme: operators)\n"
            "- rationale: 1 sentence explaining why this angle matters\n"
            "Return only a JSON array."
        )

        raw = llm.call_json(system, prompt, max_tokens=1024, temperature=0.1)
        if not isinstance(raw, list):
            return []

        briefs: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and item.get("gdelt_query"):
                briefs.append({
                    "angle": str(item.get("angle", "")),
                    "gdelt_query": str(item.get("gdelt_query", "")),
                    "rationale": str(item.get("rationale", "")),
                })
        return briefs


# ── Module-level helpers ────────────────────────────────────────────────────────


def _format_articles_for_prompt(articles: List[Any], max_items: int = 30) -> str:
    """Format article titles for LLM context.

    Args:
        articles: List of Article objects.
        max_items: Maximum number of articles to include.

    Returns:
        Numbered list string.
    """
    lines = []
    for i, article in enumerate(articles[:max_items], 1):
        title = getattr(article, "title", str(article))
        url = getattr(article, "url", "")
        date = getattr(article, "published_at", "")[:10] if getattr(article, "published_at", "") else ""
        line = f"{i}. [{date}] {title}"
        if url:
            line += f" <{url}>"
        lines.append(line)
    return "\n".join(lines) if lines else "(no articles available)"


def cfg_max_tokens(default: int) -> int:
    """Return the specified token count (minimum 256 enforced by LLMClient).

    Args:
        default: Desired max_tokens value.

    Returns:
        The value passed through (LLMClient enforces minimum internally).
    """
    return default


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float.

    Args:
        value: Input value.

    Returns:
        Float, or None on failure.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
