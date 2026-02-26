"""4-round adversarial hypothesis debate engine for GEOEventFusion.

Generates, critiques, filters, and stress-tests geopolitical hypotheses
using the LLM backend. Each round builds on the previous to produce
a set of evidence-grounded, dimensionally diverse hypotheses.

All LLM calls go through llm_call() — never directly to anthropic/ollama.
MAX_CONFIDENCE is enforced after every round.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geoeventfusion.models.storyboard import Hypothesis

logger = logging.getLogger(__name__)


def generate_hypotheses(
    articles_negative: List[Any],
    articles_recent: List[Any],
    articles_positive: List[Any],
    query: str,
    llm_client: Optional[Any] = None,
    max_confidence: float = 0.82,
) -> List[Hypothesis]:
    """Execute the 4-round adversarial hypothesis debate.

    Round 1: Generate hypotheses from negative-toned article corpus.
    Round 2: Critique each hypothesis against recent article corpus.
    Round 3: Filter for dimensional diversity — reject duplicates.
    Round 4: Stress-test surviving hypotheses against positive-toned counterevidence.

    Args:
        articles_negative: Negatively-toned article pool (from GDELTAgent).
        articles_recent: Recent article pool for critique.
        articles_positive: Positive-toned pool for counterevidence.
        query: The pipeline query string for context grounding.
        llm_client: LLMClient instance (uses default if None).
        max_confidence: Hard cap applied after each round.

    Returns:
        List of Hypothesis objects with stress-tested claims and evidence.
    """
    if llm_client is None:
        from geoeventfusion.clients.llm_client import LLMClient

        llm_client = LLMClient()

    # Prepare article text summaries for LLM context
    neg_titles = _format_article_titles(articles_negative, max_items=20)
    recent_titles = _format_article_titles(articles_recent, max_items=20)
    pos_titles = _format_article_titles(articles_positive, max_items=10)

    # ── Round 1: Generate initial hypotheses ──────────────────────────────────
    logger.info("Hypothesis engine: Round 1 — generating initial hypotheses")
    round1_hypotheses = _round1_generate(query, neg_titles, llm_client)
    if not round1_hypotheses:
        logger.warning("Hypothesis engine: Round 1 returned no hypotheses")
        return []

    _apply_confidence_cap(round1_hypotheses, max_confidence)

    # ── Round 2: Critique hypotheses against recent coverage ──────────────────
    logger.info("Hypothesis engine: Round 2 — critiquing %d hypotheses", len(round1_hypotheses))
    round2_hypotheses = _round2_critique(round1_hypotheses, recent_titles, llm_client)
    if not round2_hypotheses:
        round2_hypotheses = round1_hypotheses
    _apply_confidence_cap(round2_hypotheses, max_confidence)

    # ── Round 3: Enforce dimensional diversity ────────────────────────────────
    logger.info("Hypothesis engine: Round 3 — diversity filtering")
    round3_hypotheses = _round3_diversify(round2_hypotheses)
    _apply_confidence_cap(round3_hypotheses, max_confidence)

    # ── Round 4: Stress-test against positive counterevidence ─────────────────
    logger.info("Hypothesis engine: Round 4 — stress-testing %d hypotheses", len(round3_hypotheses))
    final_hypotheses = _round4_stress_test(round3_hypotheses, pos_titles, llm_client)
    if not final_hypotheses:
        final_hypotheses = round3_hypotheses
    _apply_confidence_cap(final_hypotheses, max_confidence)

    logger.info("Hypothesis engine: completed — %d final hypotheses", len(final_hypotheses))
    return final_hypotheses


def _round1_generate(
    query: str, neg_titles: str, llm_client: Any
) -> List[Hypothesis]:
    """Round 1: Generate initial hypotheses from negative-toned articles."""
    system = (
        "You are a geopolitical intelligence analyst. "
        "Generate hypotheses about the situation described in the provided article titles. "
        "Each hypothesis must have a distinct analytical dimension. "
        "Return only a JSON array of hypothesis objects."
    )
    prompt = (
        f"Analysis query: {query}\n\n"
        f"Article titles (negative-toned coverage):\n{neg_titles}\n\n"
        "Generate 4-6 hypotheses. Each hypothesis must include:\n"
        "- id (integer, starting at 1)\n"
        "- dimension (string: e.g., 'Military Escalation', 'Diplomatic Posture', "
        "'Economic Impact', 'Humanitarian Consequences', 'Regional Spillover')\n"
        "- claim (string: specific, falsifiable assertion)\n"
        "- supporting_evidence (array of up to 3 article title strings that support this claim)\n"
        "- confidence (float 0.0-0.82)\n"
        "Return only a JSON array."
    )

    result = llm_client.call_json(system, prompt, max_tokens=2048, temperature=0.2)
    return _parse_hypotheses(result)


def _round2_critique(
    hypotheses: List[Hypothesis], recent_titles: str, llm_client: Any
) -> List[Hypothesis]:
    """Round 2: Critique hypotheses against recent article evidence."""
    hypotheses_json = _hypotheses_to_json(hypotheses)

    system = (
        "You are a critical intelligence reviewer. "
        "Review each hypothesis and add counter-evidence from the provided recent articles. "
        "Adjust confidence scores based on the weight of counter-evidence. "
        "Return the updated hypothesis array as JSON."
    )
    prompt = (
        f"Recent article titles for critique:\n{recent_titles}\n\n"
        f"Hypotheses to review:\n{hypotheses_json}\n\n"
        "For each hypothesis:\n"
        "- Add counter_evidence: array of article titles that contradict the claim\n"
        "- Adjust confidence down if counter-evidence is strong\n"
        "Return only a JSON array of the same hypothesis objects with updated fields."
    )

    result = llm_client.call_json(system, prompt, max_tokens=2048, temperature=0.1)
    updated = _parse_hypotheses(result)
    # Preserve originals if update fails
    return updated if updated else hypotheses


def _round3_diversify(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Round 3: Filter hypotheses to ensure dimensional diversity.

    Removes hypotheses in already-covered analytical dimensions.
    Retains the highest-confidence hypothesis per dimension.
    """
    seen_dimensions: set = set()
    diverse: List[Hypothesis] = []
    sorted_hyps = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
    for hyp in sorted_hyps:
        dim_key = hyp.dimension.lower().strip()
        if dim_key not in seen_dimensions:
            seen_dimensions.add(dim_key)
            diverse.append(hyp)
    # Re-assign sequential IDs
    for i, hyp in enumerate(diverse):
        hyp.id = i + 1
    return diverse


def _round4_stress_test(
    hypotheses: List[Hypothesis], pos_titles: str, llm_client: Any
) -> List[Hypothesis]:
    """Round 4: Stress-test surviving hypotheses against positive counterevidence."""
    hypotheses_json = _hypotheses_to_json(hypotheses)

    system = (
        "You are a stress-testing analyst. "
        "For each hypothesis, evaluate how well it holds up against positive/stabilizing evidence. "
        "Add a stress_test_result field summarizing the hypothesis's resilience. "
        "Return the updated hypothesis array as JSON."
    )
    prompt = (
        f"Positive/stabilizing article titles (counterevidence):\n{pos_titles}\n\n"
        f"Hypotheses to stress-test:\n{hypotheses_json}\n\n"
        "For each hypothesis, add:\n"
        "- stress_test_result: a 1-2 sentence assessment of hypothesis resilience\n"
        "- Adjust confidence down further if positive evidence strongly contradicts the claim\n"
        "Return only a JSON array."
    )

    result = llm_client.call_json(system, prompt, max_tokens=2048, temperature=0.1)
    updated = _parse_hypotheses(result)
    return updated if updated else hypotheses


def _parse_hypotheses(raw: Any) -> List[Hypothesis]:
    """Parse LLM JSON output into a list of Hypothesis objects."""
    if not isinstance(raw, list):
        return []
    hypotheses = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            hyp = Hypothesis(
                id=int(item.get("id", 0)),
                dimension=str(item.get("dimension", "")),
                claim=str(item.get("claim", "")),
                supporting_evidence=list(item.get("supporting_evidence", [])),
                counter_evidence=list(item.get("counter_evidence", [])),
                confidence=float(item.get("confidence", 0.0)),
                stress_test_result=str(item.get("stress_test_result", "")),
            )
            hypotheses.append(hyp)
        except (ValueError, TypeError) as exc:
            logger.debug("Failed to parse hypothesis item: %s — %s", item, exc)
    return hypotheses


def _apply_confidence_cap(hypotheses: List[Hypothesis], max_confidence: float) -> None:
    """Apply the MAX_CONFIDENCE cap to all hypotheses in place."""
    for hyp in hypotheses:
        if hyp.confidence > max_confidence:
            hyp.confidence = max_confidence


def _format_article_titles(articles: List[Any], max_items: int = 20) -> str:
    """Format article titles as a numbered list for LLM context.

    Args:
        articles: List of Article objects.
        max_items: Maximum number of titles to include.

    Returns:
        Numbered list string.
    """
    lines = []
    for i, article in enumerate(articles[:max_items], 1):
        title = getattr(article, "title", str(article))
        lines.append(f"{i}. {title}")
    return "\n".join(lines) if lines else "(no articles available)"


def _hypotheses_to_json(hypotheses: List[Hypothesis]) -> str:
    """Serialize hypotheses to a compact JSON string for LLM prompts."""
    import json
    import dataclasses

    return json.dumps([dataclasses.asdict(h) for h in hypotheses], indent=2)
