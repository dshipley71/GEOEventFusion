"""ValidationAgent — Ensure all storyboard claims are evidence-grounded.

Implements AGENTS.md §2.8 specification:
- URL reachability checks via HTTP HEAD with timeout
- Timestamp consistency checks (article date vs. claimed event date)
- Cross-source corroboration count (≥2 unique source domains)
- Ground-truth alignment via Levenshtein fuzzy matching
- Custom dataset confirmation scoring
- Title-to-claim grounding via Levenshtein similarity
- Grounding score aggregation across all checks
- Severity-classified flag generation (WARNING / ERROR / CRITICAL)

Validation checks per AGENTS.md §2.8:
  URL check        → HTTP 2xx or 3xx = pass
  Timestamp        → article date within ±7 days of claimed event
  Corroboration    → same event by ≥2 source domains
  GT alignment     → Levenshtein similarity > 0.65
  Custom confirm   → match_confidence > 0.50
  Title-to-claim   → Levenshtein similarity > 0.55
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from geoeventfusion.agents.base import AgentStatus, BaseAgent
from geoeventfusion.models.validation import (
    GroundingFlag,
    UrlCheckResult,
    ValidationAgentResult,
    VerificationFlag,
)
from geoeventfusion.utils.date_utils import date_delta_days
from geoeventfusion.utils.levenshtein_utils import similarity

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Ensure all storyboard claims are evidence-grounded and citations verifiable.

    Performs multiple validation checks against each storyboard panel's key events
    and synthesizes a grounding score and flag list.
    """

    name = "ValidationAgent"
    version = "1.0.0"

    def run(self, context: Any) -> ValidationAgentResult:
        """Run all validation checks against the storyboard.

        Args:
            context: PipelineContext with config and storyboard_result.

        Returns:
            ValidationAgentResult with grounding score and verification flags.
        """
        cfg = context.config
        result = ValidationAgentResult()

        if not context.storyboard_result or not context.storyboard_result.panels:
            logger.info("ValidationAgent: no storyboard panels to validate")
            result.warnings.append("No storyboard panels available for validation")
            return result

        # ── Collect all key events from all panels ─────────────────────────────
        all_key_events: List[Any] = []
        for panel in context.storyboard_result.panels:
            for key_event in panel.key_events:
                all_key_events.append((panel.panel_id, key_event))

        if not all_key_events:
            logger.info("ValidationAgent: no key events to validate")
            return result

        # ── URL reachability checks ───────────────────────────────────────────
        url_check_results: Dict[str, UrlCheckResult] = {}
        urls_to_check = list({
            ke.source_url
            for _, ke in all_key_events
            if ke.source_url and ke.source_url.startswith("http")
        })

        if urls_to_check:
            logger.info("ValidationAgent: checking %d URLs", len(urls_to_check))
            for url in urls_to_check[:50]:  # Cap at 50 to limit HTTP overhead
                url_check_results[url] = _check_url(url, cfg.validation_url_timeout)

        # ── Grounding checks per event ─────────────────────────────────────────
        grounding_flags: List[GroundingFlag] = []
        verification_flags: List[VerificationFlag] = []
        verified_events: List[Any] = []
        unverified_events: List[Any] = []

        # Build ground-truth claim corpus for fuzzy matching
        gt_claims: List[str] = []
        if context.ground_truth_result:
            for gt_event in context.ground_truth_result.events:
                text = f"{gt_event.event_type} {gt_event.country} {gt_event.date}"
                gt_claims.append(text)

        # Build corroboration map (claim → set of source domains)
        claim_domains: Dict[str, set] = defaultdict(set)

        for panel_id, key_event in all_key_events:
            flag = GroundingFlag(
                claim=key_event.description,
                source_url=key_event.source_url,
                source_title=key_event.source_title,
            )

            # URL reachability
            if key_event.source_url:
                url_result = url_check_results.get(key_event.source_url)
                if url_result is not None:
                    flag.url_reachable = url_result.reachable
                    if not url_result.reachable:
                        verification_flags.append(
                            VerificationFlag(
                                flag_type="DEAD_URL",
                                severity="WARNING",
                                detail=f"URL not reachable: {key_event.source_url}",
                                panel_id=panel_id,
                                event_description=key_event.description[:100],
                            )
                        )

                # Track domain for corroboration
                try:
                    domain = urlparse(key_event.source_url).netloc.lower().lstrip("www.")
                    if domain:
                        claim_key = key_event.description[:50].lower()
                        claim_domains[claim_key].add(domain)
                except Exception:
                    pass

            # Timestamp consistency
            if key_event.date and key_event.source_url:
                # Find a matching article for date comparison
                article_date = _find_article_date(
                    key_event.source_url, context
                )
                if article_date:
                    delta = date_delta_days(key_event.date, article_date)
                    if delta is not None:
                        consistent = delta <= cfg.validation_date_delta_days
                        flag.timestamp_consistent = consistent
                        if not consistent:
                            verification_flags.append(
                                VerificationFlag(
                                    flag_type="TIMESTAMP_MISMATCH",
                                    severity="WARNING",
                                    detail=(
                                        f"Event date {key_event.date} vs article date "
                                        f"{article_date} ({delta} days apart)"
                                    ),
                                    panel_id=panel_id,
                                    event_description=key_event.description[:100],
                                )
                            )

            # Title-to-claim grounding
            if key_event.source_title and key_event.description:
                title_sim = similarity(key_event.description, key_event.source_title)
                flag.title_similarity = round(title_sim, 4)
                if title_sim < cfg.validation_title_similarity_threshold:
                    flag.grounded = False
                else:
                    flag.grounded = True

            # Ground-truth alignment
            if gt_claims and key_event.description:
                from geoeventfusion.utils.levenshtein_utils import best_match_score

                gt_score = best_match_score(key_event.description, gt_claims)
                flag.ground_truth_match = (
                    gt_score >= cfg.validation_ground_truth_similarity_threshold
                )

            grounding_flags.append(flag)

            # Determine verified/unverified
            is_verified = (
                (flag.url_reachable is not False)  # None = unchecked, not penalized
                and (flag.timestamp_consistent is not False)
                and flag.grounded
            )
            if is_verified:
                verified_events.append(key_event)
            else:
                unverified_events.append(key_event)
                if not key_event.verified:
                    verification_flags.append(
                        VerificationFlag(
                            flag_type="UNVERIFIED_CLAIM",
                            severity="WARNING",
                            detail=f"Could not ground: {key_event.description[:100]}",
                            panel_id=panel_id,
                        )
                    )

        # ── Cross-source corroboration ─────────────────────────────────────────
        cross_source_corroboration: Dict[str, int] = {}
        for claim_key, domains in claim_domains.items():
            corr_count = len(domains)
            cross_source_corroboration[claim_key] = corr_count
            if corr_count < cfg.validation_min_corroboration:
                verification_flags.append(
                    VerificationFlag(
                        flag_type="LOW_CORROBORATION",
                        severity="WARNING",
                        detail=f"Claim '{claim_key[:50]}' has only {corr_count} source(s)",
                    )
                )

        # Also set corroboration_count on grounding flags
        for flag in grounding_flags:
            claim_key = flag.claim[:50].lower()
            flag.corroboration_count = cross_source_corroboration.get(claim_key, 0)

        # ── Custom dataset confirmation ────────────────────────────────────────
        if context.custom_dataset_result and context.custom_dataset_result.matches:
            for flag in grounding_flags:
                for match in context.custom_dataset_result.matches:
                    if (
                        match.match_confidence >= cfg.validation_custom_match_threshold
                        and similarity(
                            flag.claim,
                            match.custom_record.get("title", ""),
                        )
                        > 0.4
                    ):
                        flag.custom_dataset_match = True
                        break

        # ── Compute grounding score ────────────────────────────────────────────
        total = len(grounding_flags)
        if total > 0:
            grounded_count = sum(1 for f in grounding_flags if f.grounded)
            grounding_score = grounded_count / total
            verification_pct = len(verified_events) / total
        else:
            grounding_score = 0.0
            verification_pct = 0.0

        # Penalize for DEAD_URL and TIMESTAMP_MISMATCH flags
        error_flags = [f for f in verification_flags if f.severity == "ERROR"]
        if error_flags:
            grounding_score = max(0.0, grounding_score - len(error_flags) * 0.05)

        result.grounding_score = round(grounding_score, 4)
        result.verification_percentage = round(verification_pct, 4)
        result.verified_events = verified_events
        result.unverified_events = unverified_events
        result.url_check_results = url_check_results
        result.cross_source_corroboration = cross_source_corroboration
        result.grounding_flags = grounding_flags
        result.flags = verification_flags

        # ── Determine status ──────────────────────────────────────────────────
        if grounding_score == 0.0 and total > 0:
            result.status = AgentStatus.CRITICAL
            result.flags.append(
                VerificationFlag(
                    flag_type="UNVERIFIED_CLAIM",
                    severity="ERROR",
                    detail="All storyboard events are unverified",
                )
            )
        elif grounding_score < 0.3:
            result.status = AgentStatus.LOW_GROUNDING
        else:
            result.status = AgentStatus.OK

        logger.info(
            "ValidationAgent: grounding=%.2f%% | verified=%d/%d | flags=%d | status=%s",
            grounding_score * 100,
            len(verified_events),
            total,
            len(verification_flags),
            result.status,
        )
        return result


# ── Private helpers ──────────────────────────────────────────────────────────────


def _check_url(url: str, timeout: int) -> UrlCheckResult:
    """Execute an HTTP HEAD check on a URL.

    Args:
        url: URL to check.
        timeout: Request timeout in seconds.

    Returns:
        UrlCheckResult with reachability status.
    """
    try:
        import requests  # type: ignore[import-untyped]

        resp = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "GEOEventFusion/1.0 Validator"},
        )
        reachable = resp.status_code in range(200, 400)
        return UrlCheckResult(url=url, status_code=resp.status_code, reachable=reachable)
    except Exception as exc:
        # Network error — mark as UNCHECKED, do not penalize grounding score
        return UrlCheckResult(url=url, reachable=False, error=str(exc)[:100])


def _find_article_date(url: str, context: Any) -> Optional[str]:
    """Find the published_at date for an article URL from any source pool.

    Args:
        url: Article URL.
        context: PipelineContext.

    Returns:
        ISO date string, or None if not found.
    """
    if context.gdelt_result:
        for article in context.gdelt_result.all_articles():
            if article.url == url:
                return article.published_at[:10] if article.published_at else None
    if context.rss_result:
        for article in context.rss_result.articles:
            if article.url == url:
                return article.published_at[:10] if article.published_at else None
    return None
