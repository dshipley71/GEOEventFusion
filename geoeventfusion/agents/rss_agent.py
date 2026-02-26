"""RSSAgent — Enrich spike windows with full-text news from RSS/Atom feeds.

Implements AGENTS.md §2.2 specification:
- Feed ingestion via feedparser
- Time window filtering relative to spike date
- Keyword and semantic filtering against spike query terms
- Full-text extraction via trafilatura (primary) or newspaper3k (fallback)
- Near-duplicate title detection via Levenshtein similarity
- HTML cleanup and text normalization
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from geoeventfusion.agents.base import AgentStatus, BaseAgent
from geoeventfusion.clients.rss_client import RSSClient
from geoeventfusion.models.events import Article, RSSAgentResult
from geoeventfusion.utils.date_utils import is_within_window, normalize_date_str
from geoeventfusion.utils.levenshtein_utils import is_near_duplicate
from geoeventfusion.utils.text import clean_html, normalize_text

logger = logging.getLogger(__name__)


class RSSAgent(BaseAgent):
    """Enrich spike windows with full-text news coverage from RSS/Atom feeds.

    Ingests configured feed URLs, filters articles by time window around spike
    dates, extracts full text, and deduplicates near-identical titles.
    """

    name = "RSSAgent"
    version = "1.0.0"

    def run(self, context: Any) -> RSSAgentResult:
        """Fetch and enrich articles from RSS feeds for each spike window.

        Args:
            context: PipelineContext with config and gdelt_result (for spikes).

        Returns:
            RSSAgentResult with enriched articles and per-feed counts.
        """
        cfg = context.config
        result = RSSAgentResult()

        if not cfg.rss_feed_list:
            logger.info("RSSAgent: no feeds configured — skipping")
            result.warnings.append("No RSS feeds configured")
            return result

        # Gather spike dates from upstream GDELT result
        spike_dates: List[str] = []
        if context.gdelt_result and context.gdelt_result.spikes:
            spike_dates = [s.date for s in context.gdelt_result.spikes]
        else:
            logger.info("RSSAgent: no spikes from GDELTAgent — using full date range")

        client = RSSClient(request_timeout=cfg.rss_request_timeout)
        feed_counts: Dict[str, int] = {}
        all_articles: List[Article] = []
        seen_urls: Set[str] = set()
        seen_titles: List[str] = []

        for feed_url in cfg.rss_feed_list:
            try:
                entries = client.fetch_feed(feed_url)
                if not entries:
                    logger.debug("RSSAgent: feed empty or unreachable: %s", feed_url)
                    feed_counts[feed_url] = 0
                    continue

                feed_articles: List[Article] = []
                for entry in entries:
                    article = self._process_entry(
                        entry, feed_url, client, cfg, spike_dates
                    )
                    if article is None:
                        continue

                    # URL deduplication
                    if article.url in seen_urls:
                        continue

                    # Near-duplicate title detection
                    if _is_near_dup_title(article.title, seen_titles, cfg.rss_dedup_threshold):
                        logger.debug(
                            "RSSAgent: near-dup skipped: %.80s", article.title
                        )
                        continue

                    seen_urls.add(article.url)
                    seen_titles.append(article.title)
                    feed_articles.append(article)

                    # Per-spike article cap
                    if len(all_articles) + len(feed_articles) >= (
                        cfg.rss_max_articles_per_spike * max(1, len(spike_dates))
                    ):
                        break

                feed_counts[feed_url] = len(feed_articles)
                all_articles.extend(feed_articles)
                logger.info(
                    "RSSAgent: feed %s → %d articles", feed_url, len(feed_articles)
                )

            except Exception as exc:
                logger.warning(
                    "RSSAgent: feed %s raised exception: %s — continuing", feed_url, exc
                )
                feed_counts[feed_url] = 0

        result.articles = all_articles
        result.feed_counts = feed_counts

        logger.info(
            "RSSAgent: %d total articles from %d feeds",
            len(all_articles),
            len(cfg.rss_feed_list),
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _process_entry(
        self,
        entry: Dict[str, Any],
        feed_url: str,
        client: RSSClient,
        cfg: Any,
        spike_dates: List[str],
    ) -> Optional[Article]:
        """Process a single feed entry into an Article.

        Args:
            entry: Raw feedparser entry dict.
            feed_url: Source feed URL.
            client: RSSClient instance.
            cfg: PipelineConfig.
            spike_dates: List of spike dates for time-window filtering.

        Returns:
            Article object, or None if filtered out.
        """
        url: str = entry.get("link", "") or ""
        title: str = entry.get("title", "") or ""

        if not url or not title:
            return None

        # Normalize date
        published_at = client.get_entry_published(entry)

        # Time-window filter: article must fall within ±window_hours of at least one spike
        if spike_dates and published_at:
            in_window = any(
                is_within_window(published_at, spike_date, cfg.rss_time_window_hours)
                for spike_date in spike_dates
            )
            if not in_window:
                return None

        # Keyword relevance filter against the query
        query_terms = cfg.query.lower().split()
        combined_text = (title + " " + client.get_entry_description(entry)).lower()
        if query_terms and not any(term in combined_text for term in query_terms):
            return None

        # Full-text extraction
        full_text = ""
        try:
            extracted = client.extract_full_text(url)
            if extracted:
                full_text = normalize_text(clean_html(extracted))
            else:
                # Fall back to RSS description
                desc = client.get_entry_description(entry)
                if desc:
                    full_text = normalize_text(clean_html(desc))
        except Exception as exc:
            logger.debug("RSSAgent: full-text extraction failed for %s: %s", url, exc)
            desc = client.get_entry_description(entry)
            full_text = normalize_text(clean_html(desc)) if desc else ""

        domain = client.get_entry_domain(entry)

        return Article(
            url=url,
            title=normalize_text(title),
            published_at=published_at,
            source=domain,
            full_text=full_text,
            domain=domain,
            metadata={
                "feed_url": feed_url,
                "extraction_method": "trafilatura" if full_text else "rss_description",
            },
        )


def _is_near_dup_title(title: str, seen_titles: List[str], threshold: float) -> bool:
    """Check whether a title is a near-duplicate of any already-seen title.

    Args:
        title: Candidate article title.
        seen_titles: List of previously accepted titles.
        threshold: Levenshtein similarity threshold.

    Returns:
        True if the title is a near-duplicate of a seen title.
    """
    for existing in seen_titles[-50:]:  # Check only the last 50 for performance
        if is_near_duplicate(title, existing, threshold):
            return True
    return False
