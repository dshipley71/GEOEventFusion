"""RSS/Atom feed client for GEOEventFusion.

Handles feed ingestion via feedparser, full-text extraction via trafilatura
(primary) or newspaper3k (fallback), and HTML cleanup.
No business logic — raw article fetching only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class RSSClient:
    """Client for fetching and parsing RSS/Atom feeds with full-text extraction.

    Args:
        request_timeout: HTTP timeout for feed and article fetches (seconds).
        user_agent: User agent string for HTTP requests.
    """

    _USER_AGENT = (
        "Mozilla/5.0 (compatible; GEOEventFusion/1.0; +https://github.com/geoeventfusion)"
    )

    def __init__(
        self,
        request_timeout: int = 15,
        user_agent: Optional[str] = None,
    ) -> None:
        self.request_timeout = request_timeout
        self.user_agent = user_agent or self._USER_AGENT

    def fetch_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse an RSS/Atom feed.

        Args:
            feed_url: URL of the RSS or Atom feed.

        Returns:
            List of raw feed entry dicts from feedparser.
            Empty list on timeout, parse error, or empty feed.
        """
        try:
            import feedparser  # type: ignore[import]

            feed = feedparser.parse(
                feed_url,
                agent=self.user_agent,
                request_headers={"User-Agent": self.user_agent},
            )
            if feed.bozo and not feed.entries:
                logger.warning("Feed parse error for %s: %s", feed_url, feed.bozo_exception)
                return []
            return list(feed.entries)
        except Exception as exc:
            logger.warning("Failed to fetch feed %s: %s", feed_url, exc)
            return []

    def extract_full_text(self, url: str) -> Optional[str]:
        """Extract full article text from a URL.

        Tries trafilatura first, falls back to newspaper3k, then returns None.

        Args:
            url: Article URL.

        Returns:
            Extracted plain text, or None if both extractors fail.
        """
        # Primary: trafilatura
        try:
            import trafilatura  # type: ignore[import]

            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                )
                if text and len(text.strip()) > 100:
                    return text.strip()
        except Exception as exc:
            logger.debug("trafilatura failed for %s: %s", url, exc)

        # Fallback: newspaper3k
        try:
            from newspaper import Article  # type: ignore[import]

            article = Article(url, request_timeout=self.request_timeout)
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 100:
                return article.text.strip()
        except Exception as exc:
            logger.debug("newspaper3k failed for %s: %s", url, exc)

        return None

    def get_entry_description(self, entry: Dict[str, Any]) -> str:
        """Extract description/summary text from a feed entry.

        Args:
            entry: Raw feedparser entry dict.

        Returns:
            Description text string (may be HTML).
        """
        for field in ("summary", "description", "content"):
            value = entry.get(field)
            if value:
                if isinstance(value, list) and value:
                    return value[0].get("value", "")
                if isinstance(value, str):
                    return value
        return ""

    def get_entry_published(self, entry: Dict[str, Any]) -> str:
        """Extract and normalize the published date from a feed entry.

        Args:
            entry: Raw feedparser entry dict.

        Returns:
            ISO date string, or empty string if unparseable.
        """
        from geoeventfusion.utils.date_utils import normalize_date_str

        for field in ("published", "updated", "created", "modified"):
            raw = entry.get(field)
            if raw:
                return normalize_date_str(str(raw))
        return ""

    def get_entry_domain(self, entry: Dict[str, Any]) -> str:
        """Extract the domain from a feed entry's link.

        Args:
            entry: Raw feedparser entry dict.

        Returns:
            Domain string (e.g., 'reuters.com').
        """
        link = entry.get("link", "")
        if not link:
            return ""
        try:
            return urlparse(link).netloc.lower().lstrip("www.")
        except Exception:
            return ""
