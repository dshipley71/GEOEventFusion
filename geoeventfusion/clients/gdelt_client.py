"""GDELT DOC 2.0 REST API client for GEOEventFusion.

Handles all HTTP communication with the GDELT API: request construction,
rate-limit detection, exponential backoff retry, and safe JSON parsing.

No business logic lives here — this client returns raw parsed API responses.
All spike detection, actor extraction, and analysis happen in agent/analysis layers.

Known GDELT API gotchas (from CLAUDE.md):
- Responses occasionally contain HTTP header blocks instead of JSON bodies.
  Always use _safe_parse_json() — never resp.json() directly.
- Unofficial rate limit: never submit more than 2 concurrent requests.
  Always stagger submissions by >= 0.75 seconds.
- Date fields are inconsistent across modes — normalize via date_utils.
"""

from __future__ import annotations

import ast
import logging
import math
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from requests import Session
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# GDELT DOC 2.0 API base URL
_GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Known HTTP header line prefixes that GDELT occasionally returns in response bodies
_HTTP_HEADER_PREFIXES = (
    "HTTP/",
    "Date:",
    "Content-Type:",
    "Server:",
    "Transfer-Encoding:",
    "Connection:",
    "Cache-Control:",
    "Pragma:",
    "Expires:",
    "X-",
    "Vary:",
    "Set-Cookie:",
    "Access-Control:",
    "ETag:",
    "Last-Modified:",
)


# Regex for GDELT TIMESPAN strings like '30d', '2w', '3m', '48h', '15min', '1y'
_TIMESPAN_RE = re.compile(r"^(\d+)(min|h|d|w|m|y)?$", re.IGNORECASE)


def _parse_timespan_days(timespan: str) -> int:
    """Convert a GDELT TIMESPAN string to an approximate number of days.

    Handles the suffixes accepted by the GDELT DOC 2.0 API:
    'd' (days), 'w' (weeks), 'h' (hours), 'm' (months ≈ 30 d),
    'y' (years ≈ 365 d), 'min' (minutes).  A plain integer is treated as days.

    Args:
        timespan: GDELT TIMESPAN value, e.g. '30d', '2w', '3m', '48h'.

    Returns:
        Approximate number of days represented by the timespan (minimum 1).
        Returns 1 on parse failure.
    """
    match = _TIMESPAN_RE.match(timespan.strip())
    if not match:
        logger.warning("Could not parse TIMESPAN %r — assuming 1 day", timespan)
        return 1
    value = int(match.group(1))
    suffix = (match.group(2) or "d").lower()
    days = {
        "min": max(1, value // 1440),
        "h":   max(1, value // 24),
        "d":   value,
        "w":   value * 7,
        "m":   value * 30,
        "y":   value * 365,
    }.get(suffix, value)
    return max(1, days)


def _safe_parse_json(text: str) -> Optional[Any]:
    """Defensive JSON parser that handles GDELT HTTP header bleed-through.

    GDELT occasionally returns HTTP header lines prepended to the JSON body.
    This function strips those and attempts multiple parse strategies.

    Args:
        text: Raw response text from GDELT.

    Returns:
        Parsed Python object, or None on failure.
    """
    if not text or not text.strip():
        return None

    # Detect HTTP header bleed-through and extract the JSON portion
    lines = text.split("\n")
    json_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(_HTTP_HEADER_PREFIXES):
            json_start = i + 1
        elif stripped.startswith("{") or stripped.startswith("["):
            json_start = i
            break

    if json_start > 0:
        text = "\n".join(lines[json_start:]).strip()
        if not text:
            logger.warning("GDELT response body contained only HTTP headers")
            return None

    # Primary: standard json.loads
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: ast.literal_eval for near-JSON Python literals
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    logger.debug("GDELT unparseable response body: %.200s", text)
    logger.warning("Failed to parse GDELT response body (length=%d)", len(text))
    return None


class GDELTClient:
    """Client for the GDELT DOC 2.0 REST API.

    Handles request construction, staggered submission, exponential backoff,
    and defensive JSON parsing. All fetch methods return raw API response dicts.

    Args:
        max_retries: Maximum retry attempts on transient HTTP errors.
        backoff_base: Base seconds for exponential backoff (doubles per attempt).
        request_timeout: HTTP request timeout in seconds.
        stagger_seconds: Minimum seconds between successive API calls.
    """

    def __init__(
        self,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        request_timeout: int = 30,
        stagger_seconds: float = 0.75,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.request_timeout = request_timeout
        self.stagger_seconds = stagger_seconds
        self._last_request_time: float = 0.0
        self._request_lock: threading.Lock = threading.Lock()

        self._session = Session()
        adapter = HTTPAdapter(max_retries=0)   # We handle retries manually
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _build_url(self, params: Dict[str, Any]) -> str:
        """Construct a GDELT DOC API URL from a parameter dict.

        Args:
            params: GDELT query parameters.

        Returns:
            Full URL string.
        """
        return f"{_GDELT_BASE_URL}?{urlencode(params)}"

    def _enforce_stagger(self) -> None:
        """Enforce minimum delay between API submissions (thread-safe).

        The lock ensures that when two worker threads both want to make a request,
        one blocks until the other has both completed its sleep and updated
        _last_request_time. Without the lock, both threads can read the same
        elapsed value and race to fire simultaneous HTTP requests.
        """
        with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.stagger_seconds:
                time.sleep(self.stagger_seconds - elapsed)
            self._last_request_time = time.monotonic()

    def _get_with_retry(self, url: str) -> Optional[str]:
        """Execute an HTTP GET with exponential backoff retry.

        _enforce_stagger() is called at the top of every loop iteration — including
        retries — so that after sleeping for a 429 backoff the thread re-acquires
        the stagger lock before firing the next request. This prevents a thread
        that woke from a backoff sleep from colliding with a concurrent thread that
        is already mid-request.

        Args:
            url: URL to fetch.

        Returns:
            Response text on success, None on exhausted retries.
        """
        for attempt in range(self.max_retries + 1):
            self._enforce_stagger()
            try:
                resp = self._session.get(url, timeout=self.request_timeout)

                if resp.status_code == 429:
                    wait = self.backoff_base * (2 ** attempt)
                    logger.warning("GDELT rate limit (429) — backing off %.1fs", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code in (500, 502, 503, 504):
                    wait = self.backoff_base * (attempt + 1)
                    logger.warning(
                        "GDELT server error %d — retrying in %.1fs (attempt %d/%d)",
                        resp.status_code,
                        wait,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.warning("GDELT returned HTTP %d for URL: %s", resp.status_code, url)
                    return None

                return resp.text

            except requests.exceptions.Timeout:
                wait = self.backoff_base * (2 ** attempt)
                logger.warning(
                    "GDELT request timeout — retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(wait)
            except requests.exceptions.ConnectionError as exc:
                wait = self.backoff_base * (2 ** attempt)
                logger.warning(
                    "GDELT connection error: %s — retrying in %.1fs (attempt %d/%d)",
                    exc,
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(wait)
            except requests.exceptions.RequestException as exc:
                logger.error("GDELT request failed permanently: %s", exc)
                return None

        logger.error("GDELT: exhausted %d retries for URL: %s", self.max_retries, url)
        return None

    def fetch(
        self,
        query: str,
        mode: str,
        max_records: int = 250,
        sort: str = "DateDesc",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timespan: Optional[str] = None,
        timeline_smooth: int = 3,
        distribute: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a GDELT DOC 2.0 API fetch.

        Args:
            query: GDELT query string (may include operators like tone<, toneabs>, etc.).
            mode: GDELT mode (ArtList, TimelineVolInfo, TimelineVolRaw, ToneChart, etc.).
            max_records: Maximum records to return (GDELT limit: 250 for ArtList).
            sort: Sort order for ArtList mode (DateDesc, ToneAsc, ToneDesc, HybridRel).
            start_date: Start date in YYYYMMDDHHMMSS or YYYY-MM-DD format.
                Ignored when timespan is provided.
            end_date: End date in YYYYMMDDHHMMSS or YYYY-MM-DD format.
                Ignored when timespan is provided.
            timespan: GDELT TIMESPAN string (e.g. '7d', '30d', '90d', '1w', '1m').
                When set, overrides start_date and end_date — GDELT treats TIMESPAN
                and startdatetime/enddatetime as mutually exclusive. Prefer this over
                start_date/end_date for relative lookback windows; GDELT returns more
                reliable responses with TIMESPAN than with explicit date ranges.
            timeline_smooth: Smoothing window for timeline modes (1–30).
            distribute: ArtList only. When True, splits the time window into weekly
                buckets (max 13) and fetches a proportional share of articles from
                each bucket, returning up to max_records deduplicated articles spread
                uniformly across the window instead of clustering at the most recent
                end. Requires timespan or start_date+end_date; falls back to a normal
                single-call fetch if no time window is provided. Has no effect for
                non-ArtList modes.
            extra_params: Additional raw query parameters.

        Returns:
            Parsed API response dict, or None on failure.
        """
        # ── Distributed ArtList fetch ──────────────────────────────────────────
        if distribute and mode == "ArtList":
            _start_dt: Optional[datetime] = None
            _end_dt: Optional[datetime] = None

            if timespan:
                total_days = _parse_timespan_days(timespan)
                _end_dt = datetime.utcnow()
                _start_dt = _end_dt - timedelta(days=total_days)
            elif start_date and end_date:
                from geoeventfusion.utils.date_utils import gdelt_date_format

                try:
                    _start_dt = datetime.strptime(gdelt_date_format(start_date), "%Y%m%d%H%M%S")
                    _end_dt = datetime.strptime(gdelt_date_format(end_date), "%Y%m%d%H%M%S")
                except (ValueError, TypeError):
                    logger.warning(
                        "distribute=True: could not parse start/end dates — falling back to normal fetch"
                    )
            else:
                logger.warning(
                    "distribute=True: no timespan or date range provided — falling back to normal fetch"
                )

            if _start_dt is not None and _end_dt is not None:
                return self._distribute_artlist_fetch(
                    query, max_records, sort, _start_dt, _end_dt, extra_params
                )

        # ── Standard single-call fetch ─────────────────────────────────────────
        params: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "format": "json",
        }

        if mode == "ArtList":
            params["maxrecords"] = min(max_records, 250)
            params["sort"] = sort

        if mode.startswith("Timeline") or mode == "ToneChart":
            params["TIMELINESMOOTH"] = timeline_smooth

        if timespan:
            params["TIMESPAN"] = timespan
        else:
            if start_date:
                from geoeventfusion.utils.date_utils import gdelt_date_format

                params["startdatetime"] = gdelt_date_format(start_date)

            if end_date:
                from geoeventfusion.utils.date_utils import gdelt_date_format

                params["enddatetime"] = gdelt_date_format(end_date)

        if extra_params:
            params.update(extra_params)

        url = self._build_url(params)
        logger.debug("GDELT fetch: mode=%s sort=%s query=%.80s", mode, sort, query)

        raw = self._get_with_retry(url)
        if raw is None:
            return None

        parsed = _safe_parse_json(raw)
        if parsed is None:
            logger.warning("GDELT returned unparseable body for mode=%s", mode)
        return parsed

    def _distribute_artlist_fetch(
        self,
        query: str,
        max_records: int,
        sort: str,
        start_dt: datetime,
        end_dt: datetime,
        extra_params: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Fetch ArtList articles distributed evenly across a date range.

        Divides [start_dt, end_dt] into weekly buckets (max 13) and fetches a
        proportional share of articles from each bucket, returning up to
        max_records deduplicated articles spread uniformly across the window.
        Each bucket call goes through _get_with_retry, which enforces the
        0.75 s stagger required by GDELT's rate limit.

        Args:
            query: GDELT query string.
            max_records: Total article cap across all buckets.
            sort: ArtList sort mode applied within each bucket.
            start_dt: Window start (UTC).
            end_dt: Window end (UTC).
            extra_params: Additional raw parameters forwarded to every bucket call.

        Returns:
            {'articles': [...]} with up to max_records deduplicated articles,
            or None if every bucket fetch failed or returned no articles.
        """
        total_days = max(1, (end_dt - start_dt).days)
        num_buckets = max(1, min(math.ceil(total_days / 7), 13))
        records_per_bucket = math.ceil(max_records / num_buckets)
        bucket_size = (end_dt - start_dt) / num_buckets

        logger.debug(
            "GDELT distribute: %d records across %d weekly buckets (%d days total, %d per bucket)",
            max_records, num_buckets, total_days, records_per_bucket,
        )

        all_articles: List[Dict[str, Any]] = []
        seen_urls: set = set()

        for i in range(num_buckets):
            bucket_start = start_dt + bucket_size * i
            # Align the final bucket's end exactly to end_dt to avoid float drift
            bucket_end = end_dt if i == num_buckets - 1 else (start_dt + bucket_size * (i + 1))

            params: Dict[str, Any] = {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": min(records_per_bucket, 250),
                "sort": sort,
                "startdatetime": bucket_start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": bucket_end.strftime("%Y%m%d%H%M%S"),
            }
            if extra_params:
                params.update(extra_params)

            url = self._build_url(params)
            raw = self._get_with_retry(url)
            if raw is None:
                logger.warning(
                    "GDELT distribute: bucket %d/%d returned no response — skipping",
                    i + 1, num_buckets,
                )
                continue

            parsed = _safe_parse_json(raw)
            if parsed is None:
                logger.warning(
                    "GDELT distribute: bucket %d/%d returned unparseable body — skipping",
                    i + 1, num_buckets,
                )
                continue

            for article in (parsed.get("articles") or []):
                article_url = article.get("url", "")
                if article_url not in seen_urls:
                    seen_urls.add(article_url)
                    all_articles.append(article)
                    if len(all_articles) >= max_records:
                        logger.debug(
                            "GDELT distribute: reached max_records=%d at bucket %d/%d",
                            max_records, i + 1, num_buckets,
                        )
                        return {"articles": all_articles}

        if not all_articles:
            logger.warning("GDELT distribute: all %d buckets returned no articles", num_buckets)
            return None

        logger.debug(
            "GDELT distribute: collected %d/%d articles across %d buckets",
            len(all_articles), max_records, num_buckets,
        )
        return {"articles": all_articles}

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "GDELTClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
