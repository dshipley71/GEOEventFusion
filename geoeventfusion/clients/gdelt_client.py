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
import re
import time
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
        """Enforce minimum delay between API submissions."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.stagger_seconds:
            time.sleep(self.stagger_seconds - elapsed)
        self._last_request_time = time.monotonic()

    def _get_with_retry(self, url: str) -> Optional[str]:
        """Execute an HTTP GET with exponential backoff retry.

        Args:
            url: URL to fetch.

        Returns:
            Response text on success, None on exhausted retries.
        """
        self._enforce_stagger()

        for attempt in range(self.max_retries + 1):
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
        timeline_smooth: int = 3,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a GDELT DOC 2.0 API fetch.

        Args:
            query: GDELT query string (may include operators like tone<, toneabs>, etc.).
            mode: GDELT mode (ArtList, TimelineVolInfo, TimelineVolRaw, ToneChart, etc.).
            max_records: Maximum records to return (GDELT limit: 250 for ArtList).
            sort: Sort order for ArtList mode (DateDesc, ToneAsc, ToneDesc, HybridRel).
            start_date: Start date in YYYYMMDDHHMMSS or YYYY-MM-DD format.
            end_date: End date in YYYYMMDDHHMMSS or YYYY-MM-DD format.
            timeline_smooth: Smoothing window for timeline modes (1–30).
            extra_params: Additional raw query parameters.

        Returns:
            Parsed API response dict, or None on failure.
        """
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

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "GDELTClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
