"""Ground-truth dataset clients for GEOEventFusion.

Provides API clients for ACLED and file-based loaders for ICEWS.
Returns raw data only — no schema normalization or business logic here.
Normalization happens in the GroundTruthAgent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ACLED API v2 base URL
_ACLED_API_BASE = "https://api.acleddata.com/acled/read"

# Request timeout for ACLED API calls
_ACLED_TIMEOUT = 30


class ACLEDClient:
    """Client for the ACLED REST API.

    Requires ACLED_API_KEY and ACLED_EMAIL environment variables.
    See https://developer.acleddata.com/ for API documentation.

    Args:
        api_key: ACLED API key (defaults to ACLED_API_KEY env var).
        email: ACLED account email (defaults to ACLED_EMAIL env var).
        request_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        request_timeout: int = _ACLED_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.getenv("ACLED_API_KEY", "")
        self.email = email or os.getenv("ACLED_EMAIL", "")
        self.request_timeout = request_timeout

        if not self.api_key or not self.email:
            logger.warning(
                "ACLED API credentials not configured. "
                "Set ACLED_API_KEY and ACLED_EMAIL environment variables."
            )

    def fetch_events(
        self,
        country: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        page_size: int = 500,
        max_pages: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch conflict events from the ACLED API.

        Args:
            country: Country name (as used in ACLED — e.g., "Yemen", "Ukraine").
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            event_types: List of ACLED event type strings (e.g., ["Battles", "Explosions"]).
            page_size: Results per API page (ACLED max: 500).
            max_pages: Maximum number of pages to fetch.

        Returns:
            List of raw ACLED event dicts from the API response.
        """
        if not self.api_key or not self.email:
            logger.warning("ACLED fetch skipped: credentials not configured")
            return []

        all_events: List[Dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            params: Dict[str, Any] = {
                "key": self.api_key,
                "email": self.email,
                "limit": page_size,
                "page": page,
                "fields": (
                    "event_id_cnty,event_date,event_type,sub_event_type,"
                    "actor1,actor2,country,region,admin1,admin2,location,"
                    "latitude,longitude,fatalities,notes,source"
                ),
            }
            if country:
                params["country"] = country
            if start_date:
                params["event_date"] = f"{start_date}|{end_date or ''}"
            if event_types:
                params["event_type"] = "|".join(event_types)

            try:
                resp = requests.get(_ACLED_API_BASE, params=params, timeout=self.request_timeout)
                if resp.status_code != 200:
                    logger.warning("ACLED API returned HTTP %d (page %d)", resp.status_code, page)
                    break

                data = resp.json()
                events = data.get("data", [])
                if not events:
                    break

                all_events.extend(events)
                logger.debug("ACLED: fetched %d events (page %d)", len(events), page)

                # ACLED count field indicates if more pages exist
                count = data.get("count", 0)
                if len(all_events) >= int(count):
                    break

            except requests.exceptions.RequestException as exc:
                logger.warning("ACLED request failed (page %d): %s", page, exc)
                break
            except (ValueError, KeyError) as exc:
                logger.warning("ACLED response parse error (page %d): %s", page, exc)
                break

        logger.info("ACLED: retrieved %d total events", len(all_events))
        return all_events


class ICEWSClient:
    """Client for ICEWS (Integrated Crisis Early Warning System) data.

    ICEWS data is available as flat CSV/TSV files. This client handles
    file-based loading and optional Dataverse API access.

    Args:
        data_path: Path to ICEWS data directory or file.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path

    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load ICEWS events from a CSV or TSV file.

        Args:
            file_path: Path to the ICEWS data file.

        Returns:
            List of raw ICEWS event dicts.
        """
        import csv

        events: List[Dict[str, Any]] = []
        delimiter = "\t" if file_path.endswith(".tsv") else ","

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    events.append(dict(row))
            logger.info("ICEWS: loaded %d events from %s", len(events), file_path)
        except (OSError, csv.Error) as exc:
            logger.warning("Failed to load ICEWS file %s: %s", file_path, exc)

        return events
