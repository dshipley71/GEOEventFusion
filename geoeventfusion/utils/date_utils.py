"""Date normalization utilities for GEOEventFusion.

GDELT date fields are inconsistent across modes. Always route date strings
through normalize_date_str() before storing or comparing them.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

from dateutil import parser as dateutil_parser


# Patterns covering the most common GDELT date format variants
_GDELT_DATE_PATTERNS = [
    # YYYYMMDDHHMMSS (e.g., 20240115120000)
    (re.compile(r"^(\d{4})(\d{2})(\d{2})\d{6}$"), "%Y%m%d"),
    # YYYYMMDD (e.g., 20240115)
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), "%Y%m%d"),
    # YYYY-MM-DD (ISO)
    (re.compile(r"^\d{4}-\d{2}-\d{2}"), "%Y-%m-%d"),
    # MM/DD/YYYY
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"), "%m/%d/%Y"),
]


def normalize_date_str(raw_date: str) -> str:
    """Normalize any GDELT date format variant to ISO YYYY-MM-DD.

    GDELT date fields vary by endpoint — some return YYYYMMDDHHMMSS, some YYYYMMDD,
    some ISO 8601. This function handles all known variants.

    Args:
        raw_date: Raw date string from a GDELT response field.

    Returns:
        ISO 8601 date string (YYYY-MM-DD), or the original string on parse failure.
    """
    if not raw_date:
        return ""
    raw_date = raw_date.strip()

    # Try GDELT-specific compact patterns first
    for pattern, fmt in _GDELT_DATE_PATTERNS:
        if pattern.match(raw_date):
            try:
                if len(raw_date) == 14:
                    # YYYYMMDDHHMMSS — take date portion only
                    return datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d")
                if len(raw_date) == 8 and raw_date.isdigit():
                    return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")
                return dateutil_parser.parse(raw_date).strftime("%Y-%m-%d")
            except (ValueError, OverflowError):
                continue

    # Fallback: dateutil flexible parsing
    try:
        return dateutil_parser.parse(raw_date).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        return raw_date  # Return original on total failure


def parse_date_range(days_back: int) -> Tuple[str, str]:
    """Compute (start_date, end_date) for a lookback window.

    Args:
        days_back: Number of days to look back from today.

    Returns:
        Tuple of (start_date, end_date) as ISO YYYY-MM-DD strings.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def date_delta_days(date_a: str, date_b: str) -> Optional[int]:
    """Compute the absolute difference in days between two ISO date strings.

    Args:
        date_a: First date string (YYYY-MM-DD).
        date_b: Second date string (YYYY-MM-DD).

    Returns:
        Absolute number of days between the dates, or None if either is unparseable.
    """
    try:
        da = datetime.strptime(date_a[:10], "%Y-%m-%d")
        db = datetime.strptime(date_b[:10], "%Y-%m-%d")
        return abs((da - db).days)
    except (ValueError, TypeError):
        return None


def is_within_window(date_str: str, center_date: str, window_hours: int) -> bool:
    """Check whether a date falls within ±window_hours of a center date.

    Args:
        date_str: The date to check (ISO YYYY-MM-DD or ISO datetime).
        center_date: The center date (ISO YYYY-MM-DD).
        window_hours: Number of hours on each side of center_date.

    Returns:
        True if date_str falls within the window, False otherwise.
    """
    try:
        d = dateutil_parser.parse(date_str)
        c = dateutil_parser.parse(center_date)
        delta = abs((d - c).total_seconds()) / 3600
        return delta <= window_hours
    except (ValueError, TypeError, OverflowError):
        return False


def gdelt_date_format(date_str: str) -> str:
    """Convert an ISO date string to GDELT query date format (YYYYMMDDHHMMSS).

    Args:
        date_str: ISO YYYY-MM-DD date string.

    Returns:
        GDELT-format date string (YYYYMMDD000000).
    """
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return d.strftime("%Y%m%d") + "000000"
    except (ValueError, TypeError):
        return date_str
