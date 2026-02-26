"""Geographic utility functions for GEOEventFusion.

Pure geographic computations — no I/O, no external calls.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points using the Haversine formula.

    Args:
        lat1: Latitude of first point in decimal degrees.
        lon1: Longitude of first point in decimal degrees.
        lat2: Latitude of second point in decimal degrees.
        lon2: Longitude of second point in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    r = 6371.0  # Earth radius in kilometres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


# Approximate country centroids (lat, lon) for common countries in geoint analysis.
# Keyed by ISO 3166-1 alpha-2 code (upper case).
# Extend as needed; missing countries fall back to (0, 0) with a warning.
_COUNTRY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "AF": (33.93, 67.71),
    "AO": (-11.20, 17.87),
    "AZ": (40.14, 47.58),
    "BA": (43.92, 17.68),
    "BD": (23.68, 90.35),
    "BY": (53.71, 27.97),
    "CD": (-4.03, 21.76),
    "CF": (6.61, 20.94),
    "CG": (-0.23, 15.83),
    "CI": (7.54, -5.55),
    "CM": (7.37, 12.35),
    "CN": (35.86, 104.19),
    "CO": (4.57, -74.30),
    "DZ": (28.03, 1.66),
    "EG": (26.82, 30.80),
    "ER": (15.18, 39.78),
    "ET": (9.14, 40.49),
    "FR": (46.23, 2.21),
    "GB": (55.38, -3.44),
    "GE": (42.32, 43.36),
    "GH": (7.95, -1.02),
    "GM": (13.44, -15.31),
    "GN": (11.75, -15.45),
    "GW": (11.80, -15.18),
    "HT": (18.97, -72.29),
    "IL": (31.05, 34.85),
    "IN": (20.59, 78.96),
    "IQ": (33.22, 43.68),
    "IR": (32.43, 53.69),
    "JP": (36.20, 138.25),
    "KE": (-0.02, 37.91),
    "KG": (41.20, 74.77),
    "KP": (40.34, 127.51),
    "KR": (35.91, 127.77),
    "KZ": (48.02, 66.92),
    "LB": (33.85, 35.86),
    "LY": (26.34, 17.23),
    "MA": (31.79, -7.09),
    "MD": (47.41, 28.37),
    "MK": (41.61, 21.75),
    "ML": (17.57, -3.99),
    "MM": (21.92, 95.96),
    "MR": (21.01, -10.94),
    "MW": (-13.25, 34.30),
    "MX": (23.63, -102.55),
    "MZ": (-18.67, 35.53),
    "NG": (9.08, 8.68),
    "NI": (12.87, -85.21),
    "NP": (28.39, 84.12),
    "PH": (12.88, 121.77),
    "PK": (30.37, 69.35),
    "PS": (31.95, 35.23),
    "RU": (61.52, 105.32),
    "RW": (-1.94, 29.87),
    "SA": (23.89, 45.08),
    "SD": (12.86, 30.22),
    "SN": (14.50, -14.45),
    "SO": (5.15, 46.20),
    "SS": (6.88, 31.30),
    "SV": (13.79, -88.90),
    "SY": (34.80, 38.99),
    "TD": (15.45, 18.73),
    "TJ": (38.86, 71.28),
    "TM": (38.97, 59.56),
    "TN": (33.89, 9.54),
    "TR": (38.96, 35.24),
    "TZ": (-6.37, 34.89),
    "UA": (48.38, 31.17),
    "UG": (1.37, 32.29),
    "US": (37.09, -95.71),
    "UZ": (41.38, 64.59),
    "VE": (6.42, -66.59),
    "VN": (14.06, 108.28),
    "YE": (15.55, 48.52),
    "ZA": (-30.56, 22.94),
    "ZM": (-13.13, 27.85),
    "ZW": (-19.02, 29.15),
}


def country_centroid(iso2: str) -> Optional[Tuple[float, float]]:
    """Return the approximate geographic centroid for a given ISO 3166-1 alpha-2 code.

    Args:
        iso2: Two-letter ISO country code (case-insensitive).

    Returns:
        (lat, lon) tuple, or None if the country is not in the lookup table.
    """
    return _COUNTRY_CENTROIDS.get(iso2.upper())


def bbox_contains(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    """Check whether a point falls within a bounding box.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        bbox: (min_lat, min_lon, max_lat, max_lon) bounding box.

    Returns:
        True if the point is within the bounding box (inclusive).
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
