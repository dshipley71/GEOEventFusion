"""GEOEventFusion utilities package.

All utilities are stateless pure functions with no external calls or side effects.
"""

from geoeventfusion.utils.date_utils import normalize_date_str, parse_date_range
from geoeventfusion.utils.geo_utils import haversine_km, country_centroid
from geoeventfusion.utils.levenshtein_utils import similarity, is_near_duplicate
from geoeventfusion.utils.text import (
    extract_actors_from_articles,
    is_media_actor,
    clean_html,
    normalize_text,
)

__all__ = [
    "normalize_date_str",
    "parse_date_range",
    "haversine_km",
    "country_centroid",
    "similarity",
    "is_near_duplicate",
    "extract_actors_from_articles",
    "is_media_actor",
    "clean_html",
    "normalize_text",
]
