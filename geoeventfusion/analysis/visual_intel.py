"""Visual intelligence processing for GEOEventFusion.

Computes novelty scores for VGKG images, parses ImageCollageInfo API responses,
and provides image provenance chain analysis. Pure functions — no I/O.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from geoeventfusion.models.visual import VisualImage

logger = logging.getLogger(__name__)


def compute_novelty_score(web_appearance_count: int) -> float:
    """Compute the novelty score for an image based on its web appearance count.

    novelty_score = 1.0 / (1.0 + log(1 + web_appearance_count))

    A score near 1.0 indicates a never-before-seen image (strong evidence of novel events).
    A score near 0.0 indicates widely-recycled stock imagery.

    Args:
        web_appearance_count: Number of times the image has been seen on the web.

    Returns:
        Novelty score in [0.0, 1.0].
    """
    if web_appearance_count < 0:
        web_appearance_count = 0
    return 1.0 / (1.0 + math.log(1 + web_appearance_count))


def check_staleness(
    exif_capture_date: Optional[str],
    article_published_at: Optional[str],
    threshold_hours: int = 72,
) -> bool:
    """Check whether an image's EXIF capture date is stale relative to the article date.

    A staleness warning is raised when the photo was captured more than threshold_hours
    before the article publication date.

    Note: Staleness is a signal, not a disqualification. Many legitimate news images
    are file photos. Surface the flag for analyst review — do not auto-discard.

    Args:
        exif_capture_date: ISO date string from image EXIF metadata.
        article_published_at: ISO date string of the article publication.
        threshold_hours: Hours threshold for staleness (default: 72).

    Returns:
        True if the image is stale (EXIF date > threshold_hours before article).
    """
    if not exif_capture_date or not article_published_at:
        return False
    try:
        capture = datetime.fromisoformat(exif_capture_date.replace("Z", "+00:00"))
        published = datetime.fromisoformat(article_published_at.replace("Z", "+00:00"))
        if capture.tzinfo is None:
            capture = capture.replace(tzinfo=timezone.utc)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        delta_hours = (published - capture).total_seconds() / 3600
        return delta_hours > threshold_hours
    except (ValueError, TypeError, OverflowError) as exc:
        logger.debug("Staleness check failed: %s", exc)
        return False


def parse_image_collage_response(
    api_response: Optional[Dict[str, Any]],
    staleness_threshold_hours: int = 72,
) -> List[VisualImage]:
    """Parse a GDELT ImageCollageInfo API response into VisualImage objects.

    Args:
        api_response: Parsed GDELT ImageCollageInfo response dict.
        staleness_threshold_hours: Hours threshold for staleness warnings.

    Returns:
        List of VisualImage objects with computed novelty scores and staleness flags.
    """
    if not api_response or not isinstance(api_response, dict):
        return []

    images_raw = api_response.get("images", api_response.get("artlist", []))
    if not isinstance(images_raw, list):
        return []

    images: List[VisualImage] = []
    for raw in images_raw:
        if not isinstance(raw, dict):
            continue

        url = str(raw.get("url", raw.get("imageurl", "")))
        if not url:
            continue

        article_url = str(raw.get("pageurl", raw.get("article_url", "")))
        article_title = str(raw.get("title", raw.get("pagetitle", "")))

        # Deep-learning image tags
        imagetags_raw = raw.get("imagetags", raw.get("tags", []))
        if isinstance(imagetags_raw, str):
            imagetags = [t.strip() for t in imagetags_raw.split(",") if t.strip()]
        elif isinstance(imagetags_raw, list):
            imagetags = [str(t) for t in imagetags_raw]
        else:
            imagetags = []

        # Web tags from reverse image search
        imagewebtags_raw = raw.get("webtags", [])
        if isinstance(imagewebtags_raw, str):
            imagewebtags = [t.strip() for t in imagewebtags_raw.split(",") if t.strip()]
        elif isinstance(imagewebtags_raw, list):
            imagewebtags = [str(t) for t in imagewebtags_raw]
        else:
            imagewebtags = []

        # Web appearance count
        try:
            web_count = int(raw.get("imagewebcount", raw.get("webcount", 0)))
        except (ValueError, TypeError):
            web_count = 0

        # Prior web URLs (provenance chain)
        prior_urls_raw = raw.get("priorwebUrls", raw.get("prior_urls", []))
        if isinstance(prior_urls_raw, list):
            prior_urls = [str(u) for u in prior_urls_raw[:6]]
        else:
            prior_urls = []

        # EXIF capture date
        exif_date = raw.get("exifcapturedate", raw.get("capture_date"))
        exif_str: Optional[str] = str(exif_date) if exif_date else None

        # Article publication date for staleness check
        article_date = raw.get("seendate", raw.get("published_at"))
        article_date_str: Optional[str] = str(article_date) if article_date else None

        staleness = check_staleness(exif_str, article_date_str, staleness_threshold_hours)
        novelty = compute_novelty_score(web_count)

        images.append(
            VisualImage(
                url=url,
                article_url=article_url,
                article_title=article_title,
                imagetags=imagetags,
                imagewebtags=imagewebtags,
                web_appearance_count=web_count,
                prior_web_urls=prior_urls,
                exif_capture_date=exif_str,
                staleness_warning=staleness,
                novelty_score=round(novelty, 6),
                face_tone=_safe_float(raw.get("facetone")),
                face_count=_safe_int(raw.get("numfaces")),
                ocr_text=raw.get("ocrtext"),
            )
        )

    return images


def rank_images_by_novelty(images: List[VisualImage]) -> List[VisualImage]:
    """Sort images by novelty score descending (most novel first).

    Args:
        images: List of VisualImage objects.

    Returns:
        Sorted list with highest novelty scores first.
    """
    return sorted(images, key=lambda img: img.novelty_score, reverse=True)


def _safe_float(value: Any) -> Optional[float]:
    """Safe cast to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Safe cast to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
