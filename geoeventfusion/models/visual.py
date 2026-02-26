"""Visual intelligence data models for GEOEventFusion.

Defines typed schemas for visual images retrieved from GDELT's ImageCollageInfo mode,
including novelty scores, provenance chains, and VGKG content tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VisualImage:
    """A single image retrieved via GDELT ImageCollageInfo with VGKG enrichment.

    novelty_score is computed as 1.0 / (1.0 + log(1 + web_appearance_count)).
    Values near 1.0 indicate never-before-seen imagery (strong evidence of novel events).
    Values near 0.0 indicate widely-recycled stock photos.
    """

    url: str
    article_url: str = ""
    article_title: str = ""
    imagetags: List[str] = field(default_factory=list)       # Google Cloud Vision deep-learning tags
    imagewebtags: List[str] = field(default_factory=list)    # Crowdsourced reverse-image-search tags
    web_appearance_count: int = 0                             # Times this image has been seen on the web
    prior_web_urls: List[str] = field(default_factory=list)  # Up to 6 prior appearances
    exif_capture_date: Optional[str] = None                  # ISO date string from EXIF metadata
    staleness_warning: bool = False                          # True if EXIF date > 72h before article
    novelty_score: float = 0.0                               # Computed novelty metric [0.0, 1.0]
    face_tone: Optional[float] = None                        # Average emotional tone of visible faces
    face_count: Optional[int] = None                         # Number of foreground faces
    ocr_text: Optional[str] = None                          # OCR text found within the image
