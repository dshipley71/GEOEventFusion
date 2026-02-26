"""Text processing utilities for GEOEventFusion.

Pure functions for actor extraction, HTML cleaning, and text normalization.
All functions are stateless with no I/O or external calls.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import TYPE_CHECKING, List, Set, Tuple

if TYPE_CHECKING:
    from geoeventfusion.models.events import Article

# ── Media actor filter constants ──────────────────────────────────────────────
# Known media organization tokens — used to suppress news outlets from actor graph
_MEDIA_TOKENS: Set[str] = {
    "Reuters",
    "AP",
    "AFP",
    "BBC",
    "CNN",
    "NBC",
    "CBS",
    "Fox",
    "MSNBC",
    "Aljazeera",
    "Xinhua",
    "TASS",
    "Sputnik",
    "DW",
    "RFI",
    "VOA",
    "NPR",
    "Bloomberg",
    "Guardian",
    "Times",
    "Post",
    "Journal",
    "Tribune",
    "Herald",
    "Telegraph",
    "Independent",
    "Observer",
    "Chronicle",
    "Gazette",
    "Press",
    "Wire",
    "Media",
    "News",
    "Network",
    "Channel",
    "Radio",
    "Television",
    "Broadcasting",
    "Broadcast",
}

# Known media outlet bigrams — phrases that indicate a news organization
_MEDIA_BIGRAMS: Set[str] = {
    "New York Times",
    "Washington Post",
    "Wall Street Journal",
    "Los Angeles Times",
    "Financial Times",
    "Associated Press",
    "United Press International",
    "South China Morning Post",
    "Global Times",
    "Al Jazeera",
    "Arab News",
    "Daily Mail",
    "Daily Telegraph",
    "The Guardian",
    "The Independent",
}

# Uppercase stopwords — single capitalized tokens to suppress
_STOPWORDS_UPPER: Set[str] = {
    "The",
    "A",
    "An",
    "And",
    "Or",
    "But",
    "In",
    "On",
    "At",
    "To",
    "Of",
    "For",
    "With",
    "From",
    "By",
    "As",
    "Is",
    "Are",
    "Was",
    "Were",
    "Be",
    "Been",
    "Has",
    "Have",
    "Had",
    "Will",
    "Would",
    "Could",
    "Should",
    "May",
    "Might",
    "Must",
    "Can",
    "Its",
    "This",
    "That",
    "These",
    "Those",
    "Not",
    "No",
    "After",
    "Before",
    "During",
    "Over",
    "Under",
    "About",
    "Against",
    "Between",
    "Into",
    "Through",
    "While",
    "Than",
    "Then",
    "When",
    "Where",
    "Which",
    "Who",
    "Whose",
    "What",
    "How",
    "If",
    "Although",
    "Because",
    "Since",
    "So",
    "Yet",
    "Both",
    "Either",
    "Neither",
    "New",
    "Day",
    "Time",
    "Year",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "More",
    "Less",
    "Very",
    "Many",
    "Much",
    "Also",
    "Just",
    "Even",
    "Still",
    "Now",
    "Only",
    "First",
    "Last",
    "Next",
    "Other",
    "Some",
    "Any",
    "All",
    "Each",
    "Every",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Government",
    "Minister",
    "President",
    "Military",
    "Forces",
    "Officials",
    "Authorities",
    "People",
    "Country",
    "Nation",
    "Region",
    "City",
    "State",
    "United",
    "North",
    "South",
    "East",
    "West",
    "Central",
    "International",
    "National",
    "Global",
    "Local",
    "Official",
}

# Regex to match multi-token capitalized phrases (2–4 tokens)
_ACTOR_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
)

# Single capitalized word pattern — high noise, used conservatively
_SINGLE_CAP_PATTERN = re.compile(r"\b([A-Z][a-z]{2,})\b")


def is_media_actor(name: str) -> bool:
    """Return True if the actor name appears to be a media organization.

    Checks against known media tokens, known media bigrams, and common suffix patterns.
    This filter must run on every extracted entity before it enters the actor graph.

    Args:
        name: Extracted entity name.

    Returns:
        True if the name appears to be a media outlet or news organization.
    """
    # Direct bigram match
    if name in _MEDIA_BIGRAMS:
        return True
    # Token-level check
    tokens = name.split()
    for token in tokens:
        if token in _MEDIA_TOKENS:
            return True
    # Suffix-based detection
    if tokens and tokens[-1] in {
        "News",
        "Times",
        "Post",
        "Journal",
        "Tribune",
        "Herald",
        "Press",
        "Wire",
        "Network",
        "Channel",
        "Radio",
        "Television",
        "Media",
        "Broadcasting",
        "Online",
        "Digital",
        "TV",
        "FM",
        "AM",
        "Gazette",
        "Observer",
        "Chronicle",
        "Report",
        "Dispatch",
        "Bulletin",
    }:
        return True
    return False


def _is_stopword(name: str) -> bool:
    """Return True if name is a single stopword token."""
    tokens = name.split()
    if len(tokens) == 1 and name in _STOPWORDS_UPPER:
        return True
    return False


def extract_actors_from_articles(articles: "List[Article]") -> "List[Tuple[str, str, str]]":
    """Extract actor co-occurrence triples from article titles.

    Returns conservative multi-token capitalized phrases (2–4 tokens) that are not
    media organizations or stopwords. Single-word tokens are deliberately excluded
    to minimize false positives.

    Args:
        articles: List of Article objects with title and published_at fields.

    Returns:
        List of (actor_a, actor_b, date) tuples for co-occurring actor pairs.
    """
    triples: List[Tuple[str, str, str]] = []
    for article in articles:
        actors = _extract_actors_from_title(article.title)
        date = article.published_at[:10] if article.published_at else ""
        for i, a in enumerate(actors):
            for b in actors[i + 1 :]:
                triples.append((a, b, date))
    return triples


def _extract_actors_from_title(title: str) -> List[str]:
    """Extract actor names from a single article title.

    Args:
        title: Article title string.

    Returns:
        List of filtered actor name strings.
    """
    candidates = _ACTOR_PATTERN.findall(title)
    actors: List[str] = []
    for candidate in candidates:
        if _is_stopword(candidate):
            continue
        if is_media_actor(candidate):
            continue
        actors.append(candidate)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for actor in actors:
        if actor not in seen:
            seen.add(actor)
            result.append(actor)
    return result


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and decode HTML entities from a string.

    Args:
        raw_html: HTML-containing string.

    Returns:
        Plain text with tags removed and entities decoded.
    """
    # Decode HTML entities
    text = html.unescape(raw_html)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """Normalize Unicode text to NFC form, strip control characters and extra whitespace.

    Args:
        text: Input string.

    Returns:
        Normalized plain text string.
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to a maximum character count at a word boundary.

    Args:
        text: Input string.
        max_chars: Maximum character count.

    Returns:
        Truncated string ending at a word boundary.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
    return truncated + "..."
