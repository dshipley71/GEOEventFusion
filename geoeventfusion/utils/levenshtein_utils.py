"""Levenshtein similarity utilities for GEOEventFusion.

Pure string comparison functions used for deduplication, validation, and
ground-truth alignment. No I/O or external calls.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from Levenshtein import ratio as _lev_ratio
    from Levenshtein import distance as _lev_distance

    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False


def _simple_ratio(s1: str, s2: str) -> float:
    """Pure-Python Levenshtein ratio fallback when the C extension is unavailable.

    Uses dynamic programming. O(m*n) time and space.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    dist = dp[n]
    return 1.0 - dist / max(m, n)


def similarity(s1: str, s2: str) -> float:
    """Compute the normalized Levenshtein similarity ratio between two strings.

    Returns a value in [0.0, 1.0] where 1.0 means identical strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity ratio in [0.0, 1.0].
    """
    if not s1 and not s2:
        return 1.0
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if _HAS_LEVENSHTEIN:
        return float(_lev_ratio(s1, s2))
    return _simple_ratio(s1, s2)


def distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance (number of single-character insertions, deletions, or substitutions).
    """
    if _HAS_LEVENSHTEIN:
        return int(_lev_distance(s1.lower(), s2.lower()))
    # Fallback: derive distance from ratio
    max_len = max(len(s1), len(s2))
    return round((1.0 - _simple_ratio(s1.lower(), s2.lower())) * max_len)


def is_near_duplicate(s1: str, s2: str, threshold: float = 0.85) -> bool:
    """Check whether two strings are near-duplicates.

    Used for RSS article deduplication and article pool deduplication.

    Args:
        s1: First string (typically an article title).
        s2: Second string.
        threshold: Similarity ratio above which strings are considered near-duplicates.

    Returns:
        True if the strings are near-duplicates.
    """
    return similarity(s1, s2) >= threshold


def best_match(query: str, candidates: List[str]) -> Optional[str]:
    """Find the best-matching string from a list of candidates.

    Args:
        query: The string to match against.
        candidates: List of candidate strings.

    Returns:
        The candidate with the highest similarity score, or None if candidates is empty.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: similarity(query, c))


def best_match_score(query: str, candidates: List[str]) -> float:
    """Find the similarity score of the best match in a list of candidates.

    Args:
        query: The string to match against.
        candidates: List of candidate strings.

    Returns:
        The highest similarity score, or 0.0 if candidates is empty.
    """
    if not candidates:
        return 0.0
    return max(similarity(query, c) for c in candidates)
