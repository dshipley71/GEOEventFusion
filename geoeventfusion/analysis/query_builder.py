"""GDELT query construction for GEOEventFusion.

Builds enriched GDELT query strings by composing applicable operators.
Operators are additive — each adds a constraint or context layer to the base query.

Known operator rules from CLAUDE.md:
- near<N>: requires terms >= 5 characters (shorter terms produce zero results).
- repeat<N>: accepts only a single word — no phrase searches.
- domainis: requires exact domain without subdomain.
- sourcecountry: uses FIPS codes (not ISO 3166). No spaces in country names.
- imagetag: values must be in quote marks even for single words.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def _validate_gdelt_query(query: str, near_min_term_length: int = 5) -> str:
    """Strip near<N>: operators with terms shorter than the minimum length.

    Args:
        query: Raw query string that may contain near<N>: operators.
        near_min_term_length: Minimum character length for near operator terms.

    Returns:
        Query with invalid near operators removed.
    """
    # Match near<N>:"term1 term2" patterns
    near_pattern = re.compile(r'near\d+:"([^"]+)"', re.IGNORECASE)

    def _check_near(match: re.Match) -> str:
        terms = match.group(1).split()
        if any(len(t) < near_min_term_length for t in terms):
            logger.debug(
                "Stripped near operator with short terms: %s", match.group(0)
            )
            return ""
        return match.group(0)

    return near_pattern.sub(_check_near, query).strip()


class QueryBuilder:
    """Composes enriched GDELT DOC 2.0 query strings for the GDELTAgent.

    Builds the final_query by combining the base query with applicable operator
    expansions: boolean aliases, GKG themes, repeat thresholds, tone filters,
    proximity search, and source scoping operators.

    Args:
        near_min_term_length: Minimum term length for near<N>: operators.
        near_window: Proximity window in words for near<N>: operators.
        repeat_threshold: Minimum repetitions for repeat<N>: operator.
    """

    def __init__(
        self,
        near_min_term_length: int = 5,
        near_window: int = 15,
        repeat_threshold: int = 3,
    ) -> None:
        self.near_min_term_length = near_min_term_length
        self.near_window = near_window
        self.repeat_threshold = repeat_threshold

    def build_base_query(
        self,
        query: str,
        aliases: Optional[List[str]] = None,
        exclusions: Optional[List[str]] = None,
        gkg_themes: Optional[List[str]] = None,
        add_repeat: bool = False,
        add_near: bool = False,
        tone_filter: Optional[float] = None,
        toneabs_filter: Optional[float] = None,
    ) -> str:
        """Build the enriched query string for core article pool fetches.

        Args:
            query: Base keyword or keyphrase query.
            aliases: Alternative spellings or abbreviations to OR together.
            exclusions: Terms/operators to exclude (prefixed with -).
            gkg_themes: GKG theme codes (e.g., ["MARITIME_SECURITY"]).
            add_repeat: If True, add repeat<N>: operator for the first query word.
            add_near: If True, add near<N>: proximity operator where valid.
            tone_filter: If set, add tone< operator (e.g., -5.0 -> "tone<-5.0").
            toneabs_filter: If set, add toneabs> operator (e.g., 8.0 -> "toneabs>8.0").

        Returns:
            Enriched query string.
        """
        parts = []

        # Core phrase
        core = query.strip()
        if " " in core and not (core.startswith('"') and core.endswith('"')):
            parts.append(f'"{core}"')
        else:
            parts.append(core)

        # Boolean OR alias expansion
        if aliases:
            alias_group = " OR ".join(
                f'"{a}"' if " " in a else a for a in aliases
            )
            parts.append(f"({alias_group})")

        # GKG theme operators — OR'd together so any matching theme broadens recall.
        # AND-ing multiple theme: clauses requires ALL themes to appear simultaneously,
        # which is far too restrictive and typically returns zero results.
        if gkg_themes:
            if len(gkg_themes) == 1:
                parts.append(f"theme:{gkg_themes[0].upper()}")
            else:
                theme_or = " OR ".join(f"theme:{t.upper()}" for t in gkg_themes)
                parts.append(f"({theme_or})")

        # repeat<N>: relevance filter — only for single words
        if add_repeat:
            first_word = _extract_first_word(query)
            if first_word and len(first_word) >= self.near_min_term_length:
                parts.append(f'repeat{self.repeat_threshold}:"{first_word}"')

        # near<N>: proximity search — validate term lengths first
        if add_near and " " in query:
            near_expr = f'near{self.near_window}:"{query}"'
            validated = _validate_gdelt_query(near_expr, self.near_min_term_length)
            if validated:
                parts.append(validated)

        # Tone filters (applied inline to the query)
        if tone_filter is not None:
            parts.append(f"tone<{tone_filter}")

        if toneabs_filter is not None:
            parts.append(f"toneabs>{toneabs_filter}")

        # Exclusions
        if exclusions:
            for excl in exclusions:
                parts.append(f"-{excl}")

        return " ".join(parts)

    def build_source_country_query(self, base_query: str, fips_code: str) -> str:
        """Build a country-scoped query using sourcecountry: operator.

        Args:
            base_query: The enriched base query string.
            fips_code: FIPS country code (NOT ISO 3166). No spaces.

        Returns:
            Query with sourcecountry: prepended.
        """
        # Normalize: lowercase, remove spaces
        code = fips_code.lower().replace(" ", "")
        return f"sourcecountry:{code} {base_query}"

    def build_source_lang_query(self, base_query: str, iso_lang: str) -> str:
        """Build a language-scoped query using sourcelang: operator.

        Args:
            base_query: The enriched base query string.
            iso_lang: 3-character ISO language code (e.g., "ara").

        Returns:
            Query with sourcelang: prepended.
        """
        return f"sourcelang:{iso_lang.lower()} {base_query}"

    def build_authoritative_domain_query(
        self, base_query: str, domains: List[str]
    ) -> str:
        """Build an authority-domain query using domainis: OR-chain.

        domainis: requires exact domain without subdomain prefix.
        Multiple domains are OR'd together.

        Args:
            base_query: The enriched base query string.
            domains: List of exact domain names (e.g., ["un.org", "state.gov"]).

        Returns:
            Query with domainis: OR-chain prepended.
        """
        if not domains:
            return base_query
        if len(domains) == 1:
            domain_expr = f"domainis:{domains[0]}"
        else:
            or_parts = " OR ".join(f"domainis:{d}" for d in domains)
            domain_expr = f"({or_parts})"
        return f"{domain_expr} {base_query}"

    def build_imagetag_query(self, image_tags: List[str]) -> str:
        """Build an imagetag: OR-chain for visual intelligence fetches.

        imagetag: values must be in quote marks even for single words.

        Args:
            image_tags: List of VGKG image tag values.

        Returns:
            imagetag: OR-chain string.
        """
        if not image_tags:
            return ""
        tag_parts = " OR ".join(f'imagetag:"{tag}"' for tag in image_tags)
        if len(image_tags) > 1:
            return f"({tag_parts})"
        return tag_parts

    def build_high_neg_query(self, base_query: str, tone_threshold: float = -5.0) -> str:
        """Build a high-negative-tone filtered query.

        Args:
            base_query: The enriched base query string.
            tone_threshold: Maximum tone value (e.g., -5.0 means tone< -5.0).

        Returns:
            Query with inline tone< filter appended.
        """
        return f"{base_query} tone<{tone_threshold}"

    def build_high_emotion_query(self, base_query: str, toneabs_threshold: float = 8.0) -> str:
        """Build a high-absolute-tone filtered query.

        Args:
            base_query: The enriched base query string.
            toneabs_threshold: Minimum absolute tone value.

        Returns:
            Query with inline toneabs> filter appended.
        """
        return f"{base_query} toneabs>{toneabs_threshold}"

    def suggest_gkg_themes(
        self,
        query: str,
        llm_client: Optional[object] = None,
    ) -> List[str]:
        """Suggest GKG theme codes using LLM assistance.

        If no LLM client is provided, returns an empty list.
        Uses a 2-attempt retry with 256-token budget.

        Args:
            query: The base search query.
            llm_client: An LLMClient instance for theme suggestion.

        Returns:
            List of GKG theme code strings.
        """
        if llm_client is None:
            return []

        system = (
            "You are a GDELT GKG theme code expert. "
            "Return only a JSON array of GDELT GKG theme code strings. "
            "Use only known GKG codes (e.g., MARITIME_SECURITY, CRISISLEX_T10_CONFLICT, "
            "WB_1673_POLITICAL_VIOLENCE, ELECTIONS_AND_VOTING). "
            "Return at most 5 codes."
        )
        prompt = (
            f"Suggest relevant GDELT GKG theme codes for this geopolitical query: {query}\n"
            "Return only a JSON array of strings."
        )

        for attempt in range(2):
            try:
                result = llm_client.call_json(system, prompt, max_tokens=256, temperature=0.0)  # type: ignore[union-attr]
                if isinstance(result, list):
                    return [str(t).upper() for t in result if isinstance(t, str)]
            except Exception as exc:
                logger.warning("GKG theme suggestion failed (attempt %d): %s", attempt + 1, exc)

        return []


def _extract_first_word(query: str) -> str:
    """Extract the first meaningful word from a query string.

    Strips quotes and leading/trailing operators.

    Args:
        query: Query string.

    Returns:
        First word as a plain string.
    """
    clean = re.sub(r'["\(\)]', "", query).strip()
    words = clean.split()
    if words:
        return words[0]
    return ""
