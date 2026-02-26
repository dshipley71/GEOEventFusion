"""Unit tests for geoeventfusion.analysis.query_builder.

Covers:
- QueryBuilder.build_base_query: phrase wrapping, aliases, themes, repeat, near, tone filters
- build_source_country_query: FIPS code normalization
- build_source_lang_query: language code handling
- build_authoritative_domain_query: single and multi-domain OR chains
- build_imagetag_query: quoted tags, OR chains
- build_high_neg_query / build_high_emotion_query: inline tone filter appending
- _validate_gdelt_query: strips short-term near operators
- suggest_gkg_themes: LLM client delegation and no-client fallback
"""

from __future__ import annotations

import pytest

from geoeventfusion.analysis.query_builder import QueryBuilder, _validate_gdelt_query


class TestQueryBuilderBaseQuery:
    def setup_method(self):
        self.qb = QueryBuilder(near_min_term_length=5, near_window=15, repeat_threshold=3)

    def test_build_base_query_simple_single_word(self):
        """A single-word query must not be wrapped in quotes."""
        result = self.qb.build_base_query("Houthi")
        assert result == "Houthi"

    def test_build_base_query_multi_word_wrapped_in_quotes(self):
        """A multi-word query not already quoted must be wrapped in double quotes."""
        result = self.qb.build_base_query("Houthi Red Sea")
        assert result.startswith('"Houthi Red Sea"')

    def test_build_base_query_already_quoted_not_double_wrapped(self):
        """A query already in double quotes must not be double-wrapped."""
        result = self.qb.build_base_query('"Houthi Red Sea"')
        # Should start with the quoted phrase, not add extra quotes
        assert result.count('"Houthi Red Sea"') == 1

    def test_build_base_query_with_aliases(self):
        """Aliases must be joined with OR and wrapped in parentheses."""
        result = self.qb.build_base_query("Houthi", aliases=["Ansar Allah", "huthis"])
        assert '("Ansar Allah" OR huthis)' in result

    def test_build_base_query_single_word_alias_unquoted(self):
        """Single-word aliases must not be wrapped in quotes."""
        result = self.qb.build_base_query("Houthi", aliases=["huthis"])
        assert "(huthis)" in result

    def test_build_base_query_with_gkg_themes(self):
        """GKG theme codes must be added as theme:UPPER_CASE operators."""
        result = self.qb.build_base_query("Houthi", gkg_themes=["maritime_security"])
        assert "theme:MARITIME_SECURITY" in result

    def test_build_base_query_with_repeat(self):
        """add_repeat=True must add repeat<N>: operator for the first word."""
        result = self.qb.build_base_query("Houthi", add_repeat=True)
        assert 'repeat3:"Houthi"' in result

    def test_build_base_query_repeat_skipped_for_short_words(self):
        """repeat<N>: must not be added if the first word is shorter than near_min_term_length."""
        qb = QueryBuilder(near_min_term_length=5, repeat_threshold=3)
        result = qb.build_base_query("war", add_repeat=True)  # "war" < 5 chars
        assert "repeat" not in result

    def test_build_base_query_with_near_multi_word(self):
        """add_near=True with a multi-word query of long-enough terms must add near<N>:."""
        result = self.qb.build_base_query("Houthi attack", add_near=True)
        assert "near15:" in result

    def test_build_base_query_near_skipped_for_single_word(self):
        """near<N>: must not be added for single-word queries."""
        result = self.qb.build_base_query("Houthi", add_near=True)
        assert "near" not in result

    def test_build_base_query_tone_filter(self):
        """tone_filter must be appended as tone<{value}."""
        result = self.qb.build_base_query("Houthi", tone_filter=-5.0)
        assert "tone<-5.0" in result

    def test_build_base_query_toneabs_filter(self):
        """toneabs_filter must be appended as toneabs>{value}."""
        result = self.qb.build_base_query("Houthi", toneabs_filter=8.0)
        assert "toneabs>8.0" in result

    def test_build_base_query_exclusions(self):
        """Exclusions must be prefixed with - and added to the query."""
        result = self.qb.build_base_query("Houthi", exclusions=["sourcelang:spanish"])
        assert "-sourcelang:spanish" in result

    def test_build_base_query_all_options_combined(self):
        """All options combined must produce a correctly structured query."""
        result = self.qb.build_base_query(
            "Houthi Red Sea",
            aliases=["huthis"],
            gkg_themes=["MARITIME_SECURITY"],
            add_repeat=True,
            tone_filter=-5.0,
            exclusions=["sourcelang:spanish"],
        )
        assert '"Houthi Red Sea"' in result
        assert "(huthis)" in result
        assert "theme:MARITIME_SECURITY" in result
        assert "tone<-5.0" in result
        assert "-sourcelang:spanish" in result


class TestQueryBuilderSourceQueries:
    def setup_method(self):
        self.qb = QueryBuilder()

    def test_build_source_country_query_prepends_fips_code(self):
        """sourcecountry: operator must be prepended to the base query."""
        result = self.qb.build_source_country_query('"Houthi Red Sea"', "IR")
        assert result.startswith("sourcecountry:ir")

    def test_build_source_country_query_lowercases_code(self):
        """FIPS code must be lowercased."""
        result = self.qb.build_source_country_query('"Houthi"', "YE")
        assert "sourcecountry:ye" in result

    def test_build_source_country_query_strips_spaces(self):
        """Spaces in FIPS code must be removed."""
        result = self.qb.build_source_country_query('"test"', "saudi arabia")
        assert "sourcecountry:saudiarabia" in result

    def test_build_source_country_query_preserves_base_query(self):
        """The base query must be preserved after the sourcecountry: prefix."""
        base = '"Houthi Red Sea"'
        result = self.qb.build_source_country_query(base, "IR")
        assert base in result

    def test_build_source_lang_query_prepends_sourcelang(self):
        """sourcelang: operator must be prepended to the base query."""
        result = self.qb.build_source_lang_query('"Houthi"', "ara")
        assert result.startswith("sourcelang:ara")

    def test_build_source_lang_query_lowercases_lang_code(self):
        """Language code must be lowercased."""
        result = self.qb.build_source_lang_query('"Houthi"', "ARA")
        assert "sourcelang:ara" in result

    def test_build_source_lang_query_preserves_base_query(self):
        """The base query must be preserved after the sourcelang: prefix."""
        base = '"Houthi Red Sea"'
        result = self.qb.build_source_lang_query(base, "rus")
        assert base in result


class TestQueryBuilderAuthorativeDomains:
    def setup_method(self):
        self.qb = QueryBuilder()

    def test_build_authoritative_domain_query_single_domain(self):
        """Single domain must produce domainis:<domain> without OR-chain."""
        result = self.qb.build_authoritative_domain_query('"Houthi"', ["un.org"])
        assert "domainis:un.org" in result
        assert "OR" not in result

    def test_build_authoritative_domain_query_multi_domain_or_chain(self):
        """Multiple domains must be OR'd together in parentheses."""
        result = self.qb.build_authoritative_domain_query(
            '"Houthi"', ["un.org", "state.gov", "nato.int"]
        )
        assert "(domainis:un.org OR domainis:state.gov OR domainis:nato.int)" in result

    def test_build_authoritative_domain_query_empty_domains(self):
        """Empty domains list must return the base query unchanged."""
        base = '"Houthi Red Sea"'
        result = self.qb.build_authoritative_domain_query(base, [])
        assert result == base

    def test_build_authoritative_domain_query_preserves_base_query(self):
        """The base query must appear in the result."""
        base = '"test query"'
        result = self.qb.build_authoritative_domain_query(base, ["un.org"])
        assert base in result


class TestQueryBuilderImageTags:
    def setup_method(self):
        self.qb = QueryBuilder()

    def test_build_imagetag_query_single_tag_quoted(self):
        """Single imagetag must be wrapped in double quotes per GDELT rules."""
        result = self.qb.build_imagetag_query(["military"])
        assert 'imagetag:"military"' in result

    def test_build_imagetag_query_multi_tag_or_chain(self):
        """Multiple tags must be OR'd together."""
        result = self.qb.build_imagetag_query(["military", "protest", "fire"])
        assert 'imagetag:"military"' in result
        assert 'imagetag:"protest"' in result
        assert 'imagetag:"fire"' in result
        assert "OR" in result

    def test_build_imagetag_query_multi_tag_in_parentheses(self):
        """Multi-tag query must be wrapped in parentheses."""
        result = self.qb.build_imagetag_query(["military", "fire"])
        assert result.startswith("(")
        assert result.endswith(")")

    def test_build_imagetag_query_empty_list(self):
        """Empty image tags list must return empty string."""
        result = self.qb.build_imagetag_query([])
        assert result == ""


class TestQueryBuilderToneQueries:
    def setup_method(self):
        self.qb = QueryBuilder()

    def test_build_high_neg_query_appends_tone_filter(self):
        """build_high_neg_query must append tone<{threshold}."""
        result = self.qb.build_high_neg_query('"Houthi"', tone_threshold=-5.0)
        assert "tone<-5.0" in result

    def test_build_high_neg_query_preserves_base(self):
        """The base query must appear before the tone filter."""
        base = '"Houthi Red Sea"'
        result = self.qb.build_high_neg_query(base)
        assert result.startswith(base)

    def test_build_high_emotion_query_appends_toneabs_filter(self):
        """build_high_emotion_query must append toneabs>{threshold}."""
        result = self.qb.build_high_emotion_query('"Houthi"', toneabs_threshold=8.0)
        assert "toneabs>8.0" in result

    def test_build_high_emotion_query_preserves_base(self):
        """The base query must appear before the toneabs filter."""
        base = '"Houthi"'
        result = self.qb.build_high_emotion_query(base)
        assert result.startswith(base)


class TestValidateGdeltQuery:
    def test_validate_strips_short_term_near_operator(self):
        """near operator with terms shorter than min_length must be stripped."""
        query = 'near15:"war oil"'  # "war" and "oil" are 3 chars each < 5
        result = _validate_gdelt_query(query, near_min_term_length=5)
        assert "near15:" not in result

    def test_validate_keeps_valid_near_operator(self):
        """near operator with all terms >= min_length must be preserved."""
        query = 'near15:"Houthi shipping"'  # both terms >= 5 chars
        result = _validate_gdelt_query(query, near_min_term_length=5)
        assert "near15:" in result

    def test_validate_strips_mixed_near_operator(self):
        """near operator with any short term must be stripped entirely."""
        query = 'near15:"Houthi war"'  # "war" is 3 chars
        result = _validate_gdelt_query(query, near_min_term_length=5)
        assert "near15:" not in result

    def test_validate_leaves_non_near_operators_unchanged(self):
        """Non-near operators must not be affected by validation."""
        query = '"Houthi" theme:MARITIME_SECURITY tone<-5.0'
        result = _validate_gdelt_query(query, near_min_term_length=5)
        assert "theme:MARITIME_SECURITY" in result
        assert "tone<-5.0" in result

    def test_validate_empty_query(self):
        """Empty query must be returned unchanged."""
        result = _validate_gdelt_query("", near_min_term_length=5)
        assert result == ""

    def test_validate_case_insensitive_near(self):
        """near operator pattern must be matched case-insensitively."""
        query = 'NEAR15:"short test"'  # "short" >= 5, "test" >= 4 but < 5
        result = _validate_gdelt_query(query, near_min_term_length=5)
        # "test" is 4 chars < 5, so it should be stripped
        assert "NEAR15:" not in result


class TestQueryBuilderGkgThemes:
    def setup_method(self):
        self.qb = QueryBuilder()

    def test_suggest_gkg_themes_no_client_returns_empty(self):
        """Without an LLM client, suggest_gkg_themes must return an empty list."""
        result = self.qb.suggest_gkg_themes("Houthi Red Sea", llm_client=None)
        assert result == []

    def test_suggest_gkg_themes_with_mock_client(self, mock_llm_client):
        """With a mock LLM client returning a list, must return uppercase theme codes."""
        mock_llm_client.call_json.return_value = ["MARITIME_SECURITY", "WB_1673_POLITICAL_VIOLENCE"]
        result = self.qb.suggest_gkg_themes("Houthi Red Sea", llm_client=mock_llm_client)

        assert "MARITIME_SECURITY" in result
        assert "WB_1673_POLITICAL_VIOLENCE" in result

    def test_suggest_gkg_themes_uppercases_codes(self, mock_llm_client):
        """Theme codes returned by LLM must be uppercased."""
        mock_llm_client.call_json.return_value = ["maritime_security"]
        result = self.qb.suggest_gkg_themes("test", llm_client=mock_llm_client)

        assert "MARITIME_SECURITY" in result

    def test_suggest_gkg_themes_non_list_response_returns_empty(self, mock_llm_client):
        """Non-list LLM response must produce an empty list (graceful degradation)."""
        mock_llm_client.call_json.return_value = {"themes": ["MARITIME_SECURITY"]}
        result = self.qb.suggest_gkg_themes("test", llm_client=mock_llm_client)

        # call_json returns a dict, not a list — should return []
        assert result == []

    def test_suggest_gkg_themes_exception_returns_empty(self, mock_llm_client):
        """LLM client exceptions must be caught and return an empty list."""
        mock_llm_client.call_json.side_effect = RuntimeError("LLM failed")
        result = self.qb.suggest_gkg_themes("test", llm_client=mock_llm_client)

        assert result == []
