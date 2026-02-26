"""Unit tests for geoeventfusion.analysis.tone_analyzer.

Covers:
- analyze_tone_distribution: happy path, empty input, zero counts
- compute_language_stats: aggregation, diversity index, empty input
- compute_country_stats: aggregation, top-N countries, empty input
- compute_tone_trend: improving, deteriorating, stable, empty input
- _shannon_index: correctness, zero distribution
"""

from __future__ import annotations

import math

import pytest

from geoeventfusion.analysis.tone_analyzer import (
    analyze_tone_distribution,
    compute_country_stats,
    compute_language_stats,
    compute_tone_trend,
)
from geoeventfusion.models.events import TimelineStep, ToneChartBin


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _make_step(date: str, value: float, label: str = "") -> TimelineStep:
    return TimelineStep(date=date, value=value, label=label)


def _make_bin(tone_value: float, count: int) -> ToneChartBin:
    return ToneChartBin(tone_value=tone_value, count=count)


# ── analyze_tone_distribution ─────────────────────────────────────────────────────

class TestAnalyzeToneDistribution:
    def test_happy_path_fixture(self, sample_tonechart_bins):
        """Fixture tonechart must produce valid ToneStats."""
        stats = analyze_tone_distribution(sample_tonechart_bins)

        assert stats.total_articles > 0
        assert -100.0 <= stats.modal_tone <= 100.0
        assert -100.0 <= stats.mean_tone <= 100.0
        assert stats.std_dev >= 0.0
        assert 0.0 <= stats.polarity_ratio <= 1.0

    def test_modal_tone_is_highest_count_bin(self, sample_tonechart_bins):
        """modal_tone must be the tone_value of the bin with the highest count."""
        stats = analyze_tone_distribution(sample_tonechart_bins)
        # Fixture: bin -2.0 has count 201 (highest)
        assert stats.modal_tone == pytest.approx(-2.0)

    def test_polarity_ratio_negatively_skewed(self, sample_tonechart_bins):
        """The fixture is negatively skewed — polarity_ratio must be > 0.5."""
        stats = analyze_tone_distribution(sample_tonechart_bins)
        assert stats.polarity_ratio > 0.5

    def test_empty_tonechart_returns_zeroed_stats(self):
        """Empty tonechart must return a zeroed ToneStats without raising."""
        stats = analyze_tone_distribution([])

        assert stats.modal_tone == 0.0
        assert stats.mean_tone == 0.0
        assert stats.std_dev == 0.0
        assert stats.polarity_ratio == 0.0
        assert stats.total_articles == 0

    def test_zero_count_bins_returns_zeroed_stats(self):
        """All-zero-count bins must return zeroed ToneStats."""
        bins = [_make_bin(-5.0, 0), _make_bin(0.0, 0), _make_bin(5.0, 0)]
        stats = analyze_tone_distribution(bins)

        assert stats.total_articles == 0
        assert stats.mean_tone == 0.0

    def test_single_bin_stats(self):
        """Single bin must produce trivial but correct stats."""
        bins = [_make_bin(-3.0, 100)]
        stats = analyze_tone_distribution(bins)

        assert stats.modal_tone == pytest.approx(-3.0)
        assert stats.mean_tone == pytest.approx(-3.0)
        assert stats.std_dev == pytest.approx(0.0)
        assert stats.polarity_ratio == pytest.approx(1.0)
        assert stats.total_articles == 100

    def test_equal_positive_negative_polarity_ratio_half(self):
        """Equal positive and negative counts must yield polarity_ratio ≈ 0.5."""
        bins = [_make_bin(-1.0, 50), _make_bin(1.0, 50)]
        stats = analyze_tone_distribution(bins)

        assert stats.polarity_ratio == pytest.approx(0.5)

    def test_mean_tone_weighted_average(self):
        """mean_tone must equal the weighted average of tone_values."""
        bins = [_make_bin(-4.0, 100), _make_bin(2.0, 100)]
        stats = analyze_tone_distribution(bins)

        expected_mean = (-4.0 * 100 + 2.0 * 100) / 200
        assert stats.mean_tone == pytest.approx(expected_mean, rel=1e-4)

    def test_std_dev_positive_for_varied_distribution(self):
        """std_dev must be positive for a distribution with more than one tone value."""
        bins = [_make_bin(-5.0, 80), _make_bin(0.0, 40), _make_bin(5.0, 20)]
        stats = analyze_tone_distribution(bins)

        assert stats.std_dev > 0.0

    def test_total_articles_counts_all_bins(self):
        """total_articles must sum all bin counts."""
        bins = [_make_bin(-2.0, 30), _make_bin(-1.0, 70), _make_bin(1.0, 50)]
        stats = analyze_tone_distribution(bins)

        assert stats.total_articles == 150


# ── compute_language_stats ────────────────────────────────────────────────────────

class TestComputeLanguageStats:
    def _make_lang_steps(self):
        """Language timeline steps covering English, Arabic, Russian."""
        return [
            _make_step("2024-01-01", 50.0, label="English"),
            _make_step("2024-01-04", 30.0, label="English"),
            _make_step("2024-01-01", 20.0, label="Arabic"),
            _make_step("2024-01-04", 15.0, label="Arabic"),
            _make_step("2024-01-01", 10.0, label="Russian"),
        ]

    def test_happy_path_top_languages_populated(self):
        """top_languages must contain all languages from the input steps."""
        stats = compute_language_stats(self._make_lang_steps())

        language_names = [item["language"] for item in stats.top_languages]
        assert "English" in language_names
        assert "Arabic" in language_names
        assert "Russian" in language_names

    def test_top_languages_sorted_by_volume_descending(self):
        """Languages must be sorted by aggregated volume descending."""
        stats = compute_language_stats(self._make_lang_steps())

        volumes = [item["volume"] for item in stats.top_languages]
        assert volumes == sorted(volumes, reverse=True)

    def test_top_languages_english_is_first(self):
        """English (highest volume) must appear first."""
        stats = compute_language_stats(self._make_lang_steps())
        assert stats.top_languages[0]["language"] == "English"

    def test_share_sums_to_one(self):
        """Shares of all top languages must sum to approximately 1.0."""
        stats = compute_language_stats(self._make_lang_steps())
        total_share = sum(item["share"] for item in stats.top_languages)
        assert total_share == pytest.approx(1.0, abs=1e-3)

    def test_volumes_aggregated_across_steps(self):
        """Volume for each language must be the sum across all time steps."""
        stats = compute_language_stats(self._make_lang_steps())
        english = next(i for i in stats.top_languages if i["language"] == "English")
        assert english["volume"] == pytest.approx(80.0)  # 50 + 30

    def test_diversity_index_positive_for_multi_language(self):
        """Shannon diversity index must be positive when multiple languages are present."""
        stats = compute_language_stats(self._make_lang_steps())
        assert stats.diversity_index > 0.0

    def test_diversity_index_zero_for_single_language(self):
        """Shannon diversity index must be 0 when only one language is present."""
        steps = [_make_step("2024-01-01", 100.0, label="English")]
        stats = compute_language_stats(steps)
        assert stats.diversity_index == pytest.approx(0.0, abs=1e-6)

    def test_empty_input_returns_empty_stats(self):
        """Empty input must return empty top_languages and zero diversity."""
        stats = compute_language_stats([])
        assert stats.top_languages == []
        assert stats.diversity_index == pytest.approx(0.0)

    def test_steps_without_labels_excluded(self):
        """Steps with empty labels must not appear in top_languages."""
        steps = [
            _make_step("2024-01-01", 50.0, label="English"),
            _make_step("2024-01-02", 10.0, label=""),  # no label
        ]
        stats = compute_language_stats(steps)
        labels = [item["language"] for item in stats.top_languages]
        assert "" not in labels
        assert "English" in labels


# ── compute_country_stats ─────────────────────────────────────────────────────────

class TestComputeCountryStats:
    def _make_country_steps(self):
        return [
            _make_step("2024-01-01", 60.0, label="United States"),
            _make_step("2024-01-04", 40.0, label="United States"),
            _make_step("2024-01-01", 30.0, label="United Kingdom"),
            _make_step("2024-01-01", 25.0, label="Qatar"),
        ]

    def test_top_countries_populated(self):
        """top_countries must list all countries from input."""
        stats = compute_country_stats(self._make_country_steps())
        country_names = [item["country"] for item in stats.top_countries]
        assert "United States" in country_names
        assert "United Kingdom" in country_names

    def test_top_countries_sorted_by_volume_descending(self):
        """Countries must be sorted by aggregated volume descending."""
        stats = compute_country_stats(self._make_country_steps())
        volumes = [item["volume"] for item in stats.top_countries]
        assert volumes == sorted(volumes, reverse=True)

    def test_us_aggregated_volume(self):
        """US volume must aggregate across all its steps."""
        stats = compute_country_stats(self._make_country_steps())
        us = next(i for i in stats.top_countries if i["country"] == "United States")
        assert us["volume"] == pytest.approx(100.0)

    def test_diversity_index_positive(self):
        """Diversity index must be positive for multiple countries."""
        stats = compute_country_stats(self._make_country_steps())
        assert stats.diversity_index > 0.0

    def test_empty_input_returns_empty_stats(self):
        """Empty input must return empty top_countries."""
        stats = compute_country_stats([])
        assert stats.top_countries == []
        assert stats.diversity_index == pytest.approx(0.0)


# ── compute_tone_trend ────────────────────────────────────────────────────────────

class TestComputeToneTrend:
    def _make_trend(self, early_val: float, late_val: float, n: int = 9) -> list:
        """Create a timeline with distinct early and late tone values."""
        third = n // 3
        steps = []
        for i in range(third):
            steps.append(_make_step(f"2024-01-{i + 1:02d}", early_val))
        for i in range(third, 2 * third):
            steps.append(_make_step(f"2024-02-{i - third + 1:02d}", (early_val + late_val) / 2))
        for i in range(2 * third, n):
            steps.append(_make_step(f"2024-03-{i - 2 * third + 1:02d}", late_val))
        return steps

    def test_deteriorating_when_late_lower(self):
        """Tone declining by more than 0.5 must be labelled 'deteriorating'."""
        steps = self._make_trend(early_val=-2.0, late_val=-5.0)
        result = compute_tone_trend(steps)
        assert result["trend_direction"] == "deteriorating"
        assert result["trend_delta"] < -0.5

    def test_improving_when_late_higher(self):
        """Tone improving by more than 0.5 must be labelled 'improving'."""
        steps = self._make_trend(early_val=-5.0, late_val=-2.0)
        result = compute_tone_trend(steps)
        assert result["trend_direction"] == "improving"
        assert result["trend_delta"] > 0.5

    def test_stable_when_change_small(self):
        """Tone change within ±0.5 must be labelled 'stable'."""
        steps = self._make_trend(early_val=-3.0, late_val=-3.3)
        result = compute_tone_trend(steps)
        assert result["trend_direction"] == "stable"

    def test_empty_input_returns_stable_zeros(self):
        """Empty input must return a zeroed stable result."""
        result = compute_tone_trend([])
        assert result["trend_direction"] == "stable"
        assert result["early_mean"] == 0.0
        assert result["late_mean"] == 0.0
        assert result["trend_delta"] == 0.0

    def test_trend_delta_correct(self):
        """trend_delta must equal late_mean - early_mean."""
        steps = self._make_trend(early_val=-3.0, late_val=-6.0)
        result = compute_tone_trend(steps)
        expected_delta = result["late_mean"] - result["early_mean"]
        assert result["trend_delta"] == pytest.approx(expected_delta, abs=1e-3)

    def test_single_step_timeline(self):
        """A single-step timeline must return without raising."""
        steps = [_make_step("2024-01-01", -3.0)]
        result = compute_tone_trend(steps)
        assert "trend_direction" in result

    def test_result_keys_present(self):
        """Result dict must contain all expected keys."""
        steps = self._make_trend(-3.0, -5.0)
        result = compute_tone_trend(steps)
        assert "early_mean" in result
        assert "late_mean" in result
        assert "trend_direction" in result
        assert "trend_delta" in result
