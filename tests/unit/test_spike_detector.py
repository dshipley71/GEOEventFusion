"""Unit tests for geoeventfusion.analysis.spike_detector.

Covers:
- detect_spikes: happy path, empty input, zero variance, custom threshold
- rank_spikes: ordering and rank field assignment
- compute_vol_ratio: valid data, empty data, zero norm
- find_phase_boundaries: with and without community reorganization
"""

from __future__ import annotations

import math

import pytest

from geoeventfusion.analysis.spike_detector import (
    compute_vol_ratio,
    detect_spikes,
    find_phase_boundaries,
    rank_spikes,
)
from geoeventfusion.models.events import SpikeWindow, TimelineStep, TimelineStepRaw


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _make_step(date: str, value: float) -> TimelineStep:
    return TimelineStep(date=date, value=value)


def _make_raw(date: str, volume: int, norm: float) -> TimelineStepRaw:
    return TimelineStepRaw(date=date, volume=volume, norm=norm)


# ── detect_spikes ─────────────────────────────────────────────────────────────────

class TestDetectSpikes:
    def test_detect_spikes_happy_path(self, sample_timeline_steps):
        """30-step fixture with 2 embedded spikes should produce exactly 2 spikes."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)

        assert len(spikes) == 2

    def test_detect_spikes_correct_dates(self, sample_timeline_steps):
        """The spike dates must correspond to the two high-value steps in the fixture."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        spike_dates = {s.date for s in spikes}

        assert "2024-01-31" in spike_dates, "Spike at index 10 (value 9.0) must be detected"
        assert "2024-03-16" in spike_dates, "Spike at index 25 (value 8.0) must be detected"

    def test_detect_spikes_sorted_by_z_score_descending(self, sample_timeline_steps):
        """Spikes must be returned with highest Z-score first."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)

        assert spikes[0].z_score > spikes[1].z_score
        assert spikes[0].date == "2024-01-31"  # value 9.0 > 8.0

    def test_detect_spikes_rank_field(self, sample_timeline_steps):
        """Rank field must start at 1 for the highest Z-score spike."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)

        assert spikes[0].rank == 1
        assert spikes[1].rank == 2

    def test_detect_spikes_z_scores_above_threshold(self, sample_timeline_steps):
        """All returned spikes must have Z-score >= the threshold."""
        threshold = 1.5
        spikes = detect_spikes(sample_timeline_steps, z_threshold=threshold)

        for spike in spikes:
            assert spike.z_score >= threshold, (
                f"Spike at {spike.date} has Z={spike.z_score} below threshold {threshold}"
            )

    def test_detect_spikes_empty_timeline(self):
        """Empty timeline must return an empty list without raising."""
        spikes = detect_spikes([], z_threshold=1.5)
        assert spikes == []

    def test_detect_spikes_zero_variance_timeline(self):
        """A flat timeline (all identical values) must return empty — no spikes."""
        flat = [_make_step(f"2024-01-{i:02d}", 5.0) for i in range(1, 11)]
        spikes = detect_spikes(flat, z_threshold=1.5)
        assert spikes == []

    def test_detect_spikes_custom_high_threshold(self, sample_timeline_steps):
        """With a very high threshold (e.g., 5.0), even the fixture spikes may not qualify."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=5.0)
        # Z for value 9.0 ≈ 4.04 < 5.0, so no spikes expected
        assert spikes == []

    def test_detect_spikes_custom_low_threshold(self, sample_timeline_steps):
        """With a low threshold (0.1), more points may qualify — must all be >= threshold."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=0.1)
        for spike in spikes:
            assert spike.z_score >= 0.1

    def test_detect_spikes_single_step_timeline(self):
        """A one-step timeline has zero variance — no spikes should be returned."""
        single = [_make_step("2024-01-01", 10.0)]
        spikes = detect_spikes(single, z_threshold=1.5)
        assert spikes == []

    def test_detect_spikes_stores_query(self):
        """The query parameter must be stored in each SpikeWindow."""
        steps = [_make_step("2024-01-01", 1.0), _make_step("2024-01-02", 9.0)]
        spikes = detect_spikes(steps, z_threshold=0.5, query="Houthi Red Sea")
        for spike in spikes:
            assert spike.query == "Houthi Red Sea"

    def test_detect_spikes_volumes_match_timeline(self, sample_timeline_steps):
        """Spike volume must equal the value in the corresponding TimelineStep."""
        spikes = detect_spikes(sample_timeline_steps, z_threshold=1.5)
        vol_map = {s.date: s.volume for s in spikes}

        assert vol_map["2024-01-31"] == 9.0
        assert vol_map["2024-03-16"] == 8.0

    def test_detect_spikes_two_steps_one_spike(self):
        """With 2 steps and one outlier, exactly one spike is detected."""
        steps = [_make_step("2024-01-01", 1.0), _make_step("2024-01-02", 10.0)]
        spikes = detect_spikes(steps, z_threshold=0.5)
        assert len(spikes) == 1
        assert spikes[0].date == "2024-01-02"


# ── rank_spikes ───────────────────────────────────────────────────────────────────

class TestRankSpikes:
    def test_rank_spikes_sorts_descending(self):
        """rank_spikes must sort by Z-score descending."""
        spikes = [
            SpikeWindow(date="2024-01-01", z_score=1.8, volume=3.5, rank=0),
            SpikeWindow(date="2024-01-02", z_score=3.2, volume=7.0, rank=0),
            SpikeWindow(date="2024-01-03", z_score=2.4, volume=5.5, rank=0),
        ]
        ranked = rank_spikes(spikes)

        assert ranked[0].z_score == pytest.approx(3.2)
        assert ranked[1].z_score == pytest.approx(2.4)
        assert ranked[2].z_score == pytest.approx(1.8)

    def test_rank_spikes_assigns_rank_field(self):
        """rank field must be 1-indexed, starting from the highest Z-score."""
        spikes = [
            SpikeWindow(date="2024-01-01", z_score=2.0, volume=4.0, rank=0),
            SpikeWindow(date="2024-01-02", z_score=5.0, volume=9.0, rank=0),
        ]
        ranked = rank_spikes(spikes)

        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_rank_spikes_empty_list(self):
        """Empty input must return an empty list."""
        assert rank_spikes([]) == []

    def test_rank_spikes_single_spike(self):
        """Single spike must receive rank 1."""
        spikes = [SpikeWindow(date="2024-01-01", z_score=2.5, volume=6.0, rank=0)]
        ranked = rank_spikes(spikes)
        assert ranked[0].rank == 1


# ── compute_vol_ratio ─────────────────────────────────────────────────────────────

class TestComputeVolRatio:
    def test_compute_vol_ratio_happy_path(self, sample_timeline_volraw):
        """vol_ratio must be a positive float for valid input."""
        ratio = compute_vol_ratio(sample_timeline_volraw)
        assert isinstance(ratio, float)
        assert ratio > 0.0

    def test_compute_vol_ratio_expected_value(self, sample_timeline_volraw):
        """vol_ratio = mean(volume) / mean(norm) for the fixture data."""
        # volumes: [50, 60, 55, 250, 45] → mean = 92.0
        # norms:   [10000, 10500, 10200, 11000, 9800] → mean = 10300.0
        # ratio ≈ 92.0 / 10300.0 ≈ 0.008932
        ratio = compute_vol_ratio(sample_timeline_volraw)
        assert ratio == pytest.approx(92.0 / 10300.0, rel=1e-3)

    def test_compute_vol_ratio_empty_list(self):
        """Empty list must return 0.0."""
        assert compute_vol_ratio([]) == 0.0

    def test_compute_vol_ratio_zero_norm(self):
        """Steps with zero norm must be skipped; if all norms are zero, return 0.0."""
        steps = [
            _make_raw("2024-01-01", volume=100, norm=0),
            _make_raw("2024-01-02", volume=200, norm=0),
        ]
        assert compute_vol_ratio(steps) == 0.0

    def test_compute_vol_ratio_mixed_zero_norm(self):
        """Steps with norm=0 must be excluded; valid steps compute the ratio."""
        steps = [
            _make_raw("2024-01-01", volume=100, norm=0),    # excluded
            _make_raw("2024-01-02", volume=100, norm=5000),  # included
        ]
        ratio = compute_vol_ratio(steps)
        # Only 1 valid step: mean_volume=100, mean_norm=5000 → 0.02
        assert ratio == pytest.approx(0.02, rel=1e-6)

    def test_compute_vol_ratio_result_is_rounded(self, sample_timeline_volraw):
        """Result must be rounded to 6 decimal places."""
        ratio = compute_vol_ratio(sample_timeline_volraw)
        assert ratio == round(ratio, 6)


# ── find_phase_boundaries ─────────────────────────────────────────────────────────

class TestFindPhaseBoundaries:
    def _make_spikes(self, dates_and_zscores):
        return [
            SpikeWindow(date=d, z_score=z, volume=5.0, rank=i + 1)
            for i, (d, z) in enumerate(dates_and_zscores)
        ]

    def test_find_phase_boundaries_top_spikes(self):
        """Top-3 spike dates must appear in boundaries."""
        spikes = self._make_spikes([
            ("2024-01-31", 4.0),
            ("2024-03-16", 3.4),
            ("2024-02-15", 2.0),
            ("2024-01-10", 1.6),
        ])
        boundaries = find_phase_boundaries(spikes)
        # Only top 3 spikes are used
        assert "2024-01-31" in boundaries
        assert "2024-03-16" in boundaries
        assert "2024-02-15" in boundaries
        assert "2024-01-10" not in boundaries

    def test_find_phase_boundaries_sorted_chronologically(self):
        """Boundary dates must be returned in chronological order."""
        spikes = self._make_spikes([
            ("2024-03-01", 4.0),
            ("2024-01-15", 3.0),
        ])
        boundaries = find_phase_boundaries(spikes)
        assert boundaries == sorted(boundaries)

    def test_find_phase_boundaries_with_community_reorg(self):
        """High reorganization score must add the community boundary date."""
        spikes = self._make_spikes([("2024-01-31", 4.0)])
        boundaries = find_phase_boundaries(
            spikes,
            reorganization_score=0.75,
            phase_boundary_date="2024-02-20",
        )
        assert "2024-02-20" in boundaries

    def test_find_phase_boundaries_low_reorg_score_ignored(self):
        """Reorganization score ≤ 0.4 must not add the community boundary date."""
        spikes = self._make_spikes([("2024-01-31", 4.0)])
        boundaries = find_phase_boundaries(
            spikes,
            reorganization_score=0.3,
            phase_boundary_date="2024-02-20",
        )
        assert "2024-02-20" not in boundaries

    def test_find_phase_boundaries_no_duplicates(self):
        """If the community date matches a spike date, it must not be duplicated."""
        spikes = self._make_spikes([("2024-01-31", 4.0)])
        boundaries = find_phase_boundaries(
            spikes,
            reorganization_score=0.9,
            phase_boundary_date="2024-01-31",
        )
        assert boundaries.count("2024-01-31") == 1

    def test_find_phase_boundaries_empty_spikes(self):
        """Empty spikes list with no community boundary returns an empty list."""
        boundaries = find_phase_boundaries([])
        assert boundaries == []

    def test_find_phase_boundaries_no_reorg_score(self):
        """None reorganization score must not cause any error."""
        spikes = self._make_spikes([("2024-01-31", 4.0)])
        boundaries = find_phase_boundaries(spikes, reorganization_score=None)
        assert "2024-01-31" in boundaries
