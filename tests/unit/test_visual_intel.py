"""Unit tests for geoeventfusion.analysis.visual_intel.

Covers:
- compute_novelty_score: formula correctness, edge cases, negative input
- check_staleness: stale detection, fresh image, missing dates, timezone handling
- parse_image_collage_response: happy path, empty response, malformed input
- rank_images_by_novelty: ordering
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from geoeventfusion.analysis.visual_intel import (
    check_staleness,
    compute_novelty_score,
    parse_image_collage_response,
    rank_images_by_novelty,
)
from geoeventfusion.models.visual import VisualImage


# ── compute_novelty_score ─────────────────────────────────────────────────────────

class TestComputeNoveltyScore:
    def test_zero_appearances_returns_one(self):
        """Image never seen before must score exactly 1.0."""
        score = compute_novelty_score(0)
        assert score == pytest.approx(1.0)

    def test_formula_correctness(self):
        """Score must equal 1.0 / (1.0 + log(1 + count))."""
        for count in [1, 5, 10, 100, 1000]:
            expected = 1.0 / (1.0 + math.log(1 + count))
            assert compute_novelty_score(count) == pytest.approx(expected, rel=1e-9)

    def test_score_decreases_with_more_appearances(self):
        """Higher web appearance count must produce a lower novelty score."""
        assert compute_novelty_score(1) > compute_novelty_score(10)
        assert compute_novelty_score(10) > compute_novelty_score(100)
        assert compute_novelty_score(100) > compute_novelty_score(1000)

    def test_score_in_range_zero_to_one(self):
        """Novelty score must always be in [0.0, 1.0]."""
        for count in [0, 1, 10, 100, 10000]:
            score = compute_novelty_score(count)
            assert 0.0 <= score <= 1.0

    def test_negative_count_treated_as_zero(self):
        """Negative web appearance count must be treated as 0."""
        score_neg = compute_novelty_score(-5)
        score_zero = compute_novelty_score(0)
        assert score_neg == pytest.approx(score_zero)

    def test_large_count_approaches_zero(self):
        """Very large count must produce a score approaching 0 (but > 0)."""
        score = compute_novelty_score(1_000_000)
        assert score > 0.0
        assert score < 0.1  # should be very small


# ── check_staleness ───────────────────────────────────────────────────────────────

class TestCheckStaleness:
    def test_stale_image_detected(self):
        """Image captured 100 hours before article must be flagged stale."""
        result = check_staleness(
            exif_capture_date="2024-01-10T00:00:00",
            article_published_at="2024-01-14T04:00:00",  # 100h later
            threshold_hours=72,
        )
        assert result is True

    def test_fresh_image_not_stale(self):
        """Image captured 24 hours before article must not be flagged."""
        result = check_staleness(
            exif_capture_date="2024-01-15T10:00:00",
            article_published_at="2024-01-16T10:00:00",  # exactly 24h later
            threshold_hours=72,
        )
        assert result is False

    def test_exactly_at_threshold_not_stale(self):
        """Image at exactly threshold_hours is not stale (must be > threshold)."""
        result = check_staleness(
            exif_capture_date="2024-01-12T12:00:00",
            article_published_at="2024-01-15T12:00:00",  # exactly 72h
            threshold_hours=72,
        )
        assert result is False

    def test_missing_exif_date_returns_false(self):
        """Missing exif_capture_date must return False without raising."""
        result = check_staleness(
            exif_capture_date=None,
            article_published_at="2024-01-15",
            threshold_hours=72,
        )
        assert result is False

    def test_missing_article_date_returns_false(self):
        """Missing article_published_at must return False without raising."""
        result = check_staleness(
            exif_capture_date="2024-01-10",
            article_published_at=None,
            threshold_hours=72,
        )
        assert result is False

    def test_both_dates_missing_returns_false(self):
        """Both dates missing must return False."""
        result = check_staleness(None, None, threshold_hours=72)
        assert result is False

    def test_invalid_date_format_returns_false(self):
        """Unparseable date strings must return False without raising."""
        result = check_staleness(
            exif_capture_date="not-a-date",
            article_published_at="also-not-a-date",
            threshold_hours=72,
        )
        assert result is False

    def test_z_suffix_iso_dates_handled(self):
        """ISO dates with Z suffix must be parsed correctly."""
        result = check_staleness(
            exif_capture_date="2024-01-10T00:00:00Z",
            article_published_at="2024-01-14T12:00:00Z",  # 108h later
            threshold_hours=72,
        )
        assert result is True

    def test_custom_threshold_hours(self):
        """Custom threshold_hours must be respected."""
        # 50h gap, threshold is 24h → stale
        result = check_staleness(
            exif_capture_date="2024-01-01T00:00:00",
            article_published_at="2024-01-03T02:00:00",  # 50h later
            threshold_hours=24,
        )
        assert result is True


# ── parse_image_collage_response ──────────────────────────────────────────────────

class TestParseImageCollageResponse:
    def _make_response(self, images: list) -> Dict[str, Any]:
        return {"images": images}

    def _make_raw_image(self, **kwargs) -> dict:
        defaults = {
            "url": "https://example.com/img.jpg",
            "pageurl": "https://example.com/article",
            "title": "Test article",
            "imagetags": "military,protest",
            "imagewebcount": 5,
            "priorwebUrls": ["https://example.com/prior1"],
            "exifcapturedate": None,
            "seendate": "2024-01-15",
        }
        defaults.update(kwargs)
        return defaults

    def test_happy_path_returns_visual_images(self):
        """Valid API response must return a list of VisualImage objects."""
        response = self._make_response([self._make_raw_image(), self._make_raw_image(
            url="https://example.com/img2.jpg"
        )])
        images = parse_image_collage_response(response)

        assert len(images) == 2
        assert all(isinstance(img, VisualImage) for img in images)

    def test_novelty_score_computed(self):
        """novelty_score must be computed from web_appearance_count."""
        response = self._make_response([self._make_raw_image(imagewebcount=0)])
        images = parse_image_collage_response(response)

        assert images[0].novelty_score == pytest.approx(1.0)

    def test_imagetags_parsed_from_comma_string(self):
        """Comma-separated imagetags string must be split into a list."""
        response = self._make_response([self._make_raw_image(imagetags="military,protest,fire")])
        images = parse_image_collage_response(response)

        assert "military" in images[0].imagetags
        assert "protest" in images[0].imagetags
        assert "fire" in images[0].imagetags

    def test_imagetags_parsed_from_list(self):
        """imagetags provided as a list must be preserved."""
        response = self._make_response([self._make_raw_image(imagetags=["military", "protest"])])
        images = parse_image_collage_response(response)

        assert images[0].imagetags == ["military", "protest"]

    def test_staleness_warning_set_when_stale(self):
        """Stale image (EXIF date > threshold before article) must set staleness_warning."""
        response = self._make_response([self._make_raw_image(
            exifcapturedate="2024-01-10T00:00:00",
            seendate="2024-01-14T12:00:00",  # 108h later
        )])
        images = parse_image_collage_response(response, staleness_threshold_hours=72)

        assert images[0].staleness_warning is True

    def test_staleness_warning_false_for_fresh_image(self):
        """Fresh image must not set staleness_warning."""
        response = self._make_response([self._make_raw_image(
            exifcapturedate="2024-01-15T00:00:00",
            seendate="2024-01-15T12:00:00",  # only 12h later
        )])
        images = parse_image_collage_response(response, staleness_threshold_hours=72)

        assert images[0].staleness_warning is False

    def test_prior_web_urls_capped_at_six(self):
        """prior_web_urls must be capped at 6 entries."""
        prior_urls = [f"https://example.com/{i}" for i in range(10)]
        response = self._make_response([self._make_raw_image(priorwebUrls=prior_urls)])
        images = parse_image_collage_response(response)

        assert len(images[0].prior_web_urls) <= 6

    def test_missing_url_image_skipped(self):
        """Images without a URL must be skipped."""
        response = self._make_response([
            {"url": "", "pageurl": "", "title": "No URL image"},
            self._make_raw_image(url="https://example.com/valid.jpg"),
        ])
        images = parse_image_collage_response(response)

        assert len(images) == 1
        assert images[0].url == "https://example.com/valid.jpg"

    def test_empty_response_returns_empty_list(self):
        """None or empty API response must return an empty list without raising."""
        assert parse_image_collage_response(None) == []
        assert parse_image_collage_response({}) == []
        assert parse_image_collage_response({"images": []}) == []

    def test_malformed_response_type_returns_empty(self):
        """Non-dict response must return an empty list."""
        assert parse_image_collage_response("not a dict") == []  # type: ignore[arg-type]
        assert parse_image_collage_response([]) == []  # type: ignore[arg-type]

    def test_non_dict_image_entries_skipped(self):
        """Non-dict entries in the images list must be skipped."""
        response = self._make_response([
            "not a dict",
            None,
            self._make_raw_image(),
        ])
        images = parse_image_collage_response(response)
        assert len(images) == 1

    def test_artlist_key_fallback(self):
        """If 'images' key is missing, 'artlist' key must be tried as fallback."""
        response = {"artlist": [self._make_raw_image()]}
        images = parse_image_collage_response(response)
        assert len(images) == 1

    def test_imageurl_field_fallback(self):
        """If 'url' key is missing, 'imageurl' key must be used."""
        raw = self._make_raw_image()
        raw.pop("url")
        raw["imageurl"] = "https://example.com/fallback.jpg"
        response = self._make_response([raw])
        images = parse_image_collage_response(response)

        assert len(images) == 1
        assert images[0].url == "https://example.com/fallback.jpg"

    def test_web_count_invalid_type_defaults_to_zero(self):
        """Non-numeric imagewebcount must default to 0."""
        response = self._make_response([self._make_raw_image(imagewebcount="invalid")])
        images = parse_image_collage_response(response)
        assert images[0].web_appearance_count == 0

    def test_article_url_and_title_populated(self):
        """article_url and article_title must be populated from response fields."""
        response = self._make_response([self._make_raw_image(
            pageurl="https://example.com/article",
            title="Test headline"
        )])
        images = parse_image_collage_response(response)

        assert images[0].article_url == "https://example.com/article"
        assert images[0].article_title == "Test headline"


# ── rank_images_by_novelty ────────────────────────────────────────────────────────

class TestRankImagesByNovelty:
    def _make_image(self, novelty: float) -> VisualImage:
        return VisualImage(url=f"https://example.com/{novelty}.jpg", novelty_score=novelty)

    def test_sorted_descending_by_novelty(self):
        """Images must be returned with highest novelty first."""
        images = [
            self._make_image(0.3),
            self._make_image(0.9),
            self._make_image(0.6),
        ]
        ranked = rank_images_by_novelty(images)

        assert ranked[0].novelty_score == pytest.approx(0.9)
        assert ranked[1].novelty_score == pytest.approx(0.6)
        assert ranked[2].novelty_score == pytest.approx(0.3)

    def test_empty_list_returns_empty(self):
        """Empty input must return an empty list."""
        assert rank_images_by_novelty([]) == []

    def test_single_image_returned_unchanged(self):
        """Single image must be returned in a list unchanged."""
        images = [self._make_image(0.75)]
        ranked = rank_images_by_novelty(images)
        assert len(ranked) == 1
        assert ranked[0].novelty_score == pytest.approx(0.75)

    def test_does_not_modify_original_list(self):
        """rank_images_by_novelty must return a new sorted list, not modify in place."""
        images = [self._make_image(0.3), self._make_image(0.9)]
        original_order = [img.novelty_score for img in images]
        _ = rank_images_by_novelty(images)
        assert [img.novelty_score for img in images] == original_order
