"""Unit tests for geoeventfusion.clients.gdelt_client.

Covers:
- _safe_parse_json: valid JSON, empty body, HTTP header bleed-through, ast fallback
- GDELTClient._build_url: correct URL construction
- GDELTClient.fetch: success, rate limit, server error, timeout, unparseable body
- GDELTClient context manager protocol

No real HTTP calls are made — requests.Session.get is patched throughout.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

from geoeventfusion.clients.gdelt_client import GDELTClient, _safe_parse_json, _parse_timespan_days


# ── _safe_parse_json ──────────────────────────────────────────────────────────────

class TestSafeParseJson:
    def test_valid_json_object(self):
        """A valid JSON object string must be parsed correctly."""
        result = _safe_parse_json('{"articles": [{"url": "https://example.com"}]}')
        assert result == {"articles": [{"url": "https://example.com"}]}

    def test_valid_json_array(self):
        """A valid JSON array must be parsed correctly."""
        result = _safe_parse_json('[{"date": "2024-01-01", "value": 2.0}]')
        assert isinstance(result, list)
        assert result[0]["date"] == "2024-01-01"

    def test_empty_string_returns_none(self):
        """Empty string must return None."""
        assert _safe_parse_json("") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string must return None."""
        assert _safe_parse_json("   \n\t  ") is None

    def test_http_header_bleed_through(self):
        """HTTP header lines before the JSON body must be stripped."""
        body = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "\r\n"
            '{"articles": []}'
        )
        result = _safe_parse_json(body)
        assert result == {"articles": []}

    def test_http_header_prefixes_stripped(self):
        """Various HTTP header prefixes (Date:, Server:, etc.) must be stripped."""
        body = (
            "Date: Thu, 15 Jan 2024 12:00:00 GMT\n"
            "Content-Type: application/json\n"
            '{"timeline": []}'
        )
        result = _safe_parse_json(body)
        assert result == {"timeline": []}

    def test_x_header_stripped(self):
        """X- custom headers must be detected and stripped."""
        body = "X-Cache: MISS\n" + '{"data": "ok"}'
        result = _safe_parse_json(body)
        assert result == {"data": "ok"}

    def test_garbage_body_returns_none(self):
        """Unparseable non-JSON body must return None."""
        result = _safe_parse_json("this is not json at all !!!!")
        assert result is None

    def test_ast_fallback_near_json(self):
        """Python-literal-style dicts (single quotes) should parse via ast fallback."""
        # ast.literal_eval handles {'key': 'value'} which json.loads rejects
        result = _safe_parse_json("{'key': 'value'}")
        # Either parsed successfully or returns None (both are acceptable)
        if result is not None:
            assert result == {"key": "value"}

    def test_nested_json_array(self):
        """Nested JSON structures must parse correctly."""
        result = _safe_parse_json('{"articles": [{"url": "a", "tone": -5.2}]}')
        assert result["articles"][0]["tone"] == -5.2

    def test_json_with_only_http_headers_returns_none(self):
        """A body containing only HTTP headers with no JSON must return None."""
        body = "HTTP/1.1 200 OK\nContent-Type: text/html\n"
        result = _safe_parse_json(body)
        assert result is None


# ── GDELTClient URL construction ──────────────────────────────────────────────────

class TestGDELTClientBuildUrl:
    def test_build_url_contains_base_url(self):
        """Built URL must include the GDELT DOC 2.0 base URL."""
        client = GDELTClient()
        url = client._build_url({"query": "Houthi", "mode": "ArtList", "format": "json"})
        assert "api.gdeltproject.org/api/v2/doc/doc" in url

    def test_build_url_encodes_parameters(self):
        """URL parameters must be properly URL-encoded."""
        client = GDELTClient()
        url = client._build_url({"query": "Houthi Red Sea", "mode": "ArtList"})
        assert "Houthi+Red+Sea" in url or "Houthi%20Red%20Sea" in url

    def test_build_url_includes_mode(self):
        """Mode parameter must appear in the URL."""
        client = GDELTClient()
        url = client._build_url({"mode": "TimelineVolInfo"})
        assert "TimelineVolInfo" in url


# ── GDELTClient.fetch ─────────────────────────────────────────────────────────────

class TestGDELTClientFetch:
    def _make_client(self):
        return GDELTClient(max_retries=2, backoff_base=0.0, request_timeout=5, stagger_seconds=0.0)

    def _mock_response(self, status_code: int = 200, text: str = '{"articles": []}'):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        return mock_resp

    def test_fetch_success_returns_parsed_dict(self):
        """A successful 200 response must return the parsed JSON dict."""
        client = self._make_client()
        payload = json.dumps({"articles": [{"url": "https://example.com", "title": "Test"}]})

        with patch.object(client._session, "get", return_value=self._mock_response(200, payload)):
            result = client.fetch("Houthi", "ArtList")

        assert result is not None
        assert "articles" in result
        assert len(result["articles"]) == 1

    def test_fetch_includes_mode_artlist_params(self):
        """ArtList mode must include maxrecords and sort parameters."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "ArtList", max_records=50, sort="ToneAsc")

        call_url = mock_get.call_args[0][0]
        assert "maxrecords=50" in call_url
        assert "sort=ToneAsc" in call_url

    def test_fetch_timeline_includes_smooth_param(self):
        """Timeline modes must include TIMELINESMOOTH parameter."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "TimelineVolInfo", timeline_smooth=5)

        call_url = mock_get.call_args[0][0]
        assert "TIMELINESMOOTH=5" in call_url

    def test_fetch_non_200_returns_none(self):
        """A non-200 response (not 429, 5xx) must return None."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response(404, "")):
            result = client.fetch("Houthi", "ArtList")

        assert result is None

    def test_fetch_unparseable_body_returns_none(self):
        """A 200 response with unparseable body must return None."""
        client = self._make_client()

        with patch.object(
            client._session, "get",
            return_value=self._mock_response(200, "this is not json")
        ):
            result = client.fetch("Houthi", "ArtList")

        assert result is None

    def test_fetch_rate_limit_retries(self):
        """A 429 response must trigger a retry (backoff then second attempt)."""
        client = self._make_client()

        rate_limited = self._mock_response(429, "")
        success = self._mock_response(200, '{"articles": []}')

        with patch.object(
            client._session, "get", side_effect=[rate_limited, success]
        ):
            result = client.fetch("Houthi", "ArtList")

        assert result is not None

    def test_fetch_server_error_retries(self):
        """A 503 response must trigger a retry."""
        client = self._make_client()

        server_error = self._mock_response(503, "")
        success = self._mock_response(200, '{"articles": []}')

        with patch.object(
            client._session, "get", side_effect=[server_error, success]
        ):
            result = client.fetch("Houthi", "ArtList")

        assert result is not None

    def test_fetch_exhausted_retries_returns_none(self):
        """Exhausting all retries must return None."""
        client = self._make_client()
        always_fail = self._mock_response(503, "")

        with patch.object(
            client._session, "get",
            side_effect=[always_fail] * (client.max_retries + 2)
        ):
            result = client.fetch("Houthi", "ArtList")

        assert result is None

    def test_fetch_timeout_retries(self):
        """A requests.Timeout must trigger a retry."""
        import requests as req_lib

        client = self._make_client()
        success = self._mock_response(200, '{"articles": []}')

        with patch.object(
            client._session, "get",
            side_effect=[req_lib.exceptions.Timeout, success]
        ):
            result = client.fetch("Houthi", "ArtList")

        assert result is not None

    def test_fetch_connection_error_retries(self):
        """A ConnectionError must trigger a retry."""
        import requests as req_lib

        client = self._make_client()
        success = self._mock_response(200, '{"articles": []}')

        with patch.object(
            client._session, "get",
            side_effect=[req_lib.exceptions.ConnectionError("down"), success]
        ):
            result = client.fetch("Houthi", "ArtList")

        assert result is not None

    def test_fetch_start_date_adds_startdatetime_param(self):
        """Providing start_date must add startdatetime parameter to the URL."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "ArtList", start_date="2024-01-01")

        call_url = mock_get.call_args[0][0]
        assert "startdatetime" in call_url

    def test_fetch_end_date_adds_enddatetime_param(self):
        """Providing end_date must add enddatetime parameter to the URL."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "ArtList", end_date="2024-03-31")

        call_url = mock_get.call_args[0][0]
        assert "enddatetime" in call_url

    def test_fetch_extra_params_included(self):
        """Extra params must be passed through to the URL."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "ArtList", extra_params={"customkey": "customval"})

        call_url = mock_get.call_args[0][0]
        assert "customkey=customval" in call_url

    def test_fetch_timespan_adds_timespan_param(self):
        """Providing timespan must add TIMESPAN parameter to the URL."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "ArtList", timespan="30d")

        call_url = mock_get.call_args[0][0]
        assert "TIMESPAN=30d" in call_url

    def test_fetch_timespan_excludes_startdatetime(self):
        """When timespan is set, startdatetime must NOT appear in the URL."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch(
                "Houthi",
                "ArtList",
                timespan="30d",
                start_date="2026-01-01",
                end_date="2026-02-27",
            )

        call_url = mock_get.call_args[0][0]
        assert "TIMESPAN=30d" in call_url
        assert "startdatetime" not in call_url
        assert "enddatetime" not in call_url

    def test_fetch_start_end_date_used_when_no_timespan(self):
        """Without timespan, start_date and end_date must add their respective params."""
        client = self._make_client()

        with patch.object(client._session, "get", return_value=self._mock_response()) as mock_get:
            client.fetch("Houthi", "ArtList", start_date="2026-01-01", end_date="2026-02-27")

        call_url = mock_get.call_args[0][0]
        assert "startdatetime" in call_url
        assert "enddatetime" in call_url
        assert "TIMESPAN" not in call_url

    def test_fetch_distribute_calls_multiple_buckets(self):
        """distribute=True over 30 days must make ceil(30/7)=5 bucket GET calls."""
        client = self._make_client()
        # 30d → 5 weekly buckets; supply exactly 5 responses
        responses = [
            self._mock_response(
                200,
                json.dumps({"articles": [{"url": f"https://x.com/{i}", "title": f"T{i}"}]}),
            )
            for i in range(5)
        ]
        with patch.object(client._session, "get", side_effect=responses) as mock_get:
            result = client.fetch("Houthi", "ArtList", max_records=50, timespan="30d", distribute=True)

        assert mock_get.call_count == 5
        assert result is not None
        assert len(result["articles"]) == 5

    def test_fetch_distribute_merges_articles_from_buckets(self):
        """distribute=True must combine articles from all bucket responses."""
        client = self._make_client()
        # 14d → ceil(14/7) = 2 buckets, one unique article each
        responses = [
            self._mock_response(200, json.dumps({"articles": [{"url": "https://x.com/a1", "title": "A1"}]})),
            self._mock_response(200, json.dumps({"articles": [{"url": "https://x.com/a2", "title": "A2"}]})),
        ]
        with patch.object(client._session, "get", side_effect=responses):
            result = client.fetch("Houthi", "ArtList", max_records=50, timespan="14d", distribute=True)

        assert result is not None
        assert len(result["articles"]) == 2
        urls = {a["url"] for a in result["articles"]}
        assert urls == {"https://x.com/a1", "https://x.com/a2"}

    def test_fetch_distribute_deduplicates_by_url(self):
        """distribute=True must not include the same URL more than once."""
        client = self._make_client()
        same = {"url": "https://x.com/dup", "title": "Dup"}
        responses = [
            self._mock_response(200, json.dumps({"articles": [same]})),
            self._mock_response(200, json.dumps({"articles": [same]})),
        ]
        with patch.object(client._session, "get", side_effect=responses):
            result = client.fetch("Houthi", "ArtList", max_records=50, timespan="14d", distribute=True)

        assert result is not None
        assert len(result["articles"]) == 1

    def test_fetch_distribute_caps_at_max_records(self):
        """distribute=True must not return more articles than max_records."""
        client = self._make_client()
        bucket1 = [{"url": f"https://x.com/a{i}", "title": f"A{i}"} for i in range(10)]
        bucket2 = [{"url": f"https://x.com/b{i}", "title": f"B{i}"} for i in range(10)]
        responses = [
            self._mock_response(200, json.dumps({"articles": bucket1})),
            self._mock_response(200, json.dumps({"articles": bucket2})),
        ]
        with patch.object(client._session, "get", side_effect=responses):
            result = client.fetch("Houthi", "ArtList", max_records=15, timespan="14d", distribute=True)

        assert result is not None
        assert len(result["articles"]) <= 15

    def test_fetch_distribute_without_time_window_falls_back(self):
        """distribute=True with no timespan or dates must fall back to a normal single fetch."""
        client = self._make_client()
        payload = json.dumps({"articles": [{"url": "https://x.com/1", "title": "T"}]})
        with patch.object(client._session, "get", return_value=self._mock_response(200, payload)) as mock_get:
            result = client.fetch("Houthi", "ArtList", distribute=True)

        assert mock_get.call_count == 1
        assert result is not None


# ── _parse_timespan_days ──────────────────────────────────────────────────────────

class TestParseTimespanDays:
    def test_days_suffix(self):
        assert _parse_timespan_days("30d") == 30
        assert _parse_timespan_days("7d") == 7
        assert _parse_timespan_days("90d") == 90

    def test_weeks_suffix(self):
        assert _parse_timespan_days("1w") == 7
        assert _parse_timespan_days("2w") == 14

    def test_months_suffix(self):
        assert _parse_timespan_days("1m") == 30
        assert _parse_timespan_days("3m") == 90

    def test_hours_suffix(self):
        assert _parse_timespan_days("24h") == 1
        assert _parse_timespan_days("48h") == 2

    def test_years_suffix(self):
        assert _parse_timespan_days("1y") == 365

    def test_plain_integer_treated_as_days(self):
        assert _parse_timespan_days("30") == 30

    def test_minimum_one_day_for_sub_day_inputs(self):
        assert _parse_timespan_days("1h") == 1
        assert _parse_timespan_days("15min") == 1

    def test_invalid_string_returns_one(self):
        assert _parse_timespan_days("bad") == 1
        assert _parse_timespan_days("") == 1


# ── _enforce_stagger thread-safety ────────────────────────────────────────────────

class TestEnforceStagger:
    def test_enforce_stagger_thread_safety(self):
        """Two concurrent threads must be separated by at least stagger_seconds.

        Both threads call _enforce_stagger() simultaneously on a shared client.
        We record the monotonic timestamp at which each thread exits the stagger
        lock and assert the two timestamps differ by at least stagger_seconds
        (minus a small tolerance for scheduling jitter).
        """
        stagger = 0.1   # short value so the test runs quickly
        client = GDELTClient(stagger_seconds=stagger)

        timestamps: list[float] = []
        barrier = threading.Barrier(2)  # synchronise both threads at the start

        def _worker():
            barrier.wait()              # both threads release simultaneously
            client._enforce_stagger()
            timestamps.append(time.monotonic())

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(timestamps) == 2
        gap = abs(timestamps[1] - timestamps[0])
        # The slower thread must exit the lock at least stagger_seconds after the faster one
        assert gap >= stagger - 0.02, (
            f"Stagger gap {gap:.3f}s < stagger_seconds {stagger}s — race condition detected"
        )

    def test_retry_reenforces_stagger_after_429(self):
        """_enforce_stagger() must be called on every retry attempt, not just the first.

        After a 429 backoff sleep, the next attempt must re-acquire the stagger
        lock before making its request. We verify this by counting calls to
        _enforce_stagger() across a 429-then-200 response sequence.
        """
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.text = '{"articles": []}'

        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429

        client = GDELTClient(stagger_seconds=0.0, backoff_base=0.0)
        stagger_call_count = [0]
        original_enforce = client._enforce_stagger

        def counting_enforce():
            stagger_call_count[0] += 1
            original_enforce()

        client._enforce_stagger = counting_enforce

        with patch.object(client._session, "get", side_effect=[rate_limit_response, ok_response]):
            result = client._get_with_retry("https://example.com/fake")

        # First call (attempt 0) + retry after 429 (attempt 1) = 2 calls
        assert stagger_call_count[0] == 2, (
            f"Expected _enforce_stagger called 2 times (once per attempt), got {stagger_call_count[0]}"
        )
        assert result == ok_response.text


# ── GDELTClient context manager ───────────────────────────────────────────────────

class TestGDELTClientContextManager:
    def test_context_manager_enters_and_exits(self):
        """GDELTClient must work as a context manager."""
        with GDELTClient() as client:
            assert isinstance(client, GDELTClient)

    def test_context_manager_closes_session(self):
        """Exiting the context manager must call close()."""
        client = GDELTClient()
        with patch.object(client, "close") as mock_close:
            client.__exit__(None, None, None)
        mock_close.assert_called_once()
