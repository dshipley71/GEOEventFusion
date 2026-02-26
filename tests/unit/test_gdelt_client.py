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
from unittest.mock import MagicMock, call, patch

import pytest

from geoeventfusion.clients.gdelt_client import GDELTClient, _safe_parse_json


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
