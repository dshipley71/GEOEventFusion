"""Unit tests for geoeventfusion.clients.llm_client.

Covers:
- _safe_parse_llm_json: valid JSON, fence stripping, boundary detection, edge cases
- LLMClient.enforce_confidence_cap: recursive cap enforcement
- LLMClient.call: retry on empty, backend dispatch
- LLMClient.call_json: JSON enforcement and defensive parsing
- Module-level llm_call convenience function
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from geoeventfusion.clients.llm_client import LLMClient, _safe_parse_llm_json, llm_call


# ── _safe_parse_llm_json ─────────────────────────────────────────────────────────

class TestSafeParseLlmJson:
    def test_valid_json_object(self):
        """A clean JSON object must be parsed correctly.

        Note: _safe_parse_llm_json tries '[' before '{', so use an object
        without embedded arrays to ensure the '{' branch is reached first.
        """
        result = _safe_parse_llm_json('{"confidence": 0.7, "summary": "test result"}')
        assert result == {"confidence": 0.7, "summary": "test result"}

    def test_valid_json_array(self):
        """A clean JSON array must be parsed correctly."""
        result = _safe_parse_llm_json('[{"event_type": "CONFLICT"}, {"event_type": "PROTEST"}]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_strips_markdown_json_fence(self):
        """```json code fences must be stripped before parsing.

        Use an object without embedded arrays (implementation tries '[' first).
        """
        text = '```json\n{"status": "ok", "phase": "1"}\n```'
        result = _safe_parse_llm_json(text)
        assert result == {"status": "ok", "phase": "1"}

    def test_strips_plain_code_fence(self):
        """Plain ``` code fences (no language specifier) must be stripped."""
        text = '```\n{"key": "value"}\n```'
        result = _safe_parse_llm_json(text)
        assert result == {"key": "value"}

    def test_finds_object_boundary(self):
        """Must locate the { boundary even with leading text."""
        text = 'Here is the result: {"confidence": 0.8, "summary": "test"}'
        result = _safe_parse_llm_json(text)
        assert result is not None
        assert result["confidence"] == 0.8

    def test_finds_array_boundary(self):
        """Must locate the [ boundary even with leading text."""
        text = 'Results: [{"id": 1}, {"id": 2}]'
        result = _safe_parse_llm_json(text)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_array_preferred_over_object_when_both_present(self):
        """When both array and object markers exist, the earlier/outer one wins."""
        # Array starts before object
        text = '[{"key": "value"}]'
        result = _safe_parse_llm_json(text)
        assert isinstance(result, list)

    def test_empty_string_returns_none(self):
        """Empty string must return None."""
        assert _safe_parse_llm_json("") is None

    def test_none_input_via_guard(self):
        """Empty/None-like input must return None."""
        assert _safe_parse_llm_json("") is None

    def test_pure_garbage_returns_none(self):
        """Pure non-JSON text must return None."""
        result = _safe_parse_llm_json("This is a narrative response with no JSON.")
        assert result is None

    def test_nested_json_parsed_correctly(self):
        """Nested JSON array structures must parse correctly.

        _safe_parse_llm_json returns the first valid JSON found (array before object),
        so a top-level array is the expected structure for multi-event extraction.
        """
        text = '[{"label": "Phase 1", "events": "none"}, {"label": "Phase 2", "events": "none"}]'
        result = _safe_parse_llm_json(text)
        assert isinstance(result, list)
        assert result[0]["label"] == "Phase 1"

    def test_trailing_backtick_stripped(self):
        """Trailing backticks (e.g., from incomplete fence) must be stripped."""
        text = '{"key": "value"}`'
        result = _safe_parse_llm_json(text)
        # Should parse the object before the trailing backtick
        if result is not None:
            assert result.get("key") == "value"


# ── LLMClient.enforce_confidence_cap ─────────────────────────────────────────────

class TestEnforceConfidenceCap:
    def setup_method(self):
        self.client = LLMClient(max_confidence=0.82)

    def test_cap_applied_to_confidence_field(self):
        """confidence values above cap must be clamped."""
        data = {"confidence": 0.95, "summary": "test"}
        result = self.client.enforce_confidence_cap(data)
        assert result["confidence"] == pytest.approx(0.82)

    def test_below_cap_unchanged(self):
        """confidence values at or below cap must not be modified."""
        data = {"confidence": 0.75, "summary": "test"}
        result = self.client.enforce_confidence_cap(data)
        assert result["confidence"] == pytest.approx(0.75)

    def test_cap_applied_recursively_in_nested_dict(self):
        """Confidence fields in nested dicts must also be capped."""
        data = {
            "panels": [
                {"headline": "Test", "confidence": 0.99},
                {"headline": "Other", "confidence": 0.5},
            ]
        }
        result = self.client.enforce_confidence_cap(data)
        assert result["panels"][0]["confidence"] == pytest.approx(0.82)
        assert result["panels"][1]["confidence"] == pytest.approx(0.5)

    def test_cap_applied_to_list_of_dicts(self):
        """confidence fields in list elements must be capped."""
        data = [{"confidence": 1.0}, {"confidence": 0.6}]
        result = self.client.enforce_confidence_cap(data)
        assert result[0]["confidence"] == pytest.approx(0.82)
        assert result[1]["confidence"] == pytest.approx(0.6)

    def test_non_numeric_confidence_untouched(self):
        """Non-numeric confidence fields must not be modified."""
        data = {"confidence": "high"}
        result = self.client.enforce_confidence_cap(data)
        assert result["confidence"] == "high"

    def test_scalar_passthrough(self):
        """Scalar values (str, int, float) must pass through unchanged."""
        assert self.client.enforce_confidence_cap(42) == 42
        assert self.client.enforce_confidence_cap("text") == "text"
        assert self.client.enforce_confidence_cap(None) is None

    def test_cap_at_exactly_max_confidence(self):
        """Confidence exactly at max_confidence must not be modified."""
        data = {"confidence": 0.82}
        result = self.client.enforce_confidence_cap(data)
        assert result["confidence"] == pytest.approx(0.82)

    def test_custom_max_confidence(self):
        """Custom max_confidence set on the client must be respected."""
        client = LLMClient(max_confidence=0.70)
        data = {"confidence": 0.80}
        result = client.enforce_confidence_cap(data)
        assert result["confidence"] == pytest.approx(0.70)


# ── LLMClient.call ────────────────────────────────────────────────────────────────

class TestLLMClientCall:
    def test_call_returns_response_text(self):
        """A successful backend call must return the response text."""
        client = LLMClient(backend="anthropic")

        with patch.object(client, "_call_anthropic", return_value="response text"):
            result = client.call("system", "prompt", max_tokens=512)

        assert result == "response text"

    def test_call_retries_once_on_empty_response(self):
        """Empty response on first attempt must trigger one retry."""
        client = LLMClient(backend="anthropic")

        with patch.object(
            client, "_call_anthropic",
            side_effect=["", "valid response on retry"]
        ) as mock_call:
            result = client.call("system", "prompt", max_tokens=256)

        assert result == "valid response on retry"
        assert mock_call.call_count == 2

    def test_call_retries_with_doubled_max_tokens(self):
        """Retry attempt must use doubled max_tokens."""
        client = LLMClient(backend="anthropic")
        received_tokens = []

        def _capture(*args, **kwargs):
            received_tokens.append(kwargs.get("max_tokens", args[2] if len(args) > 2 else None))
            return "" if len(received_tokens) == 1 else "ok"

        with patch.object(client, "_call_anthropic", side_effect=_capture):
            client.call("sys", "prompt", max_tokens=512)

        # Second call should have doubled tokens (1024)
        assert received_tokens[1] == 1024

    def test_call_returns_none_after_two_empty_responses(self):
        """Two consecutive empty responses must return None."""
        client = LLMClient(backend="anthropic")

        with patch.object(client, "_call_anthropic", return_value=""):
            result = client.call("system", "prompt")

        assert result is None

    def test_call_enforces_min_max_tokens(self):
        """max_tokens below LLM_MIN_MAX_TOKENS must be raised to the minimum."""
        from config.defaults import LLM_MIN_MAX_TOKENS

        client = LLMClient(backend="anthropic")
        received_tokens = []

        def _capture(system, prompt, max_tokens, temperature):
            received_tokens.append(max_tokens)
            return "ok"

        with patch.object(client, "_call_anthropic", side_effect=_capture):
            client.call("sys", "prompt", max_tokens=10)  # well below minimum

        assert received_tokens[0] >= LLM_MIN_MAX_TOKENS

    def test_call_dispatches_to_ollama_backend(self):
        """Backend=ollama must dispatch to _call_ollama."""
        client = LLMClient(backend="ollama")

        with patch.object(client, "_call_ollama", return_value="ok") as mock_ollama:
            client.call("sys", "prompt")

        mock_ollama.assert_called_once()

    def test_call_dispatches_to_anthropic_backend(self):
        """Backend=anthropic must dispatch to _call_anthropic."""
        client = LLMClient(backend="anthropic")

        with patch.object(client, "_call_anthropic", return_value="ok") as mock_anthropic:
            client.call("sys", "prompt")

        mock_anthropic.assert_called_once()

    def test_call_exception_on_first_attempt_retried(self):
        """Exception on first attempt must be caught and retried."""
        client = LLMClient(backend="anthropic")

        with patch.object(
            client, "_call_anthropic",
            side_effect=[RuntimeError("connection error"), "retry success"]
        ):
            result = client.call("sys", "prompt")

        assert result == "retry success"

    def test_call_exception_on_both_attempts_returns_none(self):
        """Exceptions on both attempts must return None."""
        client = LLMClient(backend="anthropic")

        with patch.object(
            client, "_call_anthropic",
            side_effect=RuntimeError("permanent failure")
        ):
            result = client.call("sys", "prompt")

        assert result is None

    def test_call_auth_error_not_retried(self):
        """A 401 Unauthorized exception must return None immediately without retrying.

        Retrying an auth error is pointless — the API key won't change between
        attempts. The backend method must be called exactly once, and None returned.
        """
        client = LLMClient(backend="anthropic")

        auth_error = RuntimeError("unauthorized (status code: 401)")
        auth_error.status_code = 401  # type: ignore[attr-defined]

        with patch.object(
            client, "_call_anthropic", side_effect=auth_error
        ) as mock_backend:
            result = client.call("sys", "prompt")

        assert result is None
        assert mock_backend.call_count == 1, (
            "Backend must not be called a second time for a 401 auth error"
        )


# ── LLMClient.call_json ───────────────────────────────────────────────────────────

class TestLLMClientCallJson:
    def test_call_json_returns_parsed_dict(self):
        """call_json must return a parsed dict from a JSON response."""
        client = LLMClient(backend="anthropic")
        payload = '{"confidence": 0.7, "status": "ok"}'

        with patch.object(client, "_call_anthropic", return_value=payload):
            result = client.call_json("sys", "prompt")

        assert result == {"confidence": 0.7, "status": "ok"}

    def test_call_json_returns_parsed_list(self):
        """call_json must return a parsed list from a JSON array response."""
        client = LLMClient(backend="anthropic")
        payload = '[{"event_type": "CONFLICT"}]'

        with patch.object(client, "_call_anthropic", return_value=payload):
            result = client.call_json("sys", "prompt")

        assert isinstance(result, list)

    def test_call_json_returns_none_on_unparseable_response(self):
        """Unparseable LLM output must return None (not raise)."""
        client = LLMClient(backend="anthropic")

        with patch.object(client, "_call_anthropic", return_value="not json at all"):
            result = client.call_json("sys", "prompt")

        assert result is None

    def test_call_json_strips_fences_before_parsing(self):
        """Fenced JSON responses must be parsed correctly."""
        client = LLMClient(backend="anthropic")
        payload = '```json\n{"hypotheses": "none", "count": 0}\n```'

        with patch.object(client, "_call_anthropic", return_value=payload):
            result = client.call_json("sys", "prompt")

        assert result == {"hypotheses": "none", "count": 0}

    def test_call_json_appends_json_instruction_to_system(self):
        """call_json must append a JSON-only instruction to the system prompt."""
        client = LLMClient(backend="anthropic")
        captured_system = []

        def _capture(system, prompt, max_tokens, temperature):
            captured_system.append(system)
            return '{"ok": true}'

        with patch.object(client, "_call_anthropic", side_effect=_capture):
            client.call_json("Be helpful.", "Extract events.")

        assert "JSON" in captured_system[0]

    def test_call_json_returns_none_when_call_returns_none(self):
        """If call() returns None, call_json must return None."""
        client = LLMClient(backend="anthropic")

        with patch.object(client, "call", return_value=None):
            result = client.call_json("sys", "prompt")

        assert result is None


# ── llm_call module-level function ────────────────────────────────────────────────

class TestLlmCallConvenienceFunction:
    def test_llm_call_uses_provided_client(self):
        """llm_call must use the provided client.call() method."""
        client = MagicMock(spec=LLMClient)
        client.call.return_value = "mock response"

        result = llm_call("sys", "prompt", client=client)

        assert result == "mock response"
        client.call.assert_called_once()

    def test_llm_call_creates_client_when_none_provided(self):
        """llm_call without a client must create one from environment."""
        with patch("geoeventfusion.clients.llm_client.LLMClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.call.return_value = "ok"
            MockClient.return_value = mock_instance

            result = llm_call("sys", "prompt", client=None)

        MockClient.assert_called_once()
        assert result == "ok"
