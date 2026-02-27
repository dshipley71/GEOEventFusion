"""Dual-backend LLM client for GEOEventFusion.

Provides a backend-agnostic llm_call() interface that dispatches to either
the Anthropic API or Ollama depending on PipelineConfig.llm_backend.

All agent code must call llm_call() — never import anthropic or ollama directly.

Design rules from CLAUDE.md:
- Always request JSON-only output; apply defensive parsing (strip fences, find boundaries).
- Always retry once on empty LLM response before failing.
- Enforce MAX_CONFIDENCE cap after every LLM call that returns a confidence value.
- min_max_tokens: always set max_tokens >= 256 for structured extraction calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _safe_parse_llm_json(text: str) -> Optional[Any]:
    """Defensively parse LLM JSON output.

    Strips markdown code fences, then searches for the outermost JSON
    array or object boundaries and parses only that portion.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed Python object (dict or list), or None on failure.
    """
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try array first, then object
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = text.find(start_char)
        e = text.rfind(end_char)
        if s != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                pass

    return None


class LLMClient:
    """Backend-agnostic LLM client.

    Dispatches to Anthropic or Ollama based on the configured backend.
    Handles retry on empty response and enforces max_confidence capping.

    Args:
        backend: LLM backend name ("anthropic" or "ollama").
        anthropic_model: Anthropic model ID.
        ollama_model: Ollama model name.
        ollama_host: Ollama server URL.
        ollama_api_key: Ollama Cloud API key for Bearer token auth. Required when
            ollama_host points to Ollama Cloud (https://api.ollama.com). Leave
            empty for local Ollama instances that do not require authentication.
        anthropic_api_key: Anthropic API key (from environment).
        max_confidence: Hard cap for LLM confidence scores.
    """

    def __init__(
        self,
        backend: str = "ollama",
        anthropic_model: str = "claude-sonnet-4-6",
        ollama_model: str = "gemma3:27b",
        ollama_host: str = "http://localhost:11434",
        ollama_api_key: str = "",
        anthropic_api_key: Optional[str] = None,
        max_confidence: float = 0.82,
    ) -> None:
        self.backend = backend.lower()
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.ollama_api_key = ollama_api_key
        self.anthropic_api_key = anthropic_api_key
        self.max_confidence = max_confidence
        self._anthropic_client: Optional[Any] = None
        self._ollama_client: Optional[Any] = None

    def _get_anthropic_client(self) -> Any:
        """Lazily initialize and return the Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic  # type: ignore[import]

                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.anthropic_api_key
                )
            except ImportError:
                raise ImportError(
                    "anthropic package is required for the Anthropic backend. "
                    "Install with: pip install anthropic"
                )
        return self._anthropic_client

    def _get_ollama_client(self) -> Any:
        """Lazily initialize and return the Ollama client.

        When ollama_api_key is set, passes an Authorization: Bearer header
        for Ollama Cloud (https://api.ollama.com) authentication.
        """
        if self._ollama_client is None:
            try:
                import ollama  # type: ignore[import]

                kwargs: dict = {"host": self.ollama_host}
                if self.ollama_api_key:
                    kwargs["headers"] = {"Authorization": f"Bearer {self.ollama_api_key}"}
                self._ollama_client = ollama.Client(**kwargs)
            except ImportError:
                raise ImportError(
                    "ollama package is required for the Ollama backend. "
                    "Install with: pip install ollama"
                )
        return self._ollama_client

    def _call_anthropic(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """Execute a call against the Anthropic API.

        Args:
            system: System prompt string.
            prompt: User message/prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Response text string.
        """
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text or ""
        return ""

    def _call_ollama(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """Execute a call against the Ollama API.

        Args:
            system: System prompt string.
            prompt: User message/prompt string.
            max_tokens: Maximum tokens to generate (num_predict).
            temperature: Sampling temperature.

        Returns:
            Response text string.
        """
        client = self._get_ollama_client()
        response = client.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        if response and hasattr(response, "message") and response.message:
            return response.message.content or ""
        return ""

    def call(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """Execute an LLM call with retry on empty response.

        Retries once with doubled max_tokens if the first response is empty.

        Args:
            system: System prompt string.
            prompt: User message/prompt string.
            max_tokens: Maximum tokens to generate (minimum: LLM_MIN_MAX_TOKENS).
            temperature: Sampling temperature.

        Returns:
            Response text string, or None if both attempts return empty.
        """
        # Enforce minimum token budget for structured extraction
        from config.defaults import LLM_MIN_MAX_TOKENS

        max_tokens = max(max_tokens, LLM_MIN_MAX_TOKENS)

        for attempt in range(2):
            try:
                if self.backend == "anthropic":
                    result = self._call_anthropic(system, prompt, max_tokens, temperature)
                else:
                    result = self._call_ollama(system, prompt, max_tokens, temperature)

                if result and result.strip():
                    return result

                if attempt == 0:
                    logger.warning(
                        "LLM returned empty response on first attempt — retrying with "
                        "max_tokens=%d",
                        max_tokens * 2,
                    )
                    max_tokens = max_tokens * 2

            except Exception as exc:
                if attempt == 0:
                    logger.warning("LLM call failed (attempt 1): %s — retrying", exc)
                else:
                    logger.error("LLM call failed (attempt 2): %s", exc)
                    return None

        logger.error("LLM returned empty response after 2 attempts")
        return None

    def call_json(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Optional[Any]:
        """Execute an LLM call and defensively parse the JSON response.

        Automatically appends "Return only a JSON object or array." to the system prompt
        to enforce JSON-only output.

        Args:
            system: System prompt string.
            prompt: User message/prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Parsed Python object (dict or list), or None on parse failure.
        """
        json_system = system.rstrip() + "\n\nReturn only valid JSON. Do not include any explanation or markdown fences."
        raw = self.call(json_system, prompt, max_tokens, temperature)
        if raw is None:
            return None
        parsed = _safe_parse_llm_json(raw)
        if parsed is None:
            logger.warning("LLM JSON parse failed. Raw response (first 200 chars): %.200s", raw)
        return parsed

    def enforce_confidence_cap(self, data: Any) -> Any:
        """Recursively apply MAX_CONFIDENCE cap to any 'confidence' fields in a data structure.

        Args:
            data: Dict, list, or scalar value.

        Returns:
            Data with all confidence values capped at self.max_confidence.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "confidence" and isinstance(value, (int, float)):
                    data[key] = min(float(value), self.max_confidence)
                else:
                    data[key] = self.enforce_confidence_cap(value)
        elif isinstance(data, list):
            data = [self.enforce_confidence_cap(item) for item in data]
        return data


# ── Module-level convenience function ─────────────────────────────────────────

def llm_call(
    system: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    client: Optional[LLMClient] = None,
) -> Optional[str]:
    """Module-level LLM call function using a provided or default client.

    Agent code should use this function rather than instantiating LLMClient directly.
    Pass the client from PipelineConfig-derived initialization for proper backend routing.

    Args:
        system: System prompt string.
        prompt: User message/prompt string.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        client: Optional pre-initialized LLMClient instance.

    Returns:
        Response text string, or None on failure.
    """
    if client is None:
        import os

        backend = os.getenv("LLM_BACKEND", "ollama")
        client = LLMClient(backend=backend)
    return client.call(system, prompt, max_tokens, temperature)
