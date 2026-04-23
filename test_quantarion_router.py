"""
Tests for AI Router — Multi-Provider LLM Routing Engine
=========================================================
All tests use mocks — no real API keys required.
Run: pytest test_quantarion_router.py -v
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from quantarion_router import (
    AIRouter,
    CircuitBreakerConfig,
    CircuitOpenError,
    ConfigurationError,
    Provider,
    ProviderConfig,
    ProviderNotFoundError,
    RouteResult,
    RouterError,
    RoutingStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    provider: Provider = Provider.ANTHROPIC,
    api_key: str = "test-key",
    model: str = "claude-3-haiku-20240307",
    max_tokens: int = 256,
    timeout_seconds: float = 5.0,
    enabled: bool = True,
    name: str = "",
    base_url: str | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        enabled=enabled,
        name=name,
        base_url=base_url,
    )


def _anthropic_response(
    text: str, input_tokens: int = 10, output_tokens: int = 20,
) -> MagicMock:
    """Build a mock urllib response for Anthropic."""
    body = json.dumps({
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _openai_response(
    text: str, prompt_tokens: int = 8, completion_tokens: int = 15,
) -> MagicMock:
    """Build a mock urllib response for OpenAI."""
    body = json.dumps({
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _noop_sleep(seconds: float) -> None:
    """No-op sleep for tests — skip real delays."""
    pass


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestInitialization(unittest.TestCase):

    def test_default_strategy_is_fallback(self) -> None:
        router = AIRouter()
        self.assertEqual(router.strategy, RoutingStrategy.FALLBACK)

    def test_custom_strategy(self) -> None:
        router = AIRouter(strategy=RoutingStrategy.ROUND_ROBIN)
        self.assertEqual(router.strategy, RoutingStrategy.ROUND_ROBIN)

    def test_invalid_strategy_type(self) -> None:
        with self.assertRaises(TypeError):
            AIRouter(strategy="fallback")  # type: ignore

    def test_invalid_max_retries_negative(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(max_retries=-1)

    def test_invalid_max_retries_type(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(max_retries=1.5)  # type: ignore

    def test_bool_max_retries_raises(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(max_retries=True)  # type: ignore

    def test_zero_max_retries_allowed(self) -> None:
        router = AIRouter(max_retries=0)
        self.assertEqual(router.provider_count, 0)

    def test_provider_count_starts_at_zero(self) -> None:
        router = AIRouter()
        self.assertEqual(router.provider_count, 0)

    def test_negative_backoff_base_raises(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(backoff_base=-0.1)

    def test_negative_backoff_max_raises(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(backoff_max=-1.0)

    def test_zero_backoff_allowed(self) -> None:
        router = AIRouter(backoff_base=0.0, backoff_max=0.0)
        self.assertEqual(router.provider_count, 0)


# ---------------------------------------------------------------------------
# 2. add_provider
# ---------------------------------------------------------------------------

class TestAddProvider(unittest.TestCase):

    def setUp(self) -> None:
        self.router = AIRouter()

    def test_add_single_provider(self) -> None:
        self.router.add_provider(_make_config())
        self.assertEqual(self.router.provider_count, 1)

    def test_add_two_providers(self) -> None:
        self.router.add_provider(_make_config(Provider.ANTHROPIC))
        self.router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))
        self.assertEqual(self.router.provider_count, 2)

    def test_duplicate_provider_raises(self) -> None:
        self.router.add_provider(_make_config())
        with self.assertRaises(ConfigurationError):
            self.router.add_provider(_make_config())

    def test_same_provider_enum_different_names(self) -> None:
        """Two configs with same Provider enum but different names should work."""
        self.router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o", name="openai-primary"))
        self.router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini", name="openai-fallback"))
        self.assertEqual(self.router.provider_count, 2)

    def test_invalid_config_type(self) -> None:
        with self.assertRaises(TypeError):
            self.router.add_provider("not a config")  # type: ignore

    def test_empty_api_key_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.router.add_provider(_make_config(api_key=""))

    def test_empty_model_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.router.add_provider(_make_config(model=""))

    def test_zero_timeout_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.router.add_provider(_make_config(timeout_seconds=0.0))

    def test_negative_timeout_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.router.add_provider(_make_config(timeout_seconds=-1.0))

    def test_zero_max_tokens_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.router.add_provider(_make_config(max_tokens=0))

    def test_chaining_returns_self(self) -> None:
        result = self.router.add_provider(_make_config())
        self.assertIs(result, self.router)

    def test_provider_config_default_name(self) -> None:
        config = _make_config(Provider.ANTHROPIC)
        self.assertEqual(config.name, "anthropic")

    def test_provider_config_custom_name(self) -> None:
        config = _make_config(Provider.CUSTOM, name="groq-llama")
        self.assertEqual(config.name, "groq-llama")


# ---------------------------------------------------------------------------
# 3. complete — Anthropic success
# ---------------------------------------------------------------------------

class TestCompleteAnthropic(unittest.TestCase):

    def _make_router(self) -> AIRouter:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        return router

    @patch("urllib.request.urlopen")
    def test_successful_response(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("Hello from Claude")
        router = self._make_router()
        result = router.complete("Say hello")
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Hello from Claude")
        self.assertEqual(result.provider, Provider.ANTHROPIC)

    @patch("urllib.request.urlopen")
    def test_token_counts_captured(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("Hi", input_tokens=5, output_tokens=3)
        router = self._make_router()
        result = router.complete("Hi")
        self.assertEqual(result.input_tokens, 5)
        self.assertEqual(result.output_tokens, 3)

    @patch("urllib.request.urlopen")
    def test_duration_is_positive(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok")
        router = self._make_router()
        result = router.complete("test")
        self.assertGreaterEqual(result.duration_ms, 0.0)

    @patch("urllib.request.urlopen")
    def test_attempts_is_one_on_first_try(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok")
        router = self._make_router()
        result = router.complete("test")
        self.assertEqual(result.attempts, 1)

    def test_empty_prompt_raises(self) -> None:
        router = self._make_router()
        with self.assertRaises(ValueError):
            router.complete("")

    def test_non_string_prompt_raises(self) -> None:
        router = self._make_router()
        with self.assertRaises(ValueError):
            router.complete(None)  # type: ignore

    @patch("urllib.request.urlopen")
    def test_provider_name_in_result(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok")
        router = self._make_router()
        result = router.complete("test")
        self.assertEqual(result.provider_name, "anthropic")


# ---------------------------------------------------------------------------
# 4. complete — OpenAI success
# ---------------------------------------------------------------------------

class TestCompleteOpenAI(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_openai_success(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _openai_response("Hello from GPT")
        router = AIRouter()
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))
        result = router.complete("Say hello")
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Hello from GPT")
        self.assertEqual(result.provider, Provider.OPENAI)

    @patch("urllib.request.urlopen")
    def test_openai_token_counts(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _openai_response("ok", prompt_tokens=7, completion_tokens=4)
        router = AIRouter()
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))
        result = router.complete("test")
        self.assertEqual(result.input_tokens, 7)
        self.assertEqual(result.output_tokens, 4)


# ---------------------------------------------------------------------------
# 5. Messages API (multi-turn)
# ---------------------------------------------------------------------------

class TestMessagesAPI(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_messages_list_input(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("Multi-turn response")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = router.complete(messages)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Multi-turn response")

    @patch("urllib.request.urlopen")
    def test_system_prompt_anthropic(self, mock_urlopen: MagicMock) -> None:
        """Verify system prompt is passed in Anthropic payload."""
        mock_urlopen.return_value = _anthropic_response("ok")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        router.complete("test", system="You are a pirate.")

        # Inspect the payload sent to urlopen
        req_obj = mock_urlopen.call_args[0][0]
        payload = json.loads(req_obj.data.decode("utf-8"))
        self.assertEqual(payload["system"], "You are a pirate.")

    @patch("urllib.request.urlopen")
    def test_system_prompt_openai(self, mock_urlopen: MagicMock) -> None:
        """Verify system prompt becomes first message for OpenAI."""
        mock_urlopen.return_value = _openai_response("ok")
        router = AIRouter()
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        router.complete("test", system="You are a pirate.")

        req_obj = mock_urlopen.call_args[0][0]
        payload = json.loads(req_obj.data.decode("utf-8"))
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "You are a pirate.")
        self.assertEqual(payload["messages"][1]["role"], "user")

    def test_empty_messages_list_raises(self) -> None:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        with self.assertRaises(ValueError):
            router.complete([])


# ---------------------------------------------------------------------------
# 6. Fallback strategy
# ---------------------------------------------------------------------------

class TestFallbackStrategy(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_falls_back_to_second_provider(self, mock_urlopen: MagicMock) -> None:
        openai_resp = _openai_response("Fallback response")
        mock_urlopen.side_effect = [
            urllib.error.URLError("timeout"),
            openai_resp,
        ]
        router = AIRouter(strategy=RoutingStrategy.FALLBACK, max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        result = router.complete("test")
        self.assertTrue(result.success)
        self.assertEqual(result.provider, Provider.OPENAI)
        self.assertEqual(result.content, "Fallback response")

    @patch("urllib.request.urlopen")
    def test_all_fail_raises_router_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        router = AIRouter(strategy=RoutingStrategy.FALLBACK, max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        with self.assertRaises(RouterError):
            router.complete("test")

    @patch("urllib.request.urlopen")
    def test_retry_before_fallback(self, mock_urlopen: MagicMock) -> None:
        openai_resp = _openai_response("ok")
        mock_urlopen.side_effect = [
            urllib.error.URLError("fail 1"),
            urllib.error.URLError("fail 2"),
            openai_resp,
        ]
        router = AIRouter(strategy=RoutingStrategy.FALLBACK, max_retries=1)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        result = router.complete("test")
        self.assertTrue(result.success)
        self.assertEqual(result.provider, Provider.OPENAI)
        self.assertEqual(mock_urlopen.call_count, 3)


# ---------------------------------------------------------------------------
# 7. Primary strategy
# ---------------------------------------------------------------------------

class TestPrimaryStrategy(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_primary_does_not_fallback(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        router = AIRouter(strategy=RoutingStrategy.PRIMARY, max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        with self.assertRaises(RouterError):
            router.complete("test")

        self.assertEqual(mock_urlopen.call_count, 1)


# ---------------------------------------------------------------------------
# 8. Round-robin strategy
# ---------------------------------------------------------------------------

class TestRoundRobinStrategy(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_round_robin_alternates(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            _anthropic_response("from anthropic"),
            _openai_response("from openai"),
            _anthropic_response("from anthropic again"),
        ]
        router = AIRouter(strategy=RoutingStrategy.ROUND_ROBIN, max_retries=0)
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        r1 = router.complete("test 1")
        r2 = router.complete("test 2")
        r3 = router.complete("test 3")

        self.assertEqual(r1.provider, Provider.ANTHROPIC)
        self.assertEqual(r2.provider, Provider.OPENAI)
        self.assertEqual(r3.provider, Provider.ANTHROPIC)


# ---------------------------------------------------------------------------
# 9. disable / enable provider
# ---------------------------------------------------------------------------

class TestEnableDisable(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_disabled_provider_is_skipped(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _openai_response("from openai")
        router = AIRouter(strategy=RoutingStrategy.FALLBACK, max_retries=0)
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))
        router.disable_provider(Provider.ANTHROPIC)

        result = router.complete("test")
        self.assertEqual(result.provider, Provider.OPENAI)
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("urllib.request.urlopen")
    def test_re_enabled_provider_is_used(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("from anthropic")
        router = AIRouter(max_retries=0)
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.disable_provider(Provider.ANTHROPIC)
        router.enable_provider(Provider.ANTHROPIC)

        result = router.complete("test")
        self.assertEqual(result.provider, Provider.ANTHROPIC)

    def test_disable_unknown_provider_raises(self) -> None:
        router = AIRouter()
        with self.assertRaises(ProviderNotFoundError):
            router.disable_provider(Provider.OPENAI)

    def test_enable_unknown_provider_raises(self) -> None:
        router = AIRouter()
        with self.assertRaises(ProviderNotFoundError):
            router.enable_provider(Provider.ANTHROPIC)

    def test_all_disabled_raises_configuration_error(self) -> None:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.disable_provider(Provider.ANTHROPIC)
        with self.assertRaises(ConfigurationError):
            router.complete("test")

    def test_disable_by_name_string(self) -> None:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.disable_provider("anthropic")
        with self.assertRaises(ConfigurationError):
            router.complete("test")

    def test_enable_by_name_string(self) -> None:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.disable_provider("anthropic")
        router.enable_provider("anthropic")
        # Should not raise — provider is enabled again
        # (will fail on actual API call, but won't raise ConfigurationError)

    def test_disable_by_enum_affects_all_same_vendor(self) -> None:
        """
        Regression: disable_provider(Provider.ANTHROPIC) must disable EVERY
        registered config of that vendor, not just the first match.
        Previously _find_provider returned only the first, leaving other
        Anthropic configs silently still active.
        """
        router = AIRouter()
        router.add_provider(_make_config(
            Provider.ANTHROPIC, model="haiku", name="cheap-claude",
        ))
        router.add_provider(_make_config(
            Provider.ANTHROPIC, model="opus", name="smart-claude",
        ))
        router.add_provider(_make_config(
            Provider.OPENAI, model="gpt-4o", name="openai-main",
        ))

        router.disable_provider(Provider.ANTHROPIC)

        # Both anthropic configs must be disabled, openai must stay enabled.
        names_to_state = {p.name: p.enabled for p in router._providers}
        self.assertFalse(names_to_state["cheap-claude"])
        self.assertFalse(names_to_state["smart-claude"])
        self.assertTrue(names_to_state["openai-main"])

    def test_disable_by_name_string_affects_only_exact_match(self) -> None:
        """
        disable_provider(name_string) should only target the provider with
        that exact name, even if multiple configs share the same Provider enum.
        """
        router = AIRouter()
        router.add_provider(_make_config(
            Provider.ANTHROPIC, model="haiku", name="cheap-claude",
        ))
        router.add_provider(_make_config(
            Provider.ANTHROPIC, model="opus", name="smart-claude",
        ))

        router.disable_provider("cheap-claude")

        names_to_state = {p.name: p.enabled for p in router._providers}
        self.assertFalse(names_to_state["cheap-claude"])
        self.assertTrue(names_to_state["smart-claude"])

    def test_enable_by_enum_affects_all_same_vendor(self) -> None:
        """Symmetric case: enable_provider(enum) re-enables all matches."""
        router = AIRouter()
        router.add_provider(_make_config(
            Provider.ANTHROPIC, model="haiku", name="a1",
        ))
        router.add_provider(_make_config(
            Provider.ANTHROPIC, model="opus", name="a2",
        ))
        router.disable_provider(Provider.ANTHROPIC)

        router.enable_provider(Provider.ANTHROPIC)

        self.assertTrue(all(p.enabled for p in router._providers))


# ---------------------------------------------------------------------------
# 10. Metrics
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_success_increments_metrics(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok", input_tokens=5, output_tokens=3)
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.complete("test")

        m = router.get_metrics()["anthropic"]
        self.assertEqual(m["calls"], 1)
        self.assertEqual(m["successes"], 1)
        self.assertEqual(m["failures"], 0)
        self.assertEqual(m["total_input_tokens"], 5)
        self.assertEqual(m["total_output_tokens"], 3)

    @patch("urllib.request.urlopen")
    def test_failure_increments_failures(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))

        with self.assertRaises(RouterError):
            router.complete("test")

        m = router.get_metrics()["anthropic"]
        self.assertEqual(m["failures"], 1)
        self.assertEqual(m["successes"], 0)

    @patch("urllib.request.urlopen")
    def test_success_rate_calculated(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            _anthropic_response("ok"),
            urllib.error.URLError("fail"),
        ]
        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))

        router.complete("test 1")
        with self.assertRaises(RouterError):
            router.complete("test 2")

        m = router.get_metrics()["anthropic"]
        self.assertAlmostEqual(m["success_rate"], 0.5)

    @patch("urllib.request.urlopen")
    def test_reset_metrics(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.complete("test")
        router.reset_metrics()

        m = router.get_metrics()["anthropic"]
        self.assertEqual(m["calls"], 0)
        self.assertEqual(m["successes"], 0)
        self.assertEqual(m["total_input_tokens"], 0)

    @patch("urllib.request.urlopen")
    def test_avg_ms_positive_after_call(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.complete("test")

        m = router.get_metrics()["anthropic"]
        self.assertGreaterEqual(m["avg_ms"], 0.0)


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):
    """Tests for HTTP error body extraction (bug #5)."""

    def _make_http_error(self, status: int, body: bytes, reason: str = "Error") -> urllib.error.HTTPError:
        """Build a urllib HTTPError with a readable body."""
        import io
        return urllib.error.HTTPError(
            url="https://api.example.com",
            code=status,
            msg=reason,
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(body),
        )

    def test_format_api_error_parses_openai_json(self) -> None:
        """OpenAI-style {"error": {"message": ...}} is parsed."""
        from quantarion_router import _format_api_error
        body = json.dumps({
            "error": {"message": "Invalid API key", "type": "invalid_request_error"}
        })
        msg = _format_api_error(401, "Unauthorized", body)
        self.assertIn("401", msg)
        self.assertIn("Invalid API key", msg)

    def test_format_api_error_parses_anthropic_json(self) -> None:
        """Anthropic uses the same shape — should parse identically."""
        from quantarion_router import _format_api_error
        body = json.dumps({
            "error": {"type": "authentication_error", "message": "x-api-key header is required"}
        })
        msg = _format_api_error(401, "Unauthorized", body)
        self.assertIn("401", msg)
        self.assertIn("x-api-key header is required", msg)

    def test_format_api_error_handles_plain_text(self) -> None:
        """Non-JSON bodies should still surface as a snippet."""
        from quantarion_router import _format_api_error
        msg = _format_api_error(502, "Bad Gateway", "upstream timeout")
        self.assertIn("502", msg)
        self.assertIn("upstream timeout", msg)

    def test_format_api_error_handles_empty_body(self) -> None:
        """Empty body should not crash, just return the status line."""
        from quantarion_router import _format_api_error
        msg = _format_api_error(500, "Internal Server Error", "")
        self.assertIn("500", msg)
        # No "None" or trailing ": " noise.
        self.assertFalse(msg.endswith(": "))

    def test_format_api_error_handles_none_body(self) -> None:
        from quantarion_router import _format_api_error
        msg = _format_api_error(503, "Service Unavailable", None)
        self.assertIn("503", msg)

    def test_format_api_error_truncates_huge_body(self) -> None:
        """Bodies > 500 chars get truncated with ellipsis."""
        from quantarion_router import _format_api_error
        huge = "x" * 2000
        msg = _format_api_error(500, "Error", huge)
        self.assertLess(len(msg), 700)  # headers + truncated body
        self.assertIn("...", msg)

    @patch("urllib.request.urlopen")
    def test_anthropic_401_surfaces_api_message(self, mock_urlopen: MagicMock) -> None:
        """
        Regression: when Anthropic returns 401 with a JSON error body, the
        RouterError message must include the provider's message, not just
        the raw 'HTTP 401'.
        """
        body = json.dumps({
            "error": {"type": "authentication_error", "message": "invalid x-api-key"}
        }).encode("utf-8")
        mock_urlopen.side_effect = self._make_http_error(401, body, "Unauthorized")

        router = AIRouter(max_retries=0)
        router.add_provider(_make_config(Provider.ANTHROPIC))

        with self.assertRaises(RouterError) as ctx:
            router.complete("test")
        err_msg = str(ctx.exception)
        self.assertIn("401", err_msg)
        self.assertIn("invalid x-api-key", err_msg)

    @patch("urllib.request.urlopen")
    def test_openai_429_surfaces_rate_limit_message(self, mock_urlopen: MagicMock) -> None:
        body = json.dumps({
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        }).encode("utf-8")
        mock_urlopen.side_effect = self._make_http_error(429, body, "Too Many Requests")

        router = AIRouter(max_retries=0)
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        with self.assertRaises(RouterError) as ctx:
            router.complete("test")
        err_msg = str(ctx.exception)
        self.assertIn("429", err_msg)
        self.assertIn("Rate limit exceeded", err_msg)


class TestEdgeCases(unittest.TestCase):

    def test_no_providers_raises_configuration_error(self) -> None:
        router = AIRouter()
        with self.assertRaises(ConfigurationError):
            router.complete("test")

    @patch("urllib.request.urlopen")
    def test_empty_anthropic_content_raises(self, mock_urlopen: MagicMock) -> None:
        body = json.dumps({"content": [], "usage": {}}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        with self.assertRaises(RouterError):
            router.complete("test")

    @patch("urllib.request.urlopen")
    def test_empty_openai_choices_raises(self, mock_urlopen: MagicMock) -> None:
        body = json.dumps({"choices": [], "usage": {}}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))
        with self.assertRaises(RouterError):
            router.complete("test")

    def test_repr_contains_strategy_and_providers(self) -> None:
        router = AIRouter(strategy=RoutingStrategy.FALLBACK)
        router.add_provider(_make_config(Provider.ANTHROPIC))
        r = repr(router)
        self.assertIn("fallback", r)
        self.assertIn("anthropic", r)

    @patch("urllib.request.urlopen")
    def test_multiple_retries_counted_in_attempts(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            urllib.error.URLError("fail 1"),
            urllib.error.URLError("fail 2"),
            _anthropic_response("ok on 3rd"),
        ]
        router = AIRouter(max_retries=2)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        result = router.complete("test")
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 3)


# ---------------------------------------------------------------------------
# 12. ProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig(unittest.TestCase):

    def test_default_values(self) -> None:
        config = ProviderConfig(
            provider=Provider.ANTHROPIC,
            api_key="key",
            model="claude-3-haiku-20240307",
        )
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertTrue(config.enabled)
        self.assertEqual(config.name, "anthropic")

    def test_custom_values(self) -> None:
        config = ProviderConfig(
            provider=Provider.OPENAI,
            api_key="sk-test",
            model="gpt-4o",
            max_tokens=512,
            timeout_seconds=10.0,
            enabled=False,
        )
        self.assertEqual(config.max_tokens, 512)
        self.assertFalse(config.enabled)

    def test_base_url_and_headers(self) -> None:
        config = ProviderConfig(
            provider=Provider.CUSTOM,
            api_key="key",
            model="llama-3",
            base_url="https://my-endpoint.com/v1/chat",
            headers={"X-Custom": "value"},
            name="my-llama",
        )
        self.assertEqual(config.base_url, "https://my-endpoint.com/v1/chat")
        self.assertEqual(config.headers, {"X-Custom": "value"})
        self.assertEqual(config.name, "my-llama")


# ---------------------------------------------------------------------------
# 13. Exponential backoff
# ---------------------------------------------------------------------------

class TestBackoff(unittest.TestCase):

    def test_backoff_delay_formula(self) -> None:
        router = AIRouter(backoff_base=1.0, backoff_max=10.0, jitter=0.0)
        self.assertAlmostEqual(router._backoff_delay(0), 1.0)
        self.assertAlmostEqual(router._backoff_delay(1), 2.0)
        self.assertAlmostEqual(router._backoff_delay(2), 4.0)
        self.assertAlmostEqual(router._backoff_delay(3), 8.0)
        self.assertAlmostEqual(router._backoff_delay(4), 10.0)  # capped

    def test_backoff_zero_base(self) -> None:
        router = AIRouter(backoff_base=0.0)
        self.assertAlmostEqual(router._backoff_delay(0), 0.0)
        self.assertAlmostEqual(router._backoff_delay(5), 0.0)

    @patch("urllib.request.urlopen")
    def test_backoff_is_called_between_retries(self, mock_urlopen: MagicMock) -> None:
        """Verify sleep is called between retries."""
        mock_urlopen.side_effect = [
            urllib.error.URLError("fail 1"),
            urllib.error.URLError("fail 2"),
            _anthropic_response("ok"),
        ]
        sleep_calls: list[float] = []

        def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        router = AIRouter(max_retries=2, backoff_base=0.5, backoff_max=10.0, jitter=0.0)
        router._sleep_fn = mock_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        result = router.complete("test")

        self.assertTrue(result.success)
        # 2 retries = 2 backoff sleeps (before retry 1 and retry 2)
        self.assertEqual(len(sleep_calls), 2)
        self.assertAlmostEqual(sleep_calls[0], 0.5)   # 0.5 * 2^0
        self.assertAlmostEqual(sleep_calls[1], 1.0)   # 0.5 * 2^1

    def test_async_backoff_uses_injectable_sleep(self) -> None:
        """
        Regression: async route loop must use the injectable _async_sleep_fn,
        not a hardcoded asyncio.sleep. Previously tests on async backoff
        would wait real seconds because asyncio.sleep was called directly,
        making CI slow and breaking symmetry with the sync sleep_fn.
        """
        import quantarion_router as ar

        sleep_calls: list[float] = []

        async def mock_async_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        # Force the native aiohttp path so _async_route_loop is exercised.
        original_has = ar._HAS_AIOHTTP
        ar._HAS_AIOHTTP = True
        try:
            router = AIRouter(
                max_retries=2, backoff_base=0.5, backoff_max=10.0, jitter=0.0,
            )
            router._async_sleep_fn = mock_async_sleep
            router.add_provider(_make_config(Provider.ANTHROPIC))

            # Stub the actual provider call so we don't hit the network.
            call_count = [0]

            async def mock_acall_provider(config, messages, system):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise RuntimeError(f"simulated failure {call_count[0]}")
                return ("ok", 1, 1)

            router._acall_provider = mock_acall_provider  # type: ignore[assignment]

            result = asyncio.run(router.acomplete("test"))
            self.assertTrue(result.success)
            # 2 retries = 2 sleeps, and the mock captured them (not real wait).
            self.assertEqual(len(sleep_calls), 2)
            self.assertAlmostEqual(sleep_calls[0], 0.5)
            self.assertAlmostEqual(sleep_calls[1], 1.0)
        finally:
            ar._HAS_AIOHTTP = original_has


# ---------------------------------------------------------------------------
# 14. Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker(unittest.TestCase):

    def _make_cb_router(self, threshold: int = 3, recovery: float = 1.0) -> AIRouter:
        router = AIRouter(
            max_retries=0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=threshold,
                recovery_timeout=recovery,
                half_open_max=1,
            ),
        )
        router._sleep_fn = _noop_sleep
        return router

    @patch("urllib.request.urlopen")
    def test_circuit_opens_after_threshold(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        router = self._make_cb_router(threshold=2)
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        # Two failures should open the circuit for anthropic
        with self.assertRaises(RouterError):
            router.complete("test 1")
        with self.assertRaises(RouterError):
            router.complete("test 2")

        states = router.get_circuit_states()
        self.assertEqual(states["anthropic"], "open")

    @patch("urllib.request.urlopen")
    def test_circuit_success_resets_failures(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            urllib.error.URLError("fail"),
            _anthropic_response("ok"),
        ]
        router = self._make_cb_router(threshold=3)
        router.add_provider(_make_config(Provider.ANTHROPIC))

        with self.assertRaises(RouterError):
            router.complete("test 1")

        router.complete("test 2")  # success resets

        states = router.get_circuit_states()
        self.assertEqual(states["anthropic"], "closed")

    def test_circuit_states_empty_without_config(self) -> None:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        self.assertEqual(router.get_circuit_states(), {})

    @patch("urllib.request.urlopen")
    def test_circuit_skips_open_provider(self, mock_urlopen: MagicMock) -> None:
        """When circuit is open for primary, fallback should be used."""
        # Only anthropic is registered initially — open its circuit
        router = self._make_cb_router(threshold=2)
        router.add_provider(_make_config(Provider.ANTHROPIC))

        mock_urlopen.side_effect = urllib.error.URLError("fail")
        with self.assertRaises(RouterError):
            router.complete("test 1")
        with self.assertRaises(RouterError):
            router.complete("test 2")

        states = router.get_circuit_states()
        self.assertEqual(states["anthropic"], "open")

        # Now add OpenAI — its circuit is clean
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))
        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _openai_response("fallback ok")

        result = router.complete("test 3")
        self.assertTrue(result.success)
        self.assertEqual(result.provider, Provider.OPENAI)

    @patch("urllib.request.urlopen")
    def test_all_providers_circuit_open_raises_circuit_open_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """
        Regression (bug #6): when every provider is skipped because its
        circuit is open, the caller should get a CircuitOpenError so they
        can distinguish "temporarily protected, retry later" from
        "permanent failure". Previously only RouterError was raised.
        """
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        router = self._make_cb_router(threshold=1)  # trip immediately
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        # Trip both circuits.
        with self.assertRaises(RouterError):
            router.complete("prime")
        # At least one provider should now be circuit-open.
        states = router.get_circuit_states()
        self.assertIn("open", states.values())

        # Trip the remaining one too if still closed.
        # With threshold=1 and two providers, the first call opens anthropic,
        # falls back to openai which also fails -> openai opens too.
        # So next call should hit fully circuit-open state.
        with self.assertRaises(CircuitOpenError) as ctx:
            router.complete("second call")

        # CircuitOpenError is also a RouterError (subclass) — ensure that
        # backwards-compatible catch still works.
        self.assertIsInstance(ctx.exception, RouterError)
        self.assertIn("circuit-open", str(ctx.exception).lower())

    @patch("urllib.request.urlopen")
    def test_partial_circuit_open_still_raises_router_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """
        If ONE provider is circuit-open and ANOTHER actually attempts and
        fails, the exception should be plain RouterError — not CircuitOpenError
        — because not every provider was purely circuit-protected.
        """
        router = self._make_cb_router(threshold=1)
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.add_provider(_make_config(Provider.OPENAI, model="gpt-4o-mini"))

        # First call: fail both, anthropic circuit opens, openai circuit opens.
        mock_urlopen.side_effect = urllib.error.URLError("boom")
        with self.assertRaises(RouterError):
            router.complete("prime")

        # Manually reset only openai's circuit to closed, keeping anthropic open.
        router._circuits["openai"].state = "closed"
        router._circuits["openai"].failures = 0

        # Now: anthropic skipped (cb open), openai tries and fails.
        # Since openai actually attempted -> plain RouterError, not CircuitOpenError.
        with self.assertRaises(RouterError) as ctx:
            router.complete("second")
        # Must be RouterError but NOT CircuitOpenError.
        self.assertNotIsInstance(ctx.exception, CircuitOpenError)

    def test_all_providers_circuit_open_raises_circuit_open_error_async(self) -> None:
        """Async symmetry for the all-circuits-open sentinel."""
        import quantarion_router as ar
        original = ar._HAS_AIOHTTP
        ar._HAS_AIOHTTP = False  # executor fallback exercises _run_route_loop
        try:
            router = AIRouter(
                max_retries=0,
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=1, recovery_timeout=60.0, half_open_max=1,
                ),
            )
            router._sleep_fn = _noop_sleep
            router.add_provider(_make_config(Provider.ANTHROPIC))

            # Manually put circuit into open state.
            router._circuits["anthropic"].state = "open"
            router._circuits["anthropic"].last_failure_time = time.monotonic()

            with self.assertRaises(CircuitOpenError):
                asyncio.run(router.acomplete("test"))
        finally:
            ar._HAS_AIOHTTP = original

    def test_failures_reset_on_transition_to_half_open(self) -> None:
        """
        Regression (bug #9): when the circuit transitions from 'open' to
        'half-open' after recovery_timeout, the failure counter must be
        reset to 0. Otherwise the state remains 'dirty' and a subsequent
        failure would log at a threshold that already seemed exceeded.
        """
        router = AIRouter(
            max_retries=0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=0.01,  # tiny so we can transition quickly
                half_open_max=1,
            ),
        )
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))

        # Manually drive the circuit to "open" with failures=3.
        cb = router._circuits["anthropic"]
        cb.state = "open"
        cb.failures = 3
        cb.last_failure_time = time.monotonic() - 1.0  # already past recovery

        # Trigger the transition by calling _check_circuit.
        router._check_circuit("anthropic")

        # After transition: state=half-open, failures reset to 0.
        self.assertEqual(cb.state, "half-open")
        self.assertEqual(cb.failures, 0)


# ---------------------------------------------------------------------------
# 15. Custom provider handlers
# ---------------------------------------------------------------------------

class TestCustomHandlers(unittest.TestCase):

    def test_custom_handler_called(self) -> None:
        def my_handler(config, messages, system):
            return "custom response", 5, 10

        router = AIRouter()
        config = _make_config(Provider.CUSTOM, name="my-llm", model="custom-v1")
        router.add_provider(config)
        router.register_handler("my-llm", my_handler)

        result = router.complete("test")
        self.assertTrue(result.success)
        self.assertEqual(result.content, "custom response")
        self.assertEqual(result.input_tokens, 5)
        self.assertEqual(result.output_tokens, 10)
        self.assertEqual(result.provider_name, "my-llm")

    def test_custom_handler_receives_system(self) -> None:
        received: dict = {}

        def my_handler(config, messages, system):
            received["system"] = system
            received["messages"] = list(messages)
            return "ok", 0, 0

        router = AIRouter()
        router.add_provider(_make_config(Provider.CUSTOM, name="test-llm", model="v1"))
        router.register_handler("test-llm", my_handler)

        router.complete("hello", system="Be helpful.")
        self.assertEqual(received["system"], "Be helpful.")
        self.assertEqual(received["messages"], [{"role": "user", "content": "hello"}])

    def test_register_non_callable_raises(self) -> None:
        router = AIRouter()
        with self.assertRaises(TypeError):
            router.register_handler("test", "not a function")  # type: ignore

    def test_custom_without_handler_raises(self) -> None:
        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.CUSTOM, name="no-handler", model="v1"))

        with self.assertRaises(RouterError):
            router.complete("test")

    def test_register_handler_chaining(self) -> None:
        router = AIRouter()
        result = router.register_handler("test", lambda c, m, s: ("", 0, 0))
        self.assertIs(result, router)


# ---------------------------------------------------------------------------
# 16. Async support
# ---------------------------------------------------------------------------

class TestAsync(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_acomplete_basic(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("Async hello")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        result = asyncio.run(router.acomplete("test"))
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Async hello")

    @patch("urllib.request.urlopen")
    def test_acomplete_with_system(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("Pirate hello")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        result = asyncio.run(router.acomplete("test", system="Be a pirate"))
        self.assertTrue(result.success)

    @patch("urllib.request.urlopen")
    def test_acomplete_with_messages(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("Multi async")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        messages = [{"role": "user", "content": "Hi"}]
        result = asyncio.run(router.acomplete(messages))
        self.assertTrue(result.success)

    def test_acomplete_empty_prompt_raises(self) -> None:
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        with self.assertRaises(ValueError):
            asyncio.run(router.acomplete(""))


# ---------------------------------------------------------------------------
# 17. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_concurrent_completes(self, mock_urlopen: MagicMock) -> None:
        """Run multiple complete() calls concurrently, verify no crashes."""
        mock_urlopen.return_value = _anthropic_response("ok")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        errors: list[Exception] = []
        results: list[RouteResult] = []

        def worker() -> None:
            try:
                r = router.complete("test")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 10)

        m = router.get_metrics()["anthropic"]
        self.assertEqual(m["calls"], 10)
        self.assertEqual(m["successes"], 10)

    @patch("urllib.request.urlopen")
    def test_metrics_invariant_under_concurrent_load(
        self, mock_urlopen: MagicMock
    ) -> None:
        """
        Bug #8 (documented, not a bug): get_metrics() snapshots taken while
        requests are in flight must always satisfy ``calls >= successes + failures``
        because ``calls`` is bumped at attempt start and success/failure only
        at completion. This test hammers the router while polling metrics and
        verifies the invariant never breaks.
        """
        mock_urlopen.return_value = _anthropic_response("ok")
        router = AIRouter(max_retries=0)
        router.add_provider(_make_config(Provider.ANTHROPIC))

        stop = threading.Event()
        violations: list[tuple[int, int, int]] = []

        def hammer() -> None:
            while not stop.is_set():
                try:
                    router.complete("test")
                except Exception:
                    pass

        def poll_metrics() -> None:
            for _ in range(500):
                m = router.get_metrics().get("anthropic", {})
                calls = m.get("calls", 0)
                succ = m.get("successes", 0)
                fail = m.get("failures", 0)
                if calls < succ + fail:
                    violations.append((calls, succ, fail))

        workers = [threading.Thread(target=hammer) for _ in range(5)]
        poller = threading.Thread(target=poll_metrics)
        for w in workers:
            w.start()
        poller.start()
        poller.join()
        stop.set()
        for w in workers:
            w.join()

        self.assertEqual(violations, [], f"Invariant broken: {violations[:5]}")

    def test_repr_safe_under_concurrent_add_provider(self) -> None:
        """
        Regression (bug #7): __repr__ iterated _providers without the lock,
        which could raise RuntimeError('list changed size during iteration')
        if add_provider() ran on another thread. The fix snapshots the list
        under self._lock.
        """
        router = AIRouter()
        stop = threading.Event()
        errors: list[Exception] = []

        def adder() -> None:
            i = 0
            while not stop.is_set():
                try:
                    # Unique name per add to avoid duplicate errors.
                    router.add_provider(_make_config(
                        Provider.ANTHROPIC, name=f"p{i}",
                    ))
                    i += 1
                except Exception as e:
                    errors.append(e)

        def repr_caller() -> None:
            for _ in range(500):
                try:
                    repr(router)
                except Exception as e:
                    errors.append(e)

        t_add = threading.Thread(target=adder)
        t_repr = threading.Thread(target=repr_caller)
        t_add.start()
        t_repr.start()
        t_repr.join()
        stop.set()
        t_add.join()

        # Filter out intentional ConfigurationError (if any unique-name
        # collision happened) — we only care that __repr__ never crashes.
        repr_errors = [e for e in errors if not isinstance(e, ConfigurationError)]
        self.assertEqual(repr_errors, [], f"repr crashed: {repr_errors}")


# ---------------------------------------------------------------------------
# 18. Logging
# ---------------------------------------------------------------------------

class TestLogging(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_success_logs_debug(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok")
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))

        with self.assertLogs("quantarion_router", level="DEBUG") as cm:
            router.complete("test")

        log_output = "\n".join(cm.output)
        self.assertIn("responded in", log_output)

    @patch("urllib.request.urlopen")
    def test_failure_logs_warning(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))

        with self.assertLogs("quantarion_router", level="WARNING") as cm:
            with self.assertRaises(RouterError):
                router.complete("test")

        log_output = "\n".join(cm.output)
        self.assertIn("failed", log_output)

    def test_add_provider_logs_info(self) -> None:
        router = AIRouter()
        with self.assertLogs("quantarion_router", level="INFO") as cm:
            router.add_provider(_make_config(Provider.ANTHROPIC))

        log_output = "\n".join(cm.output)
        self.assertIn("Registered provider", log_output)


# ---------------------------------------------------------------------------
# 19. Base URL / custom headers
# ---------------------------------------------------------------------------

class TestCustomEndpoints(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_anthropic_custom_base_url(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("proxy ok")
        router = AIRouter()
        router.add_provider(_make_config(
            Provider.ANTHROPIC,
            base_url="https://proxy.example.com/v1/messages",
        ))
        router.complete("test")

        req_obj = mock_urlopen.call_args[0][0]
        self.assertEqual(req_obj.full_url, "https://proxy.example.com/v1/messages")

    @patch("urllib.request.urlopen")
    def test_openai_custom_base_url(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _openai_response("proxy ok")
        router = AIRouter()
        router.add_provider(_make_config(
            Provider.OPENAI,
            model="gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1/chat/completions",
        ))
        router.complete("test")

        req_obj = mock_urlopen.call_args[0][0]
        self.assertEqual(req_obj.full_url, "https://openrouter.ai/api/v1/chat/completions")


# ---------------------------------------------------------------------------
# 20. Jitter
# ---------------------------------------------------------------------------

class TestJitter(unittest.TestCase):

    def test_jitter_validation_low(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(jitter=-0.1)

    def test_jitter_validation_high(self) -> None:
        with self.assertRaises(ValueError):
            AIRouter(jitter=1.5)

    def test_jitter_zero_allowed(self) -> None:
        router = AIRouter(jitter=0.0)
        # With jitter=0, delay should be deterministic
        self.assertAlmostEqual(router._backoff_delay(0), 0.5)  # base=0.5 default

    def test_jitter_one_allowed(self) -> None:
        router = AIRouter(jitter=1.0)
        self.assertIsInstance(router._backoff_delay(0), float)

    def test_jitter_applies_randomization(self) -> None:
        """With jitter=0.25, delay should vary within ±25%."""
        router = AIRouter(backoff_base=1.0, backoff_max=100.0, jitter=0.25)

        # Inject deterministic random to test boundaries
        # min boundary: delay - spread = delay * (1 - jitter)
        router._random_fn = lambda lo, hi: lo  # always returns lower bound
        delay_lo = router._backoff_delay(0)  # base=1.0, jitter=0.25 → 0.75
        self.assertAlmostEqual(delay_lo, 0.75)

        router._random_fn = lambda lo, hi: hi  # always returns upper bound
        delay_hi = router._backoff_delay(0)  # → 1.25
        self.assertAlmostEqual(delay_hi, 1.25)

    def test_jitter_never_negative(self) -> None:
        """Even with aggressive jitter, delay should never go negative."""
        router = AIRouter(backoff_base=0.01, backoff_max=100.0, jitter=1.0)
        # Force random to return below lower bound (edge case)
        router._random_fn = lambda lo, hi: -0.5
        delay = router._backoff_delay(0)
        self.assertGreaterEqual(delay, 0.0)

    @patch("urllib.request.urlopen")
    def test_jitter_backoff_values_vary(self, mock_urlopen: MagicMock) -> None:
        """With real random, repeated backoffs should not all be identical."""
        mock_urlopen.side_effect = [
            urllib.error.URLError("f1"),
            urllib.error.URLError("f2"),
            urllib.error.URLError("f3"),
            _anthropic_response("ok"),
        ]
        delays: list[float] = []

        def capture_sleep(seconds: float) -> None:
            delays.append(seconds)

        router = AIRouter(max_retries=3, backoff_base=1.0, jitter=0.25)
        router._sleep_fn = capture_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.complete("test")

        self.assertEqual(len(delays), 3)
        # With jitter, at least one delay should differ from the base formula
        # (extremely unlikely all three are exactly equal with real random)
        for d in delays:
            self.assertGreater(d, 0.0)


# ---------------------------------------------------------------------------
# 21. Native async (aiohttp path)
# ---------------------------------------------------------------------------

class TestNativeAsync(unittest.TestCase):
    """Test the native async code paths by mocking aiohttp."""

    def test_has_native_async_reflects_import(self) -> None:
        from quantarion_router import _HAS_AIOHTTP
        self.assertEqual(AIRouter.has_native_async(), _HAS_AIOHTTP)

    @patch("urllib.request.urlopen")
    def test_acomplete_fallback_without_aiohttp(self, mock_urlopen: MagicMock) -> None:
        """Without aiohttp, acomplete should use executor fallback."""
        mock_urlopen.return_value = _anthropic_response("executor path")

        import quantarion_router
        original = quantarion_router._HAS_AIOHTTP
        quantarion_router._HAS_AIOHTTP = False
        try:
            router = AIRouter()
            router.add_provider(_make_config(Provider.ANTHROPIC))
            result = asyncio.run(router.acomplete("test"))
            self.assertTrue(result.success)
            self.assertEqual(result.content, "executor path")
        finally:
            quantarion_router._HAS_AIOHTTP = original

    @patch("urllib.request.urlopen")
    def test_round_robin_with_executor_fallback_no_double_increment(
        self, mock_urlopen: MagicMock
    ) -> None:
        """
        Regression: ROUND_ROBIN + executor fallback must not double-increment
        the round-robin index. Previously acomplete() called
        _get_ordered_safe() once (index++) and then self.complete() inside the
        executor called it again (index++ again), causing providers to be
        skipped on every call.
        """
        mock_urlopen.return_value = _anthropic_response("ok")

        import quantarion_router
        original = quantarion_router._HAS_AIOHTTP
        quantarion_router._HAS_AIOHTTP = False
        try:
            router = AIRouter(strategy=RoutingStrategy.ROUND_ROBIN, max_retries=0)
            router.add_provider(_make_config(Provider.ANTHROPIC, name="a"))
            router.add_provider(_make_config(Provider.OPENAI, name="b"))

            # Three async calls through the executor fallback.
            # With the bug: index goes 0 -> 2 -> 4 -> 6 (jumps of 2).
            # Fixed: index goes 0 -> 1 -> 2 -> 3 (jumps of 1).
            asyncio.run(router.acomplete("1"))
            asyncio.run(router.acomplete("2"))
            asyncio.run(router.acomplete("3"))

            # After 3 calls with 2 providers, index should be exactly 3.
            self.assertEqual(router._round_robin_index, 3)

            # And both providers should have been called at least once
            # (the whole point of round-robin).
            metrics = router.get_metrics()
            self.assertGreater(metrics["a"]["calls"], 0)
            self.assertGreater(metrics["b"]["calls"], 0)
        finally:
            quantarion_router._HAS_AIOHTTP = original

    @patch("urllib.request.urlopen")
    def test_acomplete_with_system_fallback(self, mock_urlopen: MagicMock) -> None:
        """System prompt works through executor fallback path."""
        mock_urlopen.return_value = _anthropic_response("system ok")

        import quantarion_router
        original = quantarion_router._HAS_AIOHTTP
        quantarion_router._HAS_AIOHTTP = False
        try:
            router = AIRouter()
            router.add_provider(_make_config(Provider.ANTHROPIC))
            result = asyncio.run(router.acomplete("test", system="Be helpful"))
            self.assertTrue(result.success)
        finally:
            quantarion_router._HAS_AIOHTTP = original

    def test_acomplete_native_with_custom_handler(self) -> None:
        """Custom handlers in native async run via executor."""
        call_log: list[str] = []

        def my_handler(config, messages, system):  # type: ignore[no-untyped-def]
            call_log.append("called")
            return "custom async", 1, 2

        import quantarion_router
        original = quantarion_router._HAS_AIOHTTP

        # Simulate aiohttp available to enter native path
        quantarion_router._HAS_AIOHTTP = True
        try:
            router = AIRouter()
            router.add_provider(_make_config(Provider.CUSTOM, name="async-llm", model="v1"))
            router.register_handler("async-llm", my_handler)
            result = asyncio.run(router.acomplete("test"))
            self.assertTrue(result.success)
            self.assertEqual(result.content, "custom async")
            self.assertEqual(call_log, ["called"])
        finally:
            quantarion_router._HAS_AIOHTTP = original

    def test_acomplete_native_empty_prompt_raises(self) -> None:
        """Empty prompt should raise even in native async path."""
        import quantarion_router
        original = quantarion_router._HAS_AIOHTTP
        quantarion_router._HAS_AIOHTTP = True
        try:
            router = AIRouter()
            router.add_provider(_make_config(Provider.ANTHROPIC))
            with self.assertRaises(ValueError):
                asyncio.run(router.acomplete(""))
        finally:
            quantarion_router._HAS_AIOHTTP = original

    def test_acomplete_native_no_providers_raises(self) -> None:
        """No providers should raise ConfigurationError in native path."""
        import quantarion_router
        original = quantarion_router._HAS_AIOHTTP
        quantarion_router._HAS_AIOHTTP = True
        try:
            router = AIRouter()
            with self.assertRaises(ConfigurationError):
                asyncio.run(router.acomplete("test"))
        finally:
            quantarion_router._HAS_AIOHTTP = original

    def test_aclose_without_session(self) -> None:
        """aclose on router with no session should not raise."""
        router = AIRouter()
        asyncio.run(router.aclose())  # should be no-op

    def test_aio_session_initially_none(self) -> None:
        """Shared session starts as None (lazy init)."""
        router = AIRouter()
        self.assertIsNone(router._aio_session)


# ---------------------------------------------------------------------------
# 22. _route_loop helpers
# ---------------------------------------------------------------------------

class TestRouteLoopHelpers(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_record_attempt_success(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _anthropic_response("ok", input_tokens=3, output_tokens=7)
        router = AIRouter()
        router.add_provider(_make_config(Provider.ANTHROPIC))
        router.complete("test")
        m = router.get_metrics()["anthropic"]
        self.assertEqual(m["total_input_tokens"], 3)
        self.assertEqual(m["total_output_tokens"], 7)

    @patch("urllib.request.urlopen")
    def test_record_attempt_failure_no_tokens(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        router = AIRouter(max_retries=0)
        router._sleep_fn = _noop_sleep
        router.add_provider(_make_config(Provider.ANTHROPIC))
        with self.assertRaises(RouterError):
            router.complete("test")
        m = router.get_metrics()["anthropic"]
        self.assertEqual(m["total_input_tokens"], 0)
        self.assertEqual(m["total_output_tokens"], 0)

    @patch("urllib.request.urlopen")
    def test_aclose_resets_lock(self, mock_urlopen: MagicMock) -> None:
        """After aclose, _aio_lock should be None."""
        router = AIRouter()
        asyncio.run(router.aclose())
        self.assertIsNone(router._aio_lock)


# ---------------------------------------------------------------------------
# 23. Concurrent acomplete
# ---------------------------------------------------------------------------

class TestConcurrentAsync(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_concurrent_acomplete(self, mock_urlopen: MagicMock) -> None:
        """Multiple acomplete() calls concurrently should not crash."""
        mock_urlopen.return_value = _anthropic_response("ok")

        import quantarion_router as ar
        original = ar._HAS_AIOHTTP
        ar._HAS_AIOHTTP = False  # use executor fallback (no real aiohttp needed)
        try:
            router = AIRouter()
            router.add_provider(_make_config(Provider.ANTHROPIC))

            async def run_all() -> list:
                tasks = [router.acomplete("test") for _ in range(5)]
                return await asyncio.gather(*tasks)

            results = asyncio.run(run_all())
            self.assertEqual(len(results), 5)
            self.assertTrue(all(r.success for r in results))
        finally:
            ar._HAS_AIOHTTP = original

    def test_get_aio_lock_returns_same_instance_under_contention(self) -> None:
        """
        Regression: _get_aio_lock must return the SAME asyncio.Lock instance
        even under multi-threaded contention. Previously two threads (each
        running their own event loop) could both observe ``_aio_lock is None``
        and create separate locks, defeating the guard and causing aiohttp
        session leaks.

        The race is between THREADS (not tasks in one event loop, where GIL +
        cooperative scheduling serialises access). We spin up many threads,
        each running a tiny asyncio.run() that calls _get_aio_lock, and verify
        they all see the same object.
        """
        import threading

        router = AIRouter()
        results: list = []
        barrier = threading.Barrier(20)

        def worker() -> None:
            # All threads wait here, then release at once — maximises contention.
            barrier.wait()

            async def grab() -> int:
                return id(router._get_aio_lock())

            results.append(asyncio.run(grab()))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 20)
        self.assertEqual(
            len(set(results)), 1,
            f"Expected 1 unique lock across 20 threads, got {len(set(results))}",
        )

    @patch("urllib.request.urlopen")
    def test_aclose_clears_state_atomically(self, mock_urlopen: MagicMock) -> None:
        """
        After aclose(), both the session reference and the lock reference
        must be cleared, and a subsequent _get_aio_lock() must create a
        fresh lock (not reuse a dangling reference).
        """
        import quantarion_router as ar
        original = ar._HAS_AIOHTTP
        ar._HAS_AIOHTTP = False
        try:
            router = AIRouter()
            router.add_provider(_make_config(Provider.ANTHROPIC))

            async def scenario() -> tuple:
                first_lock = router._get_aio_lock()
                await router.aclose()
                # After aclose, state must be cleared.
                cleared_lock = router._aio_lock
                cleared_session = router._aio_session
                # And a new lock can be created without error.
                second_lock = router._get_aio_lock()
                return first_lock, cleared_lock, cleared_session, second_lock

            first, cleared, sess, second = asyncio.run(scenario())
            self.assertIsNone(cleared)
            self.assertIsNone(sess)
            self.assertIsNotNone(second)
            # The new lock is a genuinely new object, not the old one.
            self.assertIsNot(first, second)
        finally:
            ar._HAS_AIOHTTP = original


if __name__ == "__main__":
    unittest.main()
