"""
QUANTARION Router — Multi-Provider LLM Routing Engine
======================================================
Routes requests across multiple AI providers (Anthropic, OpenAI, custom)
with automatic fallback, retry logic with exponential backoff + jitter,
circuit breaker, cost tracking, response metrics, and native async support.

Zero external dependencies by default (pure Python standard library).
Install ``aiohttp`` for native async I/O; otherwise async falls back
to running synchronous calls in a thread executor.

Part of QUANTARION Labs — consciousness-grade AI architectures.
https://quantarion.com

Author: Vlad M.
License: MIT
"""

from __future__ import annotations

import asyncio
import enum
import functools
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    import aiohttp  # type: ignore[import-not-found]
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


logger = logging.getLogger(__name__)

__all__ = [
    "AIRouter",
    "Provider",
    "ProviderConfig",
    "RoutingStrategy",
    "RouteResult",
    "CircuitBreakerConfig",
    "RouterError",
    "CircuitOpenError",
    "ProviderNotFoundError",
    "ConfigurationError",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Provider(enum.Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"


class RoutingStrategy(enum.Enum):
    """Strategy for selecting a provider."""

    PRIMARY = "primary"          # Always use first available
    FALLBACK = "fallback"        # Try in order until success
    ROUND_ROBIN = "round_robin"  # Cycle through providers


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RouterError(Exception):
    """Raised when all providers fail."""


class ProviderNotFoundError(Exception):
    """Raised when referencing a non-existent provider."""


class ConfigurationError(Exception):
    """Raised on invalid router configuration."""


class CircuitOpenError(RouterError):
    """Raised when circuit breaker is open for a provider."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

Message = Dict[str, str]  # {"role": "...", "content": "..."}


@dataclass
class ProviderConfig:
    """Configuration for a single AI provider."""

    provider: Provider
    api_key: str
    model: str
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    enabled: bool = True
    base_url: Optional[str] = None      # Custom endpoint URL
    headers: Optional[Dict[str, str]] = None  # Extra headers
    name: str = ""  # Display name (defaults to provider.value)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.provider.value


@dataclass
class RouteResult:
    """Result of a single routing attempt."""

    success: bool
    provider: Provider
    model: str
    content: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider_name: str = ""


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker per provider."""

    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 30.0   # Seconds before half-open
    half_open_max: int = 1           # Max test requests in half-open


@dataclass
class _CircuitState:
    """Internal circuit breaker state."""

    failures: int = 0
    state: str = "closed"   # closed | open | half-open
    last_failure_time: float = 0.0
    half_open_calls: int = 0


@dataclass
class _ProviderMetrics:
    """Accumulated metrics for a single provider."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def avg_ms(self) -> float:
        """Average response time in milliseconds."""
        return self.total_ms / self.calls if self.calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0–1.0)."""
        return self.successes / self.calls if self.calls > 0 else 0.0


# ---------------------------------------------------------------------------
# Provider call functions (built-in)
# ---------------------------------------------------------------------------

# Signature: (config, messages, system) -> (content, input_tokens, output_tokens)
ProviderCallFn = Callable[
    [ProviderConfig, Sequence[Message], Optional[str]],
    "tuple[str, int, int]",
]


def _format_api_error(status: int, reason: str, raw: Optional[str]) -> str:
    """
    Format a provider HTTP error into an actionable message.

    Accepts the raw response body text (may be ``None`` or empty) and attempts
    to parse structured provider errors like
    ``{"error": {"message": "Invalid API key", "type": "invalid_request"}}``.
    Falls back to a trimmed snippet if the body is not JSON.
    """
    label = reason or "HTTP Error"
    if not raw:
        return f"HTTP {status} {label}"

    try:
        body = json.loads(raw)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return f"HTTP {status}: {err['message']}"
            if isinstance(err, str):
                return f"HTTP {status}: {err}"
    except (json.JSONDecodeError, ValueError):
        pass

    snippet = raw.strip()
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."
    return f"HTTP {status} {label}: {snippet}" if snippet else f"HTTP {status} {label}"


def _extract_http_error(exc: urllib.error.HTTPError) -> str:
    """Extract a useful error message from a urllib HTTPError."""
    try:
        raw = exc.read()
    except Exception:
        return f"HTTP {exc.code} {exc.reason or 'HTTP Error'}"

    if not raw:
        return f"HTTP {exc.code} {exc.reason or 'HTTP Error'}"

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return f"HTTP {exc.code} {exc.reason or 'HTTP Error'}"

    return _format_api_error(exc.code, exc.reason or "HTTP Error", text)


def _call_anthropic(
    config: ProviderConfig,
    messages: Sequence[Message],
    system: Optional[str],
) -> tuple[str, int, int]:
    """Call Anthropic Messages API."""
    body: Dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": list(messages),
    }
    if system:
        body["system"] = system

    payload = json.dumps(body).encode("utf-8")
    url = config.base_url or "https://api.anthropic.com/v1/messages"
    hdrs = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if config.headers:
        hdrs.update(config.headers)

    req = urllib.request.Request(url, data=payload, headers=hdrs, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
            data: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RouterError(f"Anthropic {_extract_http_error(exc)}") from exc

    content_blocks = data.get("content", [])
    if not content_blocks:
        raise RouterError("Anthropic returned empty content")

    text = content_blocks[0].get("text", "")
    usage = data.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _call_openai(
    config: ProviderConfig,
    messages: Sequence[Message],
    system: Optional[str],
) -> tuple[str, int, int]:
    """Call OpenAI Chat Completions API."""
    msgs: List[Message] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    payload = json.dumps({
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": msgs,
    }).encode("utf-8")

    url = config.base_url or "https://api.openai.com/v1/chat/completions"
    hdrs = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    if config.headers:
        hdrs.update(config.headers)

    req = urllib.request.Request(url, data=payload, headers=hdrs, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RouterError(f"OpenAI {_extract_http_error(exc)}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise RouterError("OpenAI returned empty choices")

    text = choices[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


_BUILTIN_PROVIDERS: Dict[Provider, ProviderCallFn] = {
    Provider.ANTHROPIC: _call_anthropic,
    Provider.OPENAI: _call_openai,
}


# ---------------------------------------------------------------------------
# AIRouter
# ---------------------------------------------------------------------------

class AIRouter:
    """
    Multi-provider AI routing engine with fallback, metrics, and async.

    Routes LLM requests across Anthropic, OpenAI, and custom providers
    with automatic fallback, retry logic with exponential backoff,
    circuit breaker, per-provider metrics, and token tracking.

    Example:
        router = AIRouter(strategy=RoutingStrategy.FALLBACK)
        router.add_provider(ProviderConfig(
            provider=Provider.ANTHROPIC,
            api_key="sk-ant-...",
            model="claude-sonnet-4-20250514",
        ))
        result = router.complete("Explain REST APIs in one sentence.")
        print(result.content)

    Async example:
        result = await router.acomplete("Explain REST APIs.")
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.FALLBACK,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_max: float = 10.0,
        jitter: float = 0.25,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        if not isinstance(strategy, RoutingStrategy):
            raise TypeError(
                f"strategy must be RoutingStrategy, got {type(strategy).__name__}"
            )
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError(
                f"max_retries must be non-negative int, got {max_retries}"
            )
        if backoff_base < 0:
            raise ValueError(f"backoff_base must be non-negative, got {backoff_base}")
        if backoff_max < 0:
            raise ValueError(f"backoff_max must be non-negative, got {backoff_max}")
        if not (0.0 <= jitter <= 1.0):
            raise ValueError(f"jitter must be between 0.0 and 1.0, got {jitter}")

        self._strategy = strategy
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._jitter = jitter
        self._cb_config = circuit_breaker
        self._providers: List[ProviderConfig] = []
        self._metrics: Dict[str, _ProviderMetrics] = {}
        self._circuits: Dict[str, _CircuitState] = {}
        self._round_robin_index: int = 0
        self._lock = threading.Lock()
        self._custom_handlers: Dict[str, ProviderCallFn] = {}
        self._sleep_fn: Callable[[float], None] = time.sleep  # injectable for tests
        self._async_sleep_fn: Callable[[float], Any] = asyncio.sleep  # injectable for tests
        self._random_fn: Callable[[float, float], float] = random.uniform  # injectable
        self._aio_session: Optional[Any] = None  # lazy aiohttp.ClientSession
        self._aio_lock: Optional[asyncio.Lock] = None  # lazy asyncio.Lock

    # -- Properties ----------------------------------------------------------

    @property
    def strategy(self) -> RoutingStrategy:
        """Current routing strategy."""
        return self._strategy

    @property
    def provider_count(self) -> int:
        """Number of registered providers."""
        with self._lock:
            return len(self._providers)

    # -- Provider registration -----------------------------------------------

    def add_provider(self, config: ProviderConfig) -> "AIRouter":
        """
        Register an AI provider.

        Args:
            config: Provider configuration.

        Returns:
            Self for method chaining.

        Raises:
            TypeError: If config is not a ProviderConfig.
            ConfigurationError: If provider name is already registered.
        """
        if not isinstance(config, ProviderConfig):
            raise TypeError(
                f"config must be ProviderConfig, got {type(config).__name__}"
            )

        if not config.api_key or not isinstance(config.api_key, str):
            raise ConfigurationError("api_key must be a non-empty string")

        if not config.model or not isinstance(config.model, str):
            raise ConfigurationError("model must be a non-empty string")

        if config.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")

        if config.max_tokens <= 0:
            raise ConfigurationError("max_tokens must be positive")

        with self._lock:
            existing_names = [p.name for p in self._providers]
            if config.name in existing_names:
                raise ConfigurationError(
                    f"Provider '{config.name}' is already registered"
                )
            self._providers.append(config)
            self._metrics[config.name] = _ProviderMetrics()
            if self._cb_config:
                self._circuits[config.name] = _CircuitState()

        logger.info("Registered provider '%s' (model=%s)", config.name, config.model)
        return self

    def register_handler(self, name: str, handler: ProviderCallFn) -> "AIRouter":
        """
        Register a custom provider call handler.

        The handler signature must be:
            (config, messages, system) -> (content, input_tokens, output_tokens)

        Args:
            name: Provider name (must match ProviderConfig.name).
            handler: Callable that performs the API call.

        Returns:
            Self for method chaining.
        """
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._custom_handlers[name] = handler
        logger.info("Registered custom handler for '%s'", name)
        return self

    # -- Provider ordering ---------------------------------------------------

    def _get_ordered_providers(self) -> List[ProviderConfig]:
        """Return providers in order based on strategy (caller holds lock)."""
        enabled = [p for p in self._providers if p.enabled]

        if not enabled:
            return []

        if self._strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._round_robin_index % len(enabled)
            self._round_robin_index += 1
            return enabled[idx:] + enabled[:idx]

        return enabled  # PRIMARY and FALLBACK use registration order

    # -- Circuit breaker -----------------------------------------------------

    def _check_circuit(self, name: str) -> None:
        """Check circuit breaker state. Raises CircuitOpenError if open."""
        if not self._cb_config or name not in self._circuits:
            return

        with self._lock:
            cb = self._circuits[name]

            if cb.state == "open":
                elapsed = time.monotonic() - cb.last_failure_time
                if elapsed >= self._cb_config.recovery_timeout:
                    cb.state = "half-open"
                    cb.half_open_calls = 0
                    # Reset failure counter on state transition so a single
                    # half-open failure doesn't leave the counter at
                    # threshold+1 (causing spurious "threshold" log on the
                    # next failure instead of a clean re-open).
                    cb.failures = 0
                    logger.info("Circuit for '%s' -> half-open", name)
                else:
                    raise CircuitOpenError(
                        f"Circuit open for '{name}' "
                        f"(retry in {self._cb_config.recovery_timeout - elapsed:.1f}s)"
                    )

            if cb.state == "half-open":
                if cb.half_open_calls >= self._cb_config.half_open_max:
                    raise CircuitOpenError(
                        f"Circuit half-open for '{name}', max test calls reached"
                    )
                cb.half_open_calls += 1

    def _record_circuit_success(self, name: str) -> None:
        """Record success — close the circuit."""
        if not self._cb_config or name not in self._circuits:
            return
        with self._lock:
            cb = self._circuits[name]
            if cb.state != "closed":
                logger.info("Circuit for '%s' -> closed (success)", name)
            cb.failures = 0
            cb.state = "closed"

    def _record_circuit_failure(self, name: str) -> None:
        """Record failure — potentially open the circuit."""
        if not self._cb_config or name not in self._circuits:
            return
        with self._lock:
            cb = self._circuits[name]
            cb.failures += 1
            cb.last_failure_time = time.monotonic()

            if cb.state == "half-open":
                cb.state = "open"
                logger.warning("Circuit for '%s' -> open (half-open fail)", name)
            elif cb.failures >= self._cb_config.failure_threshold:
                cb.state = "open"
                logger.warning(
                    "Circuit for '%s' -> open (threshold %d)",
                    name, self._cb_config.failure_threshold,
                )

    # -- Provider dispatch ---------------------------------------------------

    def _call_provider(
        self,
        config: ProviderConfig,
        messages: Sequence[Message],
        system: Optional[str],
    ) -> tuple[str, int, int]:
        """Dispatch to correct provider API."""
        if config.name in self._custom_handlers:
            return self._custom_handlers[config.name](config, messages, system)

        handler = _BUILTIN_PROVIDERS.get(config.provider)
        if handler:
            return handler(config, messages, system)

        raise RouterError(
            f"No handler for provider '{config.name}' ({config.provider.value})"
        )

    # -- Backoff -------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter and cap.

        Formula: delay = base * 2^attempt, capped at max,
        then ± jitter% randomization applied.
        Example with jitter=0.25: delay 2.0s → uniform(1.5, 2.5).
        """
        delay: float = self._backoff_base * (2 ** attempt)
        delay = float(min(delay, self._backoff_max))
        if self._jitter > 0 and delay > 0:
            spread = delay * self._jitter
            delay = self._random_fn(delay - spread, delay + spread)
            delay = max(0.0, delay)  # safety: never negative
        return delay

    # -- Input normalization -------------------------------------------------

    @staticmethod
    def _normalize_input(
        prompt: Union[str, Sequence[Message]],
    ) -> Sequence[Message]:
        """Convert str prompt to messages list."""
        if isinstance(prompt, str):
            if not prompt:
                raise ValueError("prompt must be a non-empty string")
            return [{"role": "user", "content": prompt}]

        if not prompt:
            raise ValueError("messages must be a non-empty sequence")

        return prompt

    # -- Core routing --------------------------------------------------------

    def _get_ordered_safe(self) -> List[ProviderConfig]:
        """
        Validate providers and return ordered list.

        Returns:
            Ordered list of enabled providers.

        Raises:
            ConfigurationError: If no providers are registered or none enabled.
        """
        with self._lock:
            if not self._providers:
                raise ConfigurationError("No providers registered")
            ordered = self._get_ordered_providers()
        if not ordered:
            raise ConfigurationError("No enabled providers available")
        return ordered

    def _record_attempt(
        self,
        name: str,
        elapsed: float,
        success: bool,
        in_tok: int = 0,
        out_tok: int = 0,
    ) -> None:
        """Update metrics for one attempt (thread-safe)."""
        with self._lock:
            m = self._metrics[name]
            m.total_ms += elapsed
            if success:
                m.successes += 1
                m.total_input_tokens += in_tok
                m.total_output_tokens += out_tok
            else:
                m.failures += 1

    def _increment_calls(self, name: str) -> None:
        """Increment call counter (thread-safe)."""
        with self._lock:
            self._metrics[name].calls += 1

    def _run_route_loop(
        self,
        ordered: List[ProviderConfig],
        call_fn: "Callable[[ProviderConfig, Sequence[Message], Optional[str]], tuple[str, int, int]]",
        messages: Sequence[Message],
        system: Optional[str],
        sleep_fn: "Callable[[float], None]",
        label: str = "",
    ) -> RouteResult:
        """
        Shared routing loop used by both complete() and the sync fallback.

        Iterates providers in order, retries with backoff, tracks metrics,
        manages circuit breaker, and returns the first successful RouteResult.

        Args:
            ordered: Providers to try, in order.
            call_fn: Synchronous provider call function.
            messages: Normalised message list.
            system: Optional system prompt.
            sleep_fn: Sleep callable (injectable for tests).
            label: Log prefix e.g. "async" for acomplete fallback.

        Raises:
            RouterError: If all providers fail.
        """
        last_error: Optional[str] = None
        total_attempts = 0
        prefix = f"{label} " if label else ""
        # Track whether every provider was skipped by the circuit breaker.
        # If so, we raise CircuitOpenError (subclass of RouterError) so callers
        # can distinguish "all providers really failed" from "all temporarily
        # circuit-protected and will recover soon".
        all_circuit_open = True
        had_any_skip = False

        for config in ordered:
            try:
                self._check_circuit(config.name)
            except CircuitOpenError as e:
                logger.debug("Skipping '%s': %s", config.name, e)
                last_error = str(e)
                had_any_skip = True
                continue

            # At least one provider was actually attempted — not all-skipped.
            all_circuit_open = False

            for attempt in range(1 + self._max_retries):
                total_attempts += 1
                self._increment_calls(config.name)
                start = time.perf_counter()

                try:
                    content, in_tok, out_tok = call_fn(config, messages, system)
                    elapsed = (time.perf_counter() - start) * 1000

                    self._record_attempt(config.name, elapsed, True, in_tok, out_tok)
                    self._record_circuit_success(config.name)

                    logger.debug(
                        "'%s' %sresponded in %.1fms (%d+%d tok)",
                        config.name, prefix, elapsed, in_tok, out_tok,
                    )

                    return RouteResult(
                        success=True,
                        provider=config.provider,
                        model=config.model,
                        content=content,
                        attempts=total_attempts,
                        duration_ms=elapsed,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        provider_name=config.name,
                    )

                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000
                    self._record_attempt(config.name, elapsed, False)
                    self._record_circuit_failure(config.name)
                    last_error = f"[{config.name}] {exc}"

                    logger.warning(
                        "'%s' %sattempt %d/%d failed: %s",
                        config.name, prefix, attempt + 1, 1 + self._max_retries, exc,
                    )

                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.debug("%sbackoff %.2fs before retry", prefix, delay)
                        sleep_fn(delay)

            if self._strategy == RoutingStrategy.PRIMARY:
                break

        if had_any_skip and all_circuit_open:
            raise CircuitOpenError(
                f"All providers circuit-open. Last state: {last_error}"
            )
        raise RouterError(f"All providers failed. Last error: {last_error}")

    def complete(
        self,
        prompt: Union[str, Sequence[Message]],
        system: Optional[str] = None,
    ) -> RouteResult:
        """
        Send prompt to AI provider(s) and return response.

        Tries providers in order based on strategy. On failure,
        retries with exponential backoff before moving to next provider.

        Args:
            prompt: Text prompt (str) or list of message dicts.
            system: Optional system prompt.

        Returns:
            RouteResult with response content and metrics.

        Raises:
            RouterError: If all providers fail.
            CircuitOpenError: If every provider is skipped by the circuit
                breaker (subclass of RouterError, so catching RouterError
                still works).
            ValueError: If prompt is empty.
            ConfigurationError: If no providers are registered.
        """
        messages = self._normalize_input(prompt)
        ordered = self._get_ordered_safe()
        return self._run_route_loop(
            ordered, self._call_provider, messages, system, self._sleep_fn,
        )

    # -- Native async --------------------------------------------------------

    def _get_aio_lock(self) -> "asyncio.Lock":
        """
        Return the asyncio lock, creating it lazily in the running event loop.

        Uses double-checked locking with the router's threading lock to guard
        the one-time creation of ``_aio_lock``. Without this, two concurrent
        tasks calling ``_get_session`` on a freshly-constructed router could
        each read ``self._aio_lock is None`` and create separate asyncio locks,
        defeating the lock's purpose and leaking aiohttp sessions.
        """
        # Fast path: already initialised, no locking needed (atomic read).
        if self._aio_lock is not None:
            return self._aio_lock
        # Slow path: serialise creation across threads/tasks.
        with self._lock:
            if self._aio_lock is None:
                self._aio_lock = asyncio.Lock()
            return self._aio_lock

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create a shared aiohttp session (lazy, async-safe init)."""
        async with self._get_aio_lock():
            if self._aio_session is None or self._aio_session.closed:
                self._aio_session = aiohttp.ClientSession()
        return self._aio_session

    async def aclose(self) -> None:
        """Close the shared aiohttp session. Call when done with async ops."""
        # Capture the current session and lock atomically, then clear the
        # references under the threading lock so a concurrent _get_session
        # cannot observe a half-torn-down state.
        with self._lock:
            session = self._aio_session
            self._aio_session = None
            self._aio_lock = None
        if session is not None and not session.closed:
            await session.close()

    async def _acall_anthropic(
        self,
        config: ProviderConfig,
        messages: Sequence[Message],
        system: Optional[str],
    ) -> tuple[str, int, int]:
        """Native async call to Anthropic via aiohttp."""
        body: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": list(messages),
        }
        if system:
            body["system"] = system

        url = config.base_url or "https://api.anthropic.com/v1/messages"
        hdrs = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if config.headers:
            hdrs.update(config.headers)

        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        session = await self._get_session()
        async with session.post(url, json=body, headers=hdrs, timeout=timeout) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RouterError(
                    f"Anthropic {_format_api_error(resp.status, resp.reason or '', text)}"
                )
            data: Dict[str, Any] = await resp.json()

        content_blocks = data.get("content", [])
        if not content_blocks:
            raise RouterError("Anthropic returned empty content")

        text = content_blocks[0].get("text", "")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    async def _acall_openai(
        self,
        config: ProviderConfig,
        messages: Sequence[Message],
        system: Optional[str],
    ) -> tuple[str, int, int]:
        """Native async call to OpenAI via aiohttp."""
        msgs: List[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        payload = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": msgs,
        }

        url = config.base_url or "https://api.openai.com/v1/chat/completions"
        hdrs = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        if config.headers:
            hdrs.update(config.headers)

        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        session = await self._get_session()
        async with session.post(url, json=payload, headers=hdrs, timeout=timeout) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RouterError(
                    f"OpenAI {_format_api_error(resp.status, resp.reason or '', text)}"
                )
            data = await resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise RouterError("OpenAI returned empty choices")

        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    async def _acall_provider(
        self,
        config: ProviderConfig,
        messages: Sequence[Message],
        system: Optional[str],
    ) -> tuple[str, int, int]:
        """Async dispatch to correct provider API."""
        if config.name in self._custom_handlers:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(
                    self._custom_handlers[config.name], config, messages, system,
                ),
            )

        if config.provider == Provider.ANTHROPIC:
            return await self._acall_anthropic(config, messages, system)
        elif config.provider == Provider.OPENAI:
            return await self._acall_openai(config, messages, system)

        raise RouterError(
            f"No async handler for provider '{config.name}' ({config.provider.value})"
        )

    async def _async_route_loop(
        self,
        ordered: List[ProviderConfig],
        messages: Sequence[Message],
        system: Optional[str],
    ) -> RouteResult:
        """
        Async version of _run_route_loop — uses await for backoff and provider calls.

        Args:
            ordered: Providers to try, in order.
            messages: Normalised message list.
            system: Optional system prompt.

        Raises:
            RouterError: If all providers fail.
        """
        last_error: Optional[str] = None
        total_attempts = 0
        # See _run_route_loop for rationale.
        all_circuit_open = True
        had_any_skip = False

        for config in ordered:
            try:
                self._check_circuit(config.name)
            except CircuitOpenError as e:
                logger.debug("Skipping '%s': %s", config.name, e)
                last_error = str(e)
                had_any_skip = True
                continue

            all_circuit_open = False

            for attempt in range(1 + self._max_retries):
                total_attempts += 1
                self._increment_calls(config.name)
                start = time.perf_counter()

                try:
                    content, in_tok, out_tok = await self._acall_provider(
                        config, messages, system,
                    )
                    elapsed = (time.perf_counter() - start) * 1000

                    self._record_attempt(config.name, elapsed, True, in_tok, out_tok)
                    self._record_circuit_success(config.name)

                    logger.debug(
                        "'%s' async responded in %.1fms (%d+%d tok)",
                        config.name, elapsed, in_tok, out_tok,
                    )

                    return RouteResult(
                        success=True,
                        provider=config.provider,
                        model=config.model,
                        content=content,
                        attempts=total_attempts,
                        duration_ms=elapsed,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        provider_name=config.name,
                    )

                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000
                    self._record_attempt(config.name, elapsed, False)
                    self._record_circuit_failure(config.name)
                    last_error = f"[{config.name}] {exc}"

                    logger.warning(
                        "'%s' async attempt %d/%d failed: %s",
                        config.name, attempt + 1, 1 + self._max_retries, exc,
                    )

                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.debug("Async backoff %.2fs before retry", delay)
                        await self._async_sleep_fn(delay)

            if self._strategy == RoutingStrategy.PRIMARY:
                break

        if had_any_skip and all_circuit_open:
            raise CircuitOpenError(
                f"All providers circuit-open. Last state: {last_error}"
            )
        raise RouterError(f"All providers failed. Last error: {last_error}")

    async def acomplete(
        self,
        prompt: Union[str, Sequence[Message]],
        system: Optional[str] = None,
    ) -> RouteResult:
        """
        Async version of complete().

        If ``aiohttp`` is installed, uses native async I/O (non-blocking)
        with a shared, lazily-initialised ``aiohttp.ClientSession``.
        Otherwise, falls back to running synchronous ``complete()`` in a
        thread executor — still non-blocking for the event loop.

        Call ``await router.aclose()`` when done to release the session.

        Args:
            prompt: Text prompt (str) or list of message dicts.
            system: Optional system prompt.

        Returns:
            RouteResult with response content and metrics.

        Raises:
            RouterError: If all providers fail.
            CircuitOpenError: If every provider is skipped by the circuit
                breaker (subclass of RouterError).
            ValueError: If prompt is empty.
            ConfigurationError: If no providers are registered.
        """
        if not _HAS_AIOHTTP:
            # Executor fallback: delegate entirely to self.complete().
            # Do NOT call _normalize_input / _get_ordered_safe here — complete()
            # does that itself, and pre-calling would double-increment the
            # round-robin index and validate providers twice.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(self.complete, prompt, system=system),
            )

        messages = self._normalize_input(prompt)
        ordered = self._get_ordered_safe()
        return await self._async_route_loop(ordered, messages, system)

    # -- Enable / Disable ----------------------------------------------------

    def _find_provider(self, provider: Union[Provider, str]) -> Optional[ProviderConfig]:
        """
        Find a single registered provider by enum or name string.

        Returns the first match. Kept for backwards compatibility; prefer
        ``_find_providers_all`` for enum lookups when multiple providers may
        share the same ``Provider`` enum under different names.
        """
        matches = self._find_providers_all(provider)
        return matches[0] if matches else None

    def _find_providers_all(
        self, provider: Union[Provider, str]
    ) -> List[ProviderConfig]:
        """
        Find all registered providers matching enum or name string.

        Semantics:
          - String: exact match on ``ProviderConfig.name`` (at most one).
          - Enum:   all providers whose ``ProviderConfig.provider`` equals it
                    (can be several if the same vendor is registered under
                    different names, e.g. "cheap-claude" and "smart-claude").
        """
        if isinstance(provider, Provider):
            return [p for p in self._providers if p.provider == provider]
        return [p for p in self._providers if p.name == provider]

    def disable_provider(self, provider: Union[Provider, str]) -> None:
        """
        Disable a provider (exclude from routing).

        Args:
            provider: Provider enum disables ALL registered configs of that
                vendor; provider name string disables only the exact match.

        Raises:
            ProviderNotFoundError: If no matching provider is registered.
        """
        with self._lock:
            configs = self._find_providers_all(provider)
            if not configs:
                raise ProviderNotFoundError(
                    f"Provider '{provider if isinstance(provider, str) else provider.value}' not registered"
                )
            for config in configs:
                config.enabled = False
                logger.info("Disabled provider '%s'", config.name)

    def enable_provider(self, provider: Union[Provider, str]) -> None:
        """
        Re-enable a previously disabled provider.

        Args:
            provider: Provider enum enables ALL registered configs of that
                vendor; provider name string enables only the exact match.

        Raises:
            ProviderNotFoundError: If no matching provider is registered.
        """
        with self._lock:
            configs = self._find_providers_all(provider)
            if not configs:
                raise ProviderNotFoundError(
                    f"Provider '{provider if isinstance(provider, str) else provider.value}' not registered"
                )
            for config in configs:
                config.enabled = True
                logger.info("Enabled provider '%s'", config.name)

    # -- Metrics -------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get execution metrics for all providers.

        The returned snapshot is taken under the router's lock and is
        internally consistent, but note the invariant:

            ``calls >= successes + failures``

        because ``calls`` is incremented at the start of each attempt and
        ``successes``/``failures`` only after it completes. A snapshot taken
        during an in-flight request will reflect this (e.g. calls=5 with
        successes+failures=4). Once the request finishes the counters
        reconcile.
        """
        with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for name, m in self._metrics.items():
                result[name] = {
                    "calls": m.calls,
                    "successes": m.successes,
                    "failures": m.failures,
                    "success_rate": round(m.success_rate, 3),
                    "avg_ms": round(m.avg_ms, 3),
                    "total_ms": round(m.total_ms, 3),
                    "total_input_tokens": m.total_input_tokens,
                    "total_output_tokens": m.total_output_tokens,
                }
            return result

    def reset_metrics(self) -> None:
        """Reset all provider metrics to zero."""
        with self._lock:
            for m in self._metrics.values():
                m.calls = 0
                m.successes = 0
                m.failures = 0
                m.total_ms = 0.0
                m.total_input_tokens = 0
                m.total_output_tokens = 0

    def get_circuit_states(self) -> Dict[str, str]:
        """Get circuit breaker states for all providers."""
        with self._lock:
            return {name: cs.state for name, cs in self._circuits.items()}

    @staticmethod
    def has_native_async() -> bool:
        """Return True if aiohttp is available for native async I/O."""
        return _HAS_AIOHTTP

    def __repr__(self) -> str:
        # Snapshot provider names under the lock to avoid
        # "list changed size during iteration" if add_provider() runs concurrently.
        with self._lock:
            provider_names = [p.name for p in self._providers]
        return (
            f"AIRouter(strategy='{self._strategy.value}', "
            f"providers={provider_names}, "
            f"max_retries={self._max_retries})"
        )
