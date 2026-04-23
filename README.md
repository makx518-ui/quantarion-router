# quantarion-router

**Production-grade LLM routing engine. Zero required dependencies. 124 tests. mypy --strict clean.**

[![Tests](https://github.com/quantarion-labs/quantarion-router/actions/workflows/tests.yml/badge.svg)](https://github.com/quantarion-labs/quantarion-router/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-brightgreen)](https://mypy-lang.org/)

Part of [QUANTARION Labs](https://quantarion.com) — consciousness-grade AI architectures.

Routes LLM requests across Anthropic, OpenAI, and custom providers with automatic fallback,
exponential backoff with jitter, circuit breaker, per-provider metrics, token tracking, and
native async support.

## Why this exists

Every production LLM app needs the same infrastructure layer: retry on 429, fail over to a
second provider when the first goes down, stop hammering a degraded endpoint, track how much
was actually spent per provider. Most projects rebuild this from scratch. `quantarion-router`
is that layer, extracted, hardened, tested.

## Features

- **Three routing strategies** — Primary, Fallback, Round-Robin
- **Automatic fallback** across providers on failure
- **Exponential backoff + jitter** — configurable, injectable for tests
- **Circuit breaker** with open/half-open/closed states — stops hammering failing providers
- **Native async** via `aiohttp` (optional) with transparent executor fallback
- **Thread-safe** — safe for concurrent use from multiple threads
- **Custom providers** — plug in any LLM via a simple handler function
- **Custom base URLs** — proxy through OpenRouter, self-hosted endpoints
- **Actionable error messages** — parses provider JSON errors (e.g. `HTTP 401: invalid x-api-key`)
  instead of swallowing the body
- **Token + latency + success-rate metrics** per provider
- **Zero required dependencies** (`aiohttp` is optional for native async)

## Installation

```bash
pip install quantarion-router

# Optional: native async I/O
pip install quantarion-router[async]
```

## Quick Start

```python
from quantarion_router import AIRouter, Provider, ProviderConfig, RoutingStrategy

router = AIRouter(strategy=RoutingStrategy.FALLBACK, max_retries=2)

router.add_provider(ProviderConfig(
    provider=Provider.ANTHROPIC,
    api_key="sk-ant-...",
    model="claude-3-5-haiku-latest",
))

router.add_provider(ProviderConfig(
    provider=Provider.OPENAI,
    api_key="sk-...",
    model="gpt-4o-mini",
))

result = router.complete("Explain REST APIs in one sentence.")
print(result.content)
print(result.provider_name)   # which provider actually responded
print(result.duration_ms)     # latency in ms
print(result.input_tokens, result.output_tokens)
```

## Routing Strategies

### Fallback (recommended for production)

```python
router = AIRouter(strategy=RoutingStrategy.FALLBACK, max_retries=1)
# Tries Anthropic (with 1 retry); if both attempts fail, tries OpenAI.
```

### Primary (single provider, with retries)

```python
router = AIRouter(strategy=RoutingStrategy.PRIMARY, max_retries=3)
# Retries the first provider up to 3 times. Never falls through.
```

### Round-Robin (load distribution)

```python
router = AIRouter(strategy=RoutingStrategy.ROUND_ROBIN)
# Alternates: Anthropic -> OpenAI -> Anthropic -> OpenAI ...
```

## Error Handling

The router distinguishes "all providers really failed" from "all providers temporarily
circuit-protected":

```python
from quantarion_router import RouterError, CircuitOpenError

try:
    result = router.complete("...")
except CircuitOpenError:
    # All providers are in their recovery window - retry later.
    pass
except RouterError as e:
    # Actual failure - surface to user / alerting.
    print(e)  # e.g. "Anthropic HTTP 401: invalid x-api-key"
```

`CircuitOpenError` is a subclass of `RouterError`, so existing code that catches
`RouterError` continues to work.

## Multi-turn Conversations

```python
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And times 3?"},
]
result = router.complete(messages, system="You are a math tutor.")
```

## Async

```python
import asyncio

async def main():
    router = AIRouter()
    router.add_provider(ProviderConfig(provider=Provider.ANTHROPIC, ...))

    result = await router.acomplete("Hello!")
    print(result.content)

    await router.aclose()  # release aiohttp session

asyncio.run(main())
```

If `aiohttp` is installed, `acomplete()` uses it natively. Otherwise it transparently falls
back to running the sync path in a thread executor — non-blocking either way.

## Circuit Breaker

```python
from quantarion_router import CircuitBreakerConfig

router = AIRouter(
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,    # open after 5 consecutive failures
        recovery_timeout=30.0,  # attempt recovery after 30s
        half_open_max=1,        # allow 1 test request in half-open
    )
)

# Inspect state
print(router.get_circuit_states())
# {'anthropic': 'closed', 'openai': 'open'}
```

## Custom Providers

```python
def groq_handler(config, messages, system):
    # Your API call here - return (content, input_tokens, output_tokens).
    ...
    return content, in_tok, out_tok

router = AIRouter()
router.add_provider(ProviderConfig(
    provider=Provider.CUSTOM,
    api_key="gsk-...",
    model="llama-3.3-70b-versatile",
    name="groq",
))
router.register_handler("groq", groq_handler)

result = router.complete("Hello!")
```

## Proxies, OpenRouter, self-hosted endpoints

```python
router.add_provider(ProviderConfig(
    provider=Provider.OPENAI,       # OpenAI-compatible schema
    api_key="sk-or-...",
    model="anthropic/claude-3.5-haiku",
    base_url="https://openrouter.ai/api/v1/chat/completions",
    headers={"HTTP-Referer": "https://myapp.com"},
    name="openrouter",
))
```

## Enable / Disable Providers

```python
router.disable_provider(Provider.ANTHROPIC)  # disable all Anthropic configs
router.disable_provider("openai-primary")    # disable by exact name

router.enable_provider(Provider.ANTHROPIC)
```

> **Semantics:** passing an `enum` affects **all** configs of that vendor;
> passing a **string** only affects the provider with that exact name.

## Metrics

```python
print(router.get_metrics())
# {
#   'anthropic': {
#     'calls': 10, 'successes': 9, 'failures': 1,
#     'success_rate': 0.9, 'avg_ms': 312.4, 'total_ms': 3124.0,
#     'total_input_tokens': 240, 'total_output_tokens': 860,
#   }
# }

router.reset_metrics()
```

Metrics are consistent under the invariant `calls >= successes + failures` (an in-flight
request increments `calls` before it completes).

## Exceptions

| Exception | Raised when |
|---|---|
| `RouterError` | All providers exhausted |
| `CircuitOpenError` | Every provider skipped by circuit breaker (subclass of `RouterError`) |
| `ConfigurationError` | Invalid config or no enabled providers |
| `ProviderNotFoundError` | `disable_provider` / `enable_provider` on unregistered provider |

## Testing

```bash
pip install pytest
pytest test_quantarion_router.py -v
```

**Current status:** 124 tests passing, 0 failing, `mypy --strict` clean.

## Project Status

Stable. Used in production inside [QUANTARION](https://quantarion.com) platform and
[Dream Oracle](https://dreams.quantareon.com).

## Related QUANTARION Labs Projects

- **Sentinel** — AI Agent Safety Certifier *(launching 2026)*
- **Pendulum** — Multi-model Coding Orchestrator *(coming soon)*
- **Metronome** — Speed Control Protocol for local AI *(research)*

## License

MIT — see [LICENSE](LICENSE).
