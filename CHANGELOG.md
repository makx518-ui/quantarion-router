# Changelog

All notable changes to `quantarion-router` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-23

First public release under the **QUANTARION Labs** brand. Extensive hardening
pass — 9 issues discovered during audit are fixed, test count grows from 103
to 124, `mypy --strict` remains clean.

### Fixed

- **Round-robin index double-increment under executor fallback** (critical).
  `acomplete()` without `aiohttp` called `_get_ordered_safe()` in both the outer
  async path and the delegated `self.complete()`, causing the round-robin index
  to advance by 2 per call. Providers were silently skipped in rotation.
- **`disable_provider(enum)` only affected the first match** (critical).
  When multiple configs shared the same `Provider` enum under different names
  (e.g. `"cheap-claude"` and `"smart-claude"` both `Provider.ANTHROPIC`),
  disabling by enum only touched the first. New semantics: enum disables all
  matching configs, string disables only the exact-name match.
- **Race condition in `_get_aio_lock`** (critical). Two concurrent tasks on a
  fresh router could each observe `self._aio_lock is None` and create separate
  `asyncio.Lock` instances, defeating the guard and leaking aiohttp sessions.
  Now protected by double-checked locking via the router's threading lock.
- **`asyncio.sleep` was not injectable** (serious). Async backoff called
  `asyncio.sleep` directly, breaking symmetry with the injectable sync
  `_sleep_fn` and forcing real-time waits in async tests. New: `_async_sleep_fn`
  field, test-injectable.
- **HTTP error bodies were discarded** (serious). On 401/429/500 responses,
  users saw `HTTP Error 401: Unauthorized` with no provider context. Now the
  router parses OpenAI/Anthropic JSON error bodies and surfaces the actual
  message (e.g. `Anthropic HTTP 401: invalid x-api-key`). Falls back gracefully
  for non-JSON bodies.
- **`CircuitOpenError` was swallowed at the route-loop boundary** (serious).
  When every provider was skipped by its circuit breaker, callers received a
  generic `RouterError` with no way to distinguish "wait and retry" from
  "real failure". `CircuitOpenError` (subclass of `RouterError`) is now raised
  in that case — backwards-compatible for existing `except RouterError` catches.
- **`__repr__` iterated `_providers` without the lock** (minor). Could raise
  `RuntimeError: list changed size during iteration` under concurrent
  `add_provider()`. Snapshot now taken under the lock.
- **Circuit breaker `failures` not reset on open -> half-open transition**
  (minor). The counter was left at threshold after transition, producing a
  dirty state on the next failure. Now reset to 0 on every state transition.

### Documented

- **Metrics consistency invariant**: `calls >= successes + failures` because
  `calls` is incremented at attempt start and `successes`/`failures` at
  completion. Snapshots during in-flight requests correctly reflect this.

### Added

- 21 new regression tests covering every fix above.
- This CHANGELOG.
- CI workflow on GitHub Actions running `pytest` and `mypy --strict` on every push.
- PyPI packaging via `pyproject.toml`.

### Changed

- Main module renamed from `ai_router` to `quantarion_router` to match the
  package name. Import path: `from quantarion_router import AIRouter, ...`.
- Test file renamed from `test_ai_router.py` to `test_quantarion_router.py`.

### Backwards Compatibility

- All public APIs (`AIRouter`, `Provider`, `ProviderConfig`, `RoutingStrategy`,
  `RouteResult`, `CircuitBreakerConfig`, exceptions) are unchanged.
- `CircuitOpenError` remains a subclass of `RouterError`, so catching
  `RouterError` continues to handle both.
- The only breaking change is the module rename. Users on the old internal
  name should update `import ai_router` -> `import quantarion_router`.

## [1.0.0] - 2026-03-30

Initial release.

- Multi-provider routing (Anthropic, OpenAI, custom handlers)
- Primary / Fallback / Round-Robin strategies
- Exponential backoff with jitter
- Circuit breaker with half-open state
- Native async via `aiohttp` with executor fallback
- Per-provider metrics (calls, success rate, latency, tokens)
- Thread-safe
- 103 tests, `mypy --strict` clean, zero required dependencies
