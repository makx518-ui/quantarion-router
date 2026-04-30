# Case Study: Applying a 4-Voice Audit Methodology to My Own Library

> A report on the result of applying a methodology developed within [ConsciousAI Protocol](https://www.producthunt.com/products/consciousai-protocol) to my own production library `quantarion-router`.

**Author**: Vlad M., founder of QUANTARION Labs (Tbilisi).

---

## TL;DR

I published `quantarion-router` v1.0 on GitHub: 1211 lines of production code, 103 unit tests, `mypy --strict` clean. Released on March 30, 2026.

About three weeks later I applied my own structural audit methodology to this code. The audit took roughly one hour. It produced v1.1 (April 23, 2026): **8 latent bugs** found and fixed, plus **1 documented invariant**. Of the 8 bugs — 3 critical, 3 serious, 2 minor. 21 regression tests added. v1.1 ships with **124 tests, `mypy --strict` clean**.

This document walks through each finding with line-level references to the production code and regression tests, so any reader with a Python environment can verify the claims.

---

## A note on methodology

This report shows the **result** of applying the methodology, not its internal mechanics. The specific machinery of the four conditional voices, and a deeper aspect — training the model to operate in a state of pause — are deliberately left outside the public description. This document records only the observable outcome: which issues were found, how they were fixed, which regression tests were added.

An architectural implementation of the methodology is in development as a separate product. The version used here is an emulation — a lighter form of the same approach.

---

## Setup

- **Project**: [`quantarion-router`](https://github.com/makx518-ui/quantarion-router) — a multi-provider LLM routing engine.
- **Size**: 1211 lines of production code, 1808 lines of tests.
- **v1.0 status before audit**: 103 tests passing, `mypy --strict` clean, MIT-licensed, public on GitHub.
- **Time between releases**: ~3 weeks (v1.0 — March 30, 2026; v1.1 — April 23, 2026).
- **Audit duration**: roughly one hour for the full cycle — analysis, fixes, regression tests.

### Findings overview

| # | Finding | Severity |
|---|---|---|
| 1 | Round-robin index double-increment under executor fallback | critical |
| 2 | `disable_provider(enum)` only affected the first match | critical |
| 3 | Race condition in `_get_aio_lock` | critical |
| 4 | `asyncio.sleep` was not injectable | serious |
| 5 | HTTP error bodies were discarded | serious |
| 6 | `CircuitOpenError` swallowed at the route-loop boundary | serious |
| 7 | `__repr__` iterated `_providers` without the lock | minor |
| 8 | Circuit breaker `failures` not reset on transition | minor |
| 9 | Metrics consistency invariant *(documented)* | — |

---

## Bug #1 — Round-robin index double-increment under executor fallback

**Severity**: critical
**Affects**: `RoutingStrategy.ROUND_ROBIN` + async path without `aiohttp` installed.

### Symptom

When `aiohttp` is not installed, `acomplete()` falls through to a sync executor and delegates to `self.complete()`. In the pre-fix version both functions independently called `_get_ordered_safe()`, which besides returning a list also increments `self._round_robin_index`. The result: the index advanced **by 2 per user-facing call**, and providers were skipped in rotation.

### Fix

In the executor fallback branch, `acomplete()` now delegates **entirely** to `self.complete()` without any pre-validation. Documented inline at [`quantarion_router.py:1063-1072`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L1063-L1072):

```python
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
```

### Regression test

[`test_quantarion_router.py:1555`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1555) — `test_round_robin_with_executor_fallback_no_double_increment`. Asserts that after 3 async calls through the executor path, the index is exactly 3 (not 6) and both providers were used.

### Why the v1.0 test suite did not cover this

The v1.0 round-robin tests covered the path with `aiohttp` installed, where the bug does not manifest. The executor fallback path was tested for correctness of result, but not for the cleanliness of side effects between two adjacent invocations of the same `_get_ordered_safe()`. The function name (`_get_ordered_safe`) gives no hint that it mutates state — that semantic gap was the source of the defect.

---

## Bug #2 — `disable_provider(enum)` affected only the first match

**Severity**: critical
**Affects**: configurations with multiple providers from the same vendor.

### Symptom

If a user registers two configurations with the same `Provider` enum but different names (e.g. `"cheap-claude"` and `"smart-claude"`, both `Provider.ANTHROPIC`), calling `router.disable_provider(Provider.ANTHROPIC)` disabled **only the first matching** configuration. The second remained active, contradicting the natural expectation of "disable this vendor entirely".

### Fix

The method now has explicit dual semantics ([`quantarion_router.py:1107-1126`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L1107-L1126)):

- `disable_provider(Provider.ANTHROPIC)` — disables **all** configurations with this enum.
- `disable_provider("cheap-claude")` — disables **only** the configuration with this exact name.

```python
def disable_provider(self, provider: Union[Provider, str]) -> None:
    with self._lock:
        configs = self._find_providers_all(provider)
        if not configs:
            raise ProviderNotFoundError(...)
        for config in configs:
            config.enabled = False
            logger.info("Disabled provider '%s'", config.name)
```

Symmetric behavior is implemented in `enable_provider()`.

### Regression tests

- [`test_quantarion_router.py:513`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L513) — `test_disable_by_enum_affects_all_same_vendor`. Asserts that with two `Provider.ANTHROPIC` configurations, disabling by enum affects both.
- [`test_quantarion_router.py:539`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L539) — `test_disable_by_name_string_affects_only_exact_match`. Asserts that disabling by name string affects only the exact match.

### Why the v1.0 test suite did not cover this

v1.0 tests exercised the standard scenario: one provider per vendor. The case of "multiple configurations under one enum" was not in the original scenario set — it is a less common but valid usage pattern (e.g. `cheap-claude` for drafts and `smart-claude` for critical paths). The method's contract was **ambiguous in the original form**: what should happen on a collision was not specified in the docs.

---

## Bug #3 — Race condition in lazy initialization of `asyncio.Lock`

**Severity**: critical
**Affects**: `acomplete()` via `aiohttp` on first access from multiple threads simultaneously.

### Symptom

On the first async call against a freshly constructed router, two threads — each running its own event loop — could simultaneously observe `self._aio_lock is None` and create **two distinct** `asyncio.Lock` instances. After that, the guard the lock was supposed to provide stopped working. Side effect: aiohttp sessions created in parallel were not closed, producing a resource leak.

Under light load the bug is nearly invisible. Under load with concurrent threads — steady degradation.

### Fix

Applied **double-checked locking** through the router's existing threading lock. Implementation at [`quantarion_router.py:795-812`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L795-L812):

```python
def _get_aio_lock(self) -> "asyncio.Lock":
    # Fast path: already initialised, no locking needed (atomic read).
    if self._aio_lock is not None:
        return self._aio_lock
    # Slow path: serialise creation across threads/tasks.
    with self._lock:
        if self._aio_lock is None:
            self._aio_lock = asyncio.Lock()
        return self._aio_lock
```

Fast check without locking on the hot path; slow path with threading lock acquired only at creation time.

### Regression test

[`test_quantarion_router.py:1733`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1733) — `test_get_aio_lock_returns_same_instance_under_contention`. Spawns 20 threads synchronized via `threading.Barrier` to maximize the chance of a race. Assertion: the number of **unique** lock objects after 20 parallel calls equals 1.

### Why the v1.0 test suite did not cover this

Race conditions in lazy initialization are a class of bugs that **in principle** is rarely covered by ordinary unit tests. Standard TDD does not assume stress tests on concurrency. To catch a race with high probability, you need explicit barrier synchronization — that is a stress-test format, not a unit-test format. The v1.0 code looked like a textbook lazy-init pattern; the non-atomicity of the "check → create" pair is not obvious without explicitly asking about it.

---

## Bug #4 — `asyncio.sleep` was not injectable

**Severity**: serious
**Affects**: testability of async backoff.

### Symptom

In the sync path, the router used `_sleep_fn`, an injectable field for tests, to avoid real delays. In the async path, backoff called `asyncio.sleep` directly — async tests had to either wait for real seconds or use module-level monkey-patching, which is fragile.

### Fix

Added a symmetric `_async_sleep_fn` field in the router's initializer ([`quantarion_router.py:395`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L395)) and used it in all async backoff calls ([`quantarion_router.py:1023`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L1023)):

```python
self._async_sleep_fn: Callable[[float], Any] = asyncio.sleep  # injectable for tests
...
await self._async_sleep_fn(delay)
```

### Regression test

[`test_quantarion_router.py:898`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L898) — `test_async_backoff_uses_injectable_sleep`. Replaces `_async_sleep_fn` with an instrumented function and asserts it is called with the expected delays.

### Why the v1.0 test suite did not cover this

This is not a logic bug, but an **API symmetry defect**. The sync path was designed for testability; the async path was not. Standard test-coverage audits do not flag this kind of defect: the code is covered, the tests pass. The problem becomes visible only when the two parallel API surfaces are compared side by side and the question "why are they not symmetric?" is asked.

---

## Bug #5 — HTTP error bodies were discarded

**Severity**: serious
**Affects**: quality of error reporting to the user.

### Symptom

On 401/429/500 responses from providers, the user saw a generic `HTTP Error 401: Unauthorized` with no specifics. The JSON error body from Anthropic/OpenAI (with fields like `error.message`, `error.type`, etc.) was discarded entirely. This turned diagnostics into guesswork: a 401 can mean an invalid key, a malformed header, a revoked key, etc. — and the user had no way to tell which.

### Fix

Helper functions [`_format_api_error`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L191) and [`_extract_http_error`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L221) now parse JSON error bodies for OpenAI and Anthropic schemas, extract the actionable message, and produce a final string like `Anthropic HTTP 401: invalid x-api-key`. On invalid JSON or empty body — graceful fallback to the original reason phrase.

### Regression tests

Six tests at [`test_quantarion_router.py:667-712`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L667):

- `test_format_api_error_parses_openai_json`
- `test_format_api_error_parses_anthropic_json`
- `test_format_api_error_handles_plain_text`
- `test_format_api_error_handles_empty_body`
- `test_format_api_error_handles_none_body`
- `test_format_api_error_truncates_huge_body`

These cover all branches of the parser, including edge cases with huge and invalid bodies.

### Why the v1.0 test suite did not cover this

v1.0 tests verified that an HTTP error **raised an exception** — enough for a green CI. They did not verify that the **message** in the exception carried actionable information. This is the difference between "the function works" and "the function is useful in production". The gap becomes obvious only on a real diagnostic encounter in a live service.

---

## Bug #6 — `CircuitOpenError` was swallowed at the route-loop boundary

**Severity**: serious
**Affects**: ability to distinguish "temporarily unavailable" from "actually down".

### Symptom

When **every** provider was skipped by the circuit breaker (i.e. each was in open state after a recent run of failures), the caller received a generic `RouterError` with no way to distinguish two qualitatively different situations:

- "All providers are inside their recovery window — wait and retry" (transient).
- "All providers actually failed — page someone" (permanent).

Without that distinction, client-side retry logic could not make the right call.

### Fix

`CircuitOpenError` was introduced as a subclass of `RouterError` ([`quantarion_router.py:93`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L93)). When all providers are exhausted via the circuit breaker, the route loop now explicitly raises `CircuitOpenError` ([`quantarion_router.py:756`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L756) and [`quantarion_router.py:1029`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L1029) for async).

Because it is a subclass of `RouterError`, existing code of the form `except RouterError` keeps working unchanged. New code can catch `CircuitOpenError` separately for "wait and retry" semantics.

### Regression tests

- [`test_quantarion_router.py:1025`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1025) — `test_all_providers_circuit_open_raises_circuit_open_error`. Sync path.
- [`test_quantarion_router.py:1087`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1087) — `test_all_providers_circuit_open_raises_circuit_open_error_async`. Async path.
- [`test_quantarion_router.py:1059`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1059) — `test_partial_circuit_open_still_raises_router_error`. Asserts that with partial open (some providers still available), the previous behavior is preserved.

### Why the v1.0 test suite did not cover this

v1.0 tests verified that the breaker **opens** after a run of failures and that this raises an error. They did not distinguish between error types. This is the typical situation where a test asserts "something was raised" without asserting "exactly what was raised". Distinguishing error types becomes critical only when you start treating the exception type as part of the API contract.

---

## Bug #7 — `__repr__` iterated `_providers` without the lock

**Severity**: minor
**Affects**: debug safety in a multi-threaded environment.

### Symptom

`__repr__` iterated `self._providers` directly. If another thread called `add_provider()` at the same moment, Python could raise `RuntimeError: list changed size during iteration` in the middle of a debug log line.

### Fix

[`quantarion_router.py:1202-1211`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L1202-L1211): a snapshot of provider names is taken under the lock; formatting then operates on the snapshot:

```python
def __repr__(self) -> str:
    with self._lock:
        provider_names = [p.name for p in self._providers]
    return (
        f"AIRouter(strategy='{self._strategy.value}', "
        f"providers={provider_names}, "
        f"max_retries={self._max_retries})"
    )
```

### Regression test

[`test_quantarion_router.py:1336`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1336) — `test_repr_safe_under_concurrent_add_provider`. A multi-threaded stress test: one thread continuously calls `repr(router)`, others add providers. Assertion: no `repr` call raises `RuntimeError`.

### Why the v1.0 test suite did not cover this

`__repr__` is rarely treated as a method that requires thread-safety — it is a debug helper. But in a production system, `__repr__` may be invoked by a logger at any moment, on any thread. This class of bug falls into the blind spot of "no one writes tests for debug methods".

---

## Bug #8 — Circuit breaker `failures` was not reset on the transition to half-open

**Severity**: minor
**Affects**: cleanliness of circuit-breaker state across transitions.

### Symptom

On the breaker's transition from open to half-open, the `failures` counter remained equal to the threshold (i.e. at the maximum). If the very first probe request in half-open failed, the counter was already "dirty" and the decision logic operated on accumulated state from the previous failure run, rather than on the state of the current recovery attempt.

### Fix

[`quantarion_router.py:516`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L516) and [`quantarion_router.py:539`](https://github.com/makx518-ui/quantarion-router/blob/main/quantarion_router.py#L539): `failures = 0` is now explicitly reset on every state transition (open → half-open and half-open → closed).

### Regression test

[`test_quantarion_router.py:1111`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1111) — `test_failures_reset_on_transition_to_half_open`. After the circuit opens (accumulating `threshold` failures) and transitions to half-open by timeout, the test asserts `failures == 0`.

### Why the v1.0 test suite did not cover this

v1.0 tests verified that the breaker correctly opens, correctly transitions to half-open, correctly closes on success. These are **behavioral** tests. The cleanliness of **internal state** between transitions was not tested — the bug surfaced only in a specific sequence (open → half-open → fail → ...) and only indirectly affected behavior through the decision logic. This is the blind spot typical of state machines: input/output is tested, state invariants are not.

---

## #9 — Documented invariant: `calls >= successes + failures`

**Severity**: — *(documentation, not a bug)*

### What was documented

When metrics are sampled while an in-flight request is mid-flight, the snapshot can show `calls > successes + failures`. This is **not** a bug — it is the consequence of the increment order: `calls` is incremented at attempt start, `successes`/`failures` at completion. A snapshot taken between those two moments legitimately shows the difference.

### Where it is documented

The "Documented" section in [`CHANGELOG.md`](https://github.com/makx518-ui/quantarion-router/blob/main/CHANGELOG.md) for v1.1.0. Also explicitly noted in the Metrics section of the README.

### Regression test

[`test_quantarion_router.py:1291`](https://github.com/makx518-ui/quantarion-router/blob/main/test_quantarion_router.py#L1291) — `test_metrics_invariant_under_concurrent_load`. Under concurrent load, asserts that for every provider and at every moment, `calls >= successes + failures` holds.

### Why this surfaced in the audit

Not as a bug — as an **undocumented invariant**. Under load you could observe values that look surprising at first glance. Without the invariant being explicit, a library user could file a false issue, mistaking correct state for broken state. Documenting the invariant is part of API maturity.

---

## Honest limitations

Several direct statements about the boundaries of what this report shows:

- **N=1**. The result is shown on one library (1211 LOC). This is not a statistical sample; it is a case study.
- **No control group**. I did not run a parallel audit of the same code without the methodology, to isolate its contribution. Attribution of each finding to specific elements of the methodology is not provable in this format.
- **Limited reproducibility**. An external reproducer will not get an identical list of 8 findings — both because the audit involves judgment, and because the key part of the methodology (see "A note on methodology") does not transfer through text.
- **No formal stop-criterion**. The audit was halted on a subjective criterion: "no new substantive findings emerge in further passes". A formal stop-criterion is an open methodological question for the architectural version.
- **The report itself was audited the same way**. This document was reviewed through several passes of the same methodology. Earlier passes surfaced and corrected: factual distortions (incorrect counts of findings), timeline distortions, fabricated narratives. Later passes shifted from substantive to stylistic findings.

These limitations do not undo the result (8 bugs were genuinely found, fixed, and covered by regression tests — verifiable against the code). But they set the frame in which the result should be interpreted.

---

## The semantic pattern of the methodology

If only one thought from this whole report should remain — it is this.

**Unit tests catch what the author thought of. A structural audit looks for what the author did not think of.**

The 103 tests in v1.0 were not bad tests. They correctly covered the scenarios the author had explicitly chosen to verify: correctness of result, opening and closing of the circuit breaker, round-robin under normal conditions, exception raising on errors.

All 8 bugs found by the audit lay **outside** what the author was thinking about while writing the tests:

- Round-robin double-increment — because the author did not consider that a getter named `_get_ordered_safe` mutates state.
- `disable_provider` — because the author did not consider the case of two configurations sharing one enum.
- Race condition — because the author did not consider concurrent initialization of a lazy lock from different threads.
- `asyncio.sleep` — because the author did not consider the symmetry of sync and async APIs in terms of testability.
- HTTP error bodies — because the author was checking "is an exception raised?", not "does it carry actionable information?".
- `CircuitOpenError` swallowed — because the author did not consider that the caller needs to distinguish transient from permanent failures.
- `__repr__` without locking — because the author did not consider that a debug method is also called in production under load.
- Failures reset — because the author did not consider invariants of **internal** state between transitions of a state machine.

This is not a critique of the author. It is a structural property of any development process: **thinking of everything at once is impossible**. TDD says "write the test before the code" — that helps, but the test reflects the same understanding of the task as the code. If there is a blind spot in the author's mental model, it will be present in both the code and the tests at once. Tests do not pull the author beyond their own model of the task.

A structural audit is a layer external to the author. Its job is to **ask questions the author did not ask**: "what mutates behind this getter?", "what happens on a collision in this enum?", "what is visible in a state snapshot mid-transaction?", "are these two parallel API surfaces symmetric?". These questions are the content of the methodology.

This is the actual class of work. Not "find bugs the author missed by inattention". Rather, **expand the frame of verification beyond the frame of the task** in which the code was written. This is a qualitatively different operation, and within it lies the substance of the four-voice methodology and of the deeper layers left outside this document.

---

*Vlad M., QUANTARION Labs · April 2026*

*Related: [ConsciousAI Protocol](https://www.producthunt.com/products/consciousai-protocol) (methodology) · architectural implementation in development.*
