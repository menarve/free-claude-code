"""Provider-owned upstream rate limiting and retry policy."""

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from loguru import logger

from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.failure_policy import (
    ProviderFailureOverride,
    retryable_upstream_status,
    retryable_upstream_transport_error,
)

T = TypeVar("T")

UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS = 5
DEFAULT_UPSTREAM_MAX_RETRIES = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS - 1


class ProviderRateLimiter:
    """
    Rate limiter owned by one provider instance.

    Blocks that provider's requests when a rate-limit error is encountered
    (reactive) and throttles its requests with a strict rolling window
    (proactive).

    Optionally enforces a max_concurrency cap: at most N provider streams
    may be open simultaneously, independent of the sliding window.

    Proactive limits - throttles requests to stay within API limits.
    Reactive limits - pauses all requests when a 429 or 5xx retry backoff is active.
    Concurrency limit - caps simultaneously open streams.
    """

    def __init__(
        self,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
    ):
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)
        self._max_concurrency = max_concurrency
        # Proactive send throttle is keyed per model id, like the reactive block:
        # each model has its own upstream quota, so one model's pacing must not
        # slow requests to the provider's other models (critical for derivation,
        # which sweeps many models of the same provider). "" is the shared key.
        self._proactive_limiters: dict[str, StrictSlidingWindowLimiter] = {}
        # Reactive 429/5xx blocks are keyed per model id: a rate limit hit on one
        # model must not block the provider's other models, which have their own
        # upstream quotas. "" is the shared key for callers without a model.
        self._blocked_until_by_model: dict[str, float] = {}
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        logger.info(
            "ProviderRateLimiter initialized "
            f"({rate_limit} req / {rate_window}s, max_concurrency={max_concurrency})"
        )

    async def wait_if_blocked(self, model: str | None = None) -> bool:
        """
        Wait if the given model is rate limited or throttle to meet quota.

        Returns:
            True if was reactively blocked and waited, False otherwise.
        """
        # A reactive deadline can be installed or extended while this task waits
        # for proactive capacity. Commit the proactive timestamp only if that
        # deadline is still clear, so retries neither burst nor consume unused quota.
        waited_reactively = False
        proactive = self._proactive_for(model)
        while True:
            waited_reactively = (
                await self._wait_for_reactive_block(model) or waited_reactively
            )
            if await proactive.acquire_if(lambda: not self.is_blocked(model)):
                return waited_reactively

    def _proactive_for(self, model: str | None) -> StrictSlidingWindowLimiter:
        key = model or ""
        limiter = self._proactive_limiters.get(key)
        if limiter is None:
            limiter = StrictSlidingWindowLimiter(self._rate_limit, self._rate_window)
            self._proactive_limiters[key] = limiter
        return limiter

    async def _wait_for_reactive_block(self, model: str | None = None) -> bool:
        waited = False
        while (wait_time := self.remaining_wait(model)) > 0:
            logger.warning(
                "Provider rate limit active (reactive) for model={}, waiting {:.1f}s...",
                model or "-",
                wait_time,
            )
            await asyncio.sleep(wait_time)
            waited = True
        return waited

    def extend_reactive_block(self, seconds: float, model: str | None = None) -> None:
        """
        Extend the reactive block for ``model`` by at least ``seconds`` from now.

        Args:
            seconds: Positive minimum duration for the resulting block.
            model: Model id whose upstream quota was hit; blocks only that model.
        """
        if seconds <= 0:
            raise ValueError("reactive block duration must be > 0")
        now = time.monotonic()
        key = model or ""
        blocked_until = max(self._blocked_until_by_model.get(key, 0.0), now + seconds)
        self._blocked_until_by_model[key] = blocked_until
        logger.warning(
            "Provider rate limit set for {:.1f}s (reactive) model={}",
            max(0.0, blocked_until - now),
            model or "-",
        )

    def is_blocked(self, model: str | None = None) -> bool:
        """Check if the given model is currently reactively blocked."""
        return self.remaining_wait(model) > 0

    def remaining_wait(self, model: str | None = None) -> float:
        """Get remaining reactive wait time in seconds for the given model."""
        blocked_until = self._blocked_until_by_model.get(model or "", 0.0)
        return max(0.0, blocked_until - time.monotonic())

    @asynccontextmanager
    async def concurrency_slot(self) -> AsyncIterator[None]:
        """Async context manager that holds one concurrency slot for a stream.

        Blocks until a slot is available (controlled by max_concurrency).
        """
        await self._concurrency_sem.acquire()
        try:
            yield
        finally:
            self._concurrency_sem.release()

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        provider_failure_override: ProviderFailureOverride | None = None,
        max_retries: int = DEFAULT_UPSTREAM_MAX_RETRIES,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with rate limiting and retry on transient limits.

        Waits for the proactive limiter before each attempt. On ``429`` (rate limit)
        or upstream ``5xx`` server errors, applies exponential backoff with jitter
        and sets the reactive block before retrying. Pre-response transport errors
        use the same attempt budget and backoff schedule without setting the
        reactive provider block.

        Args:
            fn: Async callable to execute.
            provider_failure_override: Optional provider-specific semantic
                classifier applied before shared retry qualification.
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay: Base delay in seconds for exponential backoff.
            max_delay: Maximum delay cap in seconds.
            jitter: Maximum random jitter in seconds added to each delay.

        Returns:
            The result of the callable.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        total_attempts = 1 + max_retries
        # The upstream body carries the model id, so per-model reactive blocking
        # needs no extra plumbing through the provider call sites.
        model = kwargs.get("model")
        model_id = model if isinstance(model, str) else None

        for attempt in range(total_attempts):
            await self.wait_if_blocked(model_id)

            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                effective_error = (
                    provider_failure_override(e)
                    if provider_failure_override is not None
                    else None
                )
                if effective_error is None:
                    effective_error = e
                status = retryable_upstream_status(effective_error)
                transport_error = status is None and retryable_upstream_transport_error(
                    effective_error
                )
                if status is None and not transport_error:
                    raise
                # A 429 means this model has no quota right now; retrying inline
                # won't refill it in a few seconds. Fail fast so the caller can
                # cool this model down and move to the next one. Transient 5xx and
                # transport errors still retry, since they can genuinely recover.
                if status == 429:
                    raise

                if status is None:
                    label = f"Provider transport error ({type(e).__name__})"
                else:
                    label = (
                        "Rate limited (429)"
                        if status == 429
                        else f"Upstream server error ({status})"
                    )
                last_exc = e
                if attempt >= max_retries:
                    logger.warning(
                        "{} retry exhausted after {} retries (attempts={})",
                        label,
                        max_retries,
                        total_attempts,
                    )
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, jitter)
                attempt_no = attempt + 1
                logger.warning(
                    "{}, attempt {}/{}. Retrying in {:.1f}s...",
                    label,
                    attempt_no,
                    total_attempts,
                    delay,
                )
                trace_event(
                    stage="provider",
                    event="provider.retry.scheduled",
                    source="provider",
                    status_code=status,
                    exc_type=type(e).__name__,
                    attempt=attempt_no,
                    max_attempts=total_attempts,
                    delay_s=round(delay, 3),
                )
                if status is not None:
                    self.extend_reactive_block(delay, model_id)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
