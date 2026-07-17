"""Provider-owned SDK classification and retry qualification."""

import json
import re
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any

import httpx
import openai

from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    format_execution_failure_message,
    safe_exception_message,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind

MarkRateLimited = Callable[[float], None]
ProviderFailureOverride = Callable[[Exception], ExecutionFailure | None]

_RATE_LIMIT_MARKERS = frozenset({"rate_limit", "rate limit", "too many requests"})
_OVERLOAD_MARKERS = frozenset(
    {
        "resourceexhausted",
        "resource exhausted",
        "limit reached",
        "overloaded",
        "capacity",
    }
)
_INTERNAL_ERROR_MARKERS = frozenset({"internal_server_error", "internal server error"})
_CONTEXT_LENGTH_MARKERS = frozenset(
    {
        "context_length_exceeded",
        "maximum context length",
        "context window",
        "reduce the length",
        "too many tokens",
    }
)
# 400s that mean "this model can't serve this request, but another could" -
# safe to fall back to the next candidate instead of failing the request.
_INCOMPATIBLE_MODEL_MARKERS = frozenset(
    {
        "interactions api",
        "not supported",
        "does not support",
        "unsupported",
        "only supports",
    }
)
# 400/403s where the provider refused the request on content-policy / safety
# grounds (Azure/GitHub "content_filter"/"content management policy", Gemini
# "safety"/"blocked"). Another model with looser policies may accept the same
# prompt, so fall back to the next candidate - but do NOT park the model: it
# works fine for other prompts, only this one was refused.
_CONTENT_POLICY_MARKERS = frozenset(
    {
        "content_filter",
        "content filter",
        "content management policy",
        "content_policy",
        "content policy",
        "responsible_ai",
        "responsibleaipolicy",
        "safety",
        "prohibited_content",
        "prohibited content",
        "jailbreak",
        "flagged",
        "usage policies",
    }
)
# 400s that mean the model is catalog-listed but not usable for this tier/key
# (GitHub Models "unavailable model", Gemini "no longer available", Cerebras
# "does not exist"). These are persistent, so the model is parked in cooldown
# instead of being retried every turn - while still falling back this turn.
_MODEL_UNAVAILABLE_MARKERS = frozenset(
    {
        "unavailable model",
        "no longer available",
        "does not exist",
        "do not have access",
        "model not found",
        "model_not_found",
    }
)
_AUTHENTICATION_MESSAGE = "Provider authentication failed. Check API key."
_PERMISSION_MESSAGE = "Provider denied access to this model."
_PAYMENT_REQUIRED_MESSAGE = "Provider requires payment for this model."
_RATE_LIMIT_MESSAGE = "Provider rate limit reached. Please retry shortly."
_INVALID_REQUEST_MESSAGE = "Invalid request sent to provider."
_OVERLOADED_MESSAGE = "Provider is currently overloaded. Please retry."


def classify_provider_failure(
    exc: Exception,
    *,
    provider_name: str,
    read_timeout_s: float | None,
    request_id: str | None,
    mark_rate_limited: MarkRateLimited,
    provider_failure_override: ProviderFailureOverride | None = None,
) -> ExecutionFailure:
    """Return one detailed canonical failure after provider retries are exhausted."""
    if isinstance(exc, ExecutionFailure):
        failure = exc
        message = failure.message
        request_id_line = f"Request ID: {request_id}" if request_id else None
        if request_id_line and request_id_line not in message:
            message = f"{message}\n\n{request_id_line}"
        return replace(failure, message=message)

    failure = (
        provider_failure_override(exc)
        if provider_failure_override is not None
        else None
    )
    if failure is None:
        failure = _classify_provider_failure(
            exc,
            read_timeout_s=read_timeout_s,
            mark_rate_limited=mark_rate_limited,
        )
    message = format_execution_failure_message(
        failure,
        extract_upstream_error_detail(exc),
        upstream_name=provider_name,
        request_id=request_id,
    )
    return replace(failure, message=message)


def overloaded_provider_failure() -> ExecutionFailure:
    """Return the canonical provider-overload meaning and stable wording."""
    return _failure(FailureKind.OVERLOADED, 529, _OVERLOADED_MESSAGE, True)


def retryable_transient_status(exc: BaseException) -> int | None:
    """Infer a retryable HTTP-like status from one upstream exception."""
    if isinstance(exc, ExecutionFailure):
        status = exc.status_code
        return status if exc.retryable and _is_retryable_status(status) else None
    if isinstance(exc, openai.RateLimitError):
        return 429
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status if _is_retryable_status(status) else None

    status = _status_from_exception(exc)
    if _is_retryable_status(status):
        return status

    body_status = _status_from_body(getattr(exc, "body", None))
    if _is_retryable_status(body_status):
        return body_status

    text = transient_error_text(exc)
    if _has_marker(text, _RATE_LIMIT_MARKERS):
        return 429
    if _has_marker(text, _OVERLOAD_MARKERS):
        return 503
    if _has_marker(text, _INTERNAL_ERROR_MARKERS):
        return 500
    return None


def is_transient_overload_error(exc: BaseException) -> bool:
    """Return whether an upstream exception reports overload or capacity pressure."""
    if isinstance(exc, ExecutionFailure):
        return exc.kind == FailureKind.OVERLOADED
    return _has_marker(transient_error_text(exc), _OVERLOAD_MARKERS)


def transient_error_text(exc: BaseException) -> str:
    """Combine exception, body, and response text for provider classification."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(_body_to_text(body))
    response = getattr(exc, "response", None)
    if response is not None:
        with suppress(Exception):
            parts.append(response.text)
    return " ".join(part for part in parts if part).lower()


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Return whether provider policy permits stream retry or recovery."""
    if isinstance(exc, ExecutionFailure):
        return exc.retryable
    if isinstance(exc, openai.AuthenticationError | openai.BadRequestError):
        return False
    if retryable_transient_status(exc) is not None:
        return True
    return isinstance(
        exc,
        (
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
    )


def retryable_upstream_status(exc: BaseException) -> int | None:
    """Return a status eligible for provider-opening backoff."""
    status = retryable_transient_status(exc)
    return status if status is not None and _is_retryable_status(status) else None


def retryable_upstream_transport_error(exc: BaseException) -> bool:
    """Return whether a pre-response transport failure can be retried."""
    if isinstance(exc, ExecutionFailure):
        return exc.retryable and retryable_transient_status(exc) is None
    if isinstance(exc, openai.AuthenticationError | openai.BadRequestError):
        return False
    return isinstance(
        exc,
        (
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
    )


def provider_error_message(
    exc: BaseException,
    *,
    read_timeout_s: float | None = None,
) -> str:
    """Map raw provider exception types to stable customer-facing wording."""
    if isinstance(exc, ExecutionFailure):
        return exc.message
    if isinstance(exc, httpx.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Provider request timed out."
    if isinstance(exc, httpx.ConnectTimeout | httpx.ConnectError):
        return "Could not connect to provider."
    if isinstance(exc, httpx.RemoteProtocolError):
        return "Provider connection was interrupted before a response was received."
    if isinstance(exc, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."
    if isinstance(exc, openai.RateLimitError):
        return _RATE_LIMIT_MESSAGE
    if isinstance(exc, openai.AuthenticationError):
        return _AUTHENTICATION_MESSAGE
    if isinstance(exc, openai.BadRequestError):
        return _INVALID_REQUEST_MESSAGE
    return safe_exception_message(exc)


def _classify_provider_failure(
    exc: Exception,
    *,
    read_timeout_s: float | None,
    mark_rate_limited: MarkRateLimited,
) -> ExecutionFailure:
    if isinstance(exc, ExecutionFailure):
        if exc.kind == FailureKind.RATE_LIMIT:
            mark_rate_limited(rate_limit_cooldown_seconds(exc))
        return exc

    if _is_payment_required(exc):
        # 402: the provider lists this model but only serves it for pay (e.g.
        # HuggingFace's router bills the DeepSeek models). It will never succeed
        # on the free tier, so park it in a long cooldown and fall through to
        # the next candidate instead of re-hitting the paywall every turn.
        mark_rate_limited(_MAX_RATE_LIMIT_COOLDOWN_S)
        return _failure(
            FailureKind.PERMISSION,
            402,
            _PAYMENT_REQUIRED_MESSAGE,
            False,
            model_fallback_eligible=True,
        )

    if _is_request_too_large(exc):
        # 413: the request exceeds the model's context window (e.g. GitHub
        # Models caps free-tier context far below Claude Code's ~120K requests
        # and returns tokens_limit_reached). Like an 8K context 400, this is
        # structural, so park the model and fall through to a larger-window
        # candidate instead of retrying the same overflow every turn.
        mark_rate_limited(_MAX_RATE_LIMIT_COOLDOWN_S)
        return _failure(
            FailureKind.INVALID_REQUEST,
            413,
            _INVALID_REQUEST_MESSAGE,
            False,
            model_fallback_eligible=True,
        )

    if _is_model_not_found(exc):
        # 404: the provider lists the model in /models but does not actually
        # serve it (NVIDIA has ~45 such phantom entries - retired/EOL models the
        # catalog still returns). It will 404 every turn, so park it and fall
        # through instead of re-hitting the dead endpoint each request.
        mark_rate_limited(_MAX_RATE_LIMIT_COOLDOWN_S)
        return _failure(
            FailureKind.UNAVAILABLE,
            404,
            _PERMISSION_MESSAGE,
            False,
            model_fallback_eligible=True,
        )

    if isinstance(exc, openai.AuthenticationError):
        return _failure(FailureKind.AUTHENTICATION, 401, _AUTHENTICATION_MESSAGE, False)
    if isinstance(exc, openai.PermissionDeniedError):
        # 403: no access to this model for this tier (catalog-listed but gated).
        # Persistent, so park it in cooldown and try the next candidate instead
        # of hitting the same wall every turn.
        mark_rate_limited(_MAX_RATE_LIMIT_COOLDOWN_S)
        return _failure(
            FailureKind.PERMISSION,
            403,
            _PERMISSION_MESSAGE,
            False,
            model_fallback_eligible=True,
        )
    if isinstance(exc, openai.RateLimitError):
        mark_rate_limited(rate_limit_cooldown_seconds(exc))
        return _failure(FailureKind.RATE_LIMIT, 429, _RATE_LIMIT_MESSAGE, True)
    if isinstance(exc, openai.BadRequestError):
        return _bad_request_failure(exc, mark_rate_limited)
    if isinstance(exc, openai.APITimeoutError):
        return _failure(FailureKind.TIMEOUT, 500, _stable_upstream(500), True)
    if isinstance(exc, openai.APIConnectionError):
        return _failure(FailureKind.UNAVAILABLE, 500, _stable_upstream(500), True)
    if isinstance(exc, openai.InternalServerError):
        status = retryable_transient_status(exc) or getattr(exc, "status_code", None)
        if is_transient_overload_error(exc):
            return overloaded_provider_failure()
        if isinstance(status, int) and 500 <= status <= 599:
            return _failure(
                FailureKind.UPSTREAM,
                status,
                _stable_upstream(status),
                True,
            )
        return _failure(FailureKind.UPSTREAM, 500, _stable_upstream(500), True)
    if isinstance(exc, openai.APIError):
        status = retryable_transient_status(exc)
        if status == 429:
            mark_rate_limited(rate_limit_cooldown_seconds(exc))
            return _failure(FailureKind.RATE_LIMIT, 429, _RATE_LIMIT_MESSAGE, True)
        if is_transient_overload_error(exc):
            return overloaded_provider_failure()
        effective_status = status or getattr(exc, "status_code", None)
        if not isinstance(effective_status, int):
            effective_status = 500
        return _failure(
            FailureKind.UPSTREAM,
            effective_status,
            _stable_upstream(effective_status),
            is_retryable_provider_error(exc),
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return _failure(
                FailureKind.AUTHENTICATION, 401, _AUTHENTICATION_MESSAGE, False
            )
        if status == 403:
            mark_rate_limited(_MAX_RATE_LIMIT_COOLDOWN_S)
            return _failure(
                FailureKind.PERMISSION,
                403,
                _PERMISSION_MESSAGE,
                False,
                model_fallback_eligible=True,
            )
        if status == 429:
            mark_rate_limited(rate_limit_cooldown_seconds(exc))
            return _failure(FailureKind.RATE_LIMIT, 429, _RATE_LIMIT_MESSAGE, True)
        if status == 400:
            return _bad_request_failure(exc, mark_rate_limited)
        if status in (502, 503, 504):
            return overloaded_provider_failure()
        return _failure(
            FailureKind.UPSTREAM,
            status,
            _stable_upstream(status),
            _is_retryable_status(status),
        )

    kind = FailureKind.UPSTREAM
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        kind = FailureKind.TIMEOUT
    elif isinstance(exc, httpx.ConnectError | httpx.NetworkError):
        kind = FailureKind.UNAVAILABLE
    return _failure(
        kind,
        502,
        provider_error_message(exc, read_timeout_s=read_timeout_s),
        is_retryable_provider_error(exc),
    )


def _failure(
    kind: FailureKind,
    status_code: int,
    message: str,
    retryable: bool,
    *,
    model_fallback_eligible: bool = False,
) -> ExecutionFailure:
    return ExecutionFailure(
        kind=kind,
        status_code=status_code,
        message=message,
        retryable=retryable,
        model_fallback_eligible=model_fallback_eligible,
    )


# Default cooldown when the provider gives no retry hint. Long enough that a
# model whose daily quota is exhausted isn't retried every minute, short enough
# to re-check within the hour. Capped so a huge reset window doesn't sideline a
# model indefinitely.
_DEFAULT_RATE_LIMIT_COOLDOWN_S = 300.0
_MAX_RATE_LIMIT_COOLDOWN_S = 3600.0
# Gemini reports "retryDelay": "34s"; a generic Retry-After is seconds.
_RETRY_DELAY_RE = re.compile(r'retrydelay["\s:=]+(\d+(?:\.\d+)?)s')


def rate_limit_cooldown_seconds(exc: BaseException) -> float:
    """Cooldown for a rate-limited model, honoring the provider's retry hint.

    Prefers an explicit delay from the provider (Gemini ``retryDelay``,
    ``Retry-After`` header, OpenRouter ``X-RateLimit-Reset`` timestamp) so a
    per-minute limit clears fast while a daily-quota exhaustion stays parked;
    otherwise a sane default. Clamped to ``[1s, 1h]``.
    """
    delay = _provider_retry_delay(exc)
    if delay is None:
        return _DEFAULT_RATE_LIMIT_COOLDOWN_S
    return min(max(delay, 1.0), _MAX_RATE_LIMIT_COOLDOWN_S)


def _provider_retry_delay(exc: BaseException) -> float | None:
    match = _RETRY_DELAY_RE.search(transient_error_text(exc))
    if match:
        return float(match.group(1))

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    retry_after = headers.get("retry-after")
    if isinstance(retry_after, str) and retry_after.strip().isdigit():
        return float(retry_after)
    reset = headers.get("x-ratelimit-reset")
    if isinstance(reset, str) and reset.strip().isdigit():
        # OpenRouter sends a millisecond epoch timestamp.
        remaining = float(reset) / 1000.0 - time.time()
        if remaining > 0:
            return remaining
    return None


def _is_model_switchable_bad_request(exc: BaseException) -> bool:
    """Return whether a 400 would succeed on a different model (fallback-worthy).

    Covers prompts too large for this model's context window, models that
    simply can't serve chat completions (e.g. agent-only "Interactions API"
    models), models that are unavailable for this tier, and content-policy /
    safety refusals (another model with looser policies may accept the same
    prompt). A different candidate can still succeed, so the derivation chain
    should move on rather than fail the request.
    """
    text = transient_error_text(exc)
    return (
        _has_marker(text, _CONTEXT_LENGTH_MARKERS)
        or _has_marker(text, _INCOMPATIBLE_MODEL_MARKERS)
        or _has_marker(text, _MODEL_UNAVAILABLE_MARKERS)
        or _has_marker(text, _CONTENT_POLICY_MARKERS)
    )


def _bad_request_failure(
    exc: BaseException, mark_rate_limited: MarkRateLimited
) -> ExecutionFailure:
    """Classify a 400. Persistent 400s are parked in cooldown so the derivation
    stops retrying them every turn; they still fall back this turn.

    Persistent means either the model is unavailable for this tier, or its
    context window is too small to ever fit an agent-sized request (e.g.
    Cerebras' 8K GLM-4.7 rejects Claude Code's ~120K requests every turn).
    Parking context-length rejections trades a little precision - a model with
    a mid-size window that only overflowed on one unusually large request is
    benched for the cooldown - for a large win on models whose window simply
    cannot serve agentic coding: Claude Code requests stay large across a
    session, so a model that did not fit rarely fits again, and other
    candidates cover the turn.
    """
    text = transient_error_text(exc)
    persistent = _has_marker(text, _MODEL_UNAVAILABLE_MARKERS) or _has_marker(
        text, _CONTEXT_LENGTH_MARKERS
    )
    if persistent:
        mark_rate_limited(_MAX_RATE_LIMIT_COOLDOWN_S)
    return _failure(
        FailureKind.INVALID_REQUEST,
        400,
        _INVALID_REQUEST_MESSAGE,
        False,
        model_fallback_eligible=_is_model_switchable_bad_request(exc),
    )


def _stable_upstream(status_code: int) -> str:
    if status_code in (502, 503, 504):
        return "Provider is temporarily unavailable. Please retry."
    return "Provider API request failed."


def _status_from_exception(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_payment_required(exc: BaseException) -> bool:
    """Return whether the upstream rejected the model as paid-only (HTTP 402)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 402
    return _status_from_exception(exc) == 402


def _is_request_too_large(exc: BaseException) -> bool:
    """Return whether the request exceeded the model's size limit (HTTP 413)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 413
    return _status_from_exception(exc) == 413


def _is_model_not_found(exc: BaseException) -> bool:
    """Return whether the provider has no such model to serve (HTTP 404)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404
    return _status_from_exception(exc) == 404


def _status_from_body(body: Any) -> int | None:
    for item in _body_candidates(body):
        if not isinstance(item, Mapping):
            continue
        for key in ("status", "status_code", "code"):
            status = _coerce_status(item.get(key))
            if status is not None:
                return status
        type_status = _status_from_type_fields(item)
        if type_status is not None:
            return type_status
    return None


def _body_candidates(body: Any) -> tuple[Any, ...]:
    if isinstance(body, str):
        try:
            return _body_candidates(json.loads(body))
        except ValueError:
            return (body,)
    if isinstance(body, bytes):
        return _body_candidates(body.decode("utf-8", errors="replace"))
    if isinstance(body, Mapping):
        nested = body.get("error")
        return (body, nested) if isinstance(nested, Mapping) else (body,)
    return (body,)


def _coerce_status(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _status_from_type_fields(item: Mapping[str, Any]) -> int | None:
    values = [
        value.lower()
        for key in ("type", "code")
        if isinstance((value := item.get(key)), str)
    ]
    text = " ".join(values)
    if _has_marker(text, _RATE_LIMIT_MARKERS):
        return 429
    if _has_marker(text, _OVERLOAD_MARKERS):
        return 503
    if _has_marker(text, _INTERNAL_ERROR_MARKERS):
        return 500
    return None


def _body_to_text(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(body)


def _has_marker(text: str, markers: frozenset[str]) -> bool:
    return any(marker in text for marker in markers)


def _is_retryable_status(status: int | None) -> bool:
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)
