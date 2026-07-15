"""Application-owned provider execution contracts."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import ResolvedModel, RoutedMessagesRequest
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind


class FakeProvider:
    def __init__(self) -> None:
        self.preflight_calls: list[tuple[MessagesRequest, bool]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.stream_close_calls = 0

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        thinking_enabled: bool,
    ) -> None:
        self.preflight_calls.append((request, thinking_enabled))

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "thinking_enabled": thinking_enabled,
            }
        )
        try:
            yield "event: message_stop\ndata: {}\n\n"
        finally:
            self.stream_close_calls += 1


class FailingPreflightProvider(FakeProvider):
    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        thinking_enabled: bool,
    ) -> None:
        raise ValueError("invalid provider request")


class FailingStreamConstructionProvider(FakeProvider):
    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream construction failed")


class ContextLengthExceededProvider(FakeProvider):
    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "thinking_enabled": thinking_enabled,
            }
        )
        raise ExecutionFailure(
            kind=FailureKind.INVALID_REQUEST,
            status_code=400,
            message="maximum context length exceeded",
            retryable=False,
            model_fallback_eligible=True,
        )
        yield ""  # pragma: no cover - never reached, keeps this an async generator


def _routed_request() -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModel(
            original_model="gateway-model",
            provider_id="provider",
            provider_model="provider-model",
            provider_model_ref="provider/provider-model",
            thinking_enabled=True,
        ),
    )


@pytest.mark.asyncio
async def test_executor_uses_structural_provider_port_and_preflights_eagerly() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload=request.model_dump(),
        request_id="req_application",
    )

    assert provider.preflight_calls == [(request, True)]
    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert provider.stream_calls == [
        {
            "request": request,
            "input_tokens": 17,
            "request_id": "req_application",
            "thinking_enabled": True,
        }
    ]
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_closing_executor_stream_closes_provider_stream_once() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_early_close",
    )

    assert await anext(stream) == "event: message_stop\ndata: {}\n\n"
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_stream_construction_failure_remains_deferred_to_iteration() -> None:
    provider = FailingStreamConstructionProvider()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_deferred_construction",
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)


def test_executor_preflight_failure_stays_before_token_count_and_stream() -> None:
    provider = FailingPreflightProvider()
    token_counter = MagicMock(return_value=17)
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=token_counter,
    )

    with pytest.raises(ValueError, match="invalid provider request"):
        executor.stream(
            _routed_request(),
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload={},
            request_id="req_application",
        )

    token_counter.assert_not_called()
    assert provider.stream_calls == []


@pytest.mark.asyncio
async def test_context_length_exceeded_falls_back_to_next_candidate() -> None:
    primary = ContextLengthExceededProvider()
    fallback = FakeProvider()
    providers = {"provider": primary, "fallback_provider": fallback}
    routed = _routed_request()
    executor = ProviderExecutor(
        lambda provider_id: providers[provider_id],
        token_counter=lambda _messages, _system, _tools: 17,
    )
    model_cache = MagicMock()
    model_cache.cached_prefixed_model_infos.return_value = (
        ProviderModelInfo("fallback_provider/bigger-model"),
    )
    model_router = MagicMock()
    model_router.resolve.return_value = ResolvedModel(
        original_model="gateway-model",
        provider_id="fallback_provider",
        provider_model="bigger-model",
        provider_model_ref="fallback_provider/bigger-model",
        thinking_enabled=True,
    )

    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_context_fallback",
        model_router=model_router,
        model_cache=model_cache,
    )

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert len(primary.stream_calls) == 1
    assert len(fallback.stream_calls) == 1


@pytest.mark.asyncio
async def test_successful_stream_records_usage_stats() -> None:
    provider = FakeProvider()
    usage_stats = MagicMock()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_usage_success",
        usage_stats=usage_stats,
    )

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    usage_stats.record_success.assert_called_once_with(
        "provider", "provider-model", input_tokens=17
    )
    usage_stats.record_error.assert_not_called()


@pytest.mark.asyncio
async def test_stream_construction_failure_records_usage_error() -> None:
    provider = FailingStreamConstructionProvider()
    usage_stats = MagicMock()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_usage_error",
        usage_stats=usage_stats,
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)

    usage_stats.record_error.assert_called_once_with("provider", "provider-model")
    usage_stats.record_success.assert_not_called()


def _derivation_routed_request() -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="menarve/derivation",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModel(
            original_model="claude-opus",
            provider_id="menarve",
            provider_model="derivation",
            provider_model_ref="menarve/derivation",
            thinking_enabled=True,
            derivation=True,
        ),
    )


def _resolve_ref(ref: str) -> ResolvedModel:
    provider_id, model_id = ref.split("/", 1)
    return ResolvedModel(
        original_model="claude-opus",
        provider_id=provider_id,
        provider_model=model_id,
        provider_model_ref=ref,
        thinking_enabled=True,
    )


@pytest.mark.asyncio
async def test_derivation_mode_tries_strongest_candidate_first() -> None:
    strong = FakeProvider()
    weak = FakeProvider()
    providers = {"gemini": strong, "open_router": weak}
    executor = ProviderExecutor(
        lambda provider_id: providers[provider_id],
        token_counter=lambda _messages, _system, _tools: 5,
    )
    model_cache = MagicMock()
    model_cache.cached_prefixed_model_infos.return_value = (
        ProviderModelInfo("open_router/small-model:free"),
        ProviderModelInfo("gemini/gemini-3.1-pro"),
    )
    model_router = MagicMock()
    model_router.resolve.side_effect = _resolve_ref

    stream = executor.stream(
        _derivation_routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_derivation",
        model_router=model_router,
        model_cache=model_cache,
    )

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    # gemini-3.1-pro (large) outranks the small OpenRouter model, so it is tried
    # first, succeeds, and the weaker candidate is never reached.
    assert len(strong.stream_calls) == 1
    assert len(weak.stream_calls) == 0


@pytest.mark.asyncio
async def test_derivation_mode_without_candidates_raises_unavailable() -> None:
    executor = ProviderExecutor(
        lambda _provider_id: FakeProvider(),
        token_counter=lambda _messages, _system, _tools: 5,
    )
    model_cache = MagicMock()
    model_cache.cached_prefixed_model_infos.return_value = ()
    model_router = MagicMock()

    stream = executor.stream(
        _derivation_routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_derivation_empty",
        model_router=model_router,
        model_cache=model_cache,
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 503
