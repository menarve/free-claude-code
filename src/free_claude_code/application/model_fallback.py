"""Dynamic model fallback chain for Free Claude Code.

When a requested model (default, opus/sonnet/haiku role, or a direct
``provider/model``) fails by capacity (quota / rate-limit / overload), retry
other ``provider/model`` refs discovered from the API keys that are
configured, ordered by heuristic potency (most -> least potent). This is a
best-effort, local-only behavior that only switches before the first streamed
chunk is delivered to the client.

NOTE: this module patches the editable Free Claude Code install at
``/private/tmp/free-claude-code``. Changes are lost if fcc is reinstalled or
updated; re-apply after any upgrade.
"""

from collections.abc import Iterable

from loguru import logger

from free_claude_code.application.model_metadata import ProviderModelInfo

# Heuristic potency tiers by model-family keyword (descending). Unknown models
# score mid (50). Adjust as new families appear. ProviderModelInfo exposes no
# capability/intelligence metadata, so ordering is necessarily name-based.
_POTENCY_TIERS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        95,
        (
            "opus",
            "claude-opus",
            "gpt-5",
            "gpt5",
            "o3",
            "o1",
            "deepseek-r1",
            "llama-4",
            "qwen-max",
            "qwen3-max",
            "gemini-2.5-pro",
            "gemini-3-pro",
            "grok-3",
        ),
    ),
    (
        80,
        (
            "sonnet",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4",
            "deepseek-v3",
            "deepseek-chat",
            "gemini-2.5-flash",
            "gemini-3-flash",
            "gemini-3.1-flash",
            "qwen-plus",
            "qwen3-235b",
            "llama-3.3",
            "mistral-large",
            "command-r-plus",
            "grok-4",
            "grok-2",
        ),
    ),
    (
        60,
        (
            "haiku",
            "gpt-oss",
            "gpt-4o-mini",
            "mini",
            "deepseek-coder",
            "qwen-coder",
            "qwen2.5",
            "qwen3-32b",
            "llama-3.2",
            "llama-3.1",
            "mistral",
            "codestral",
            "gemma",
            "phi-4",
            "command-r",
        ),
    ),
    (
        35,
        (
            "nano",
            "small",
            "tiny",
            "0.5b",
            "1b",
            "1.5b",
            "3b",
            "7b",
            "8b",
            "instruct",
        ),
    ),
)

# Model kinds that are NOT chat-completion models; exclude from the chain.
_NON_CHAT_MARKERS = (
    "embedding",
    "audio",
    "tts",
    "whisper",
    "transcribe",
    "rerank",
    "reranker",
    "moderation",
    "moderate",
    "image",
    "/img",
    "dall",
    "vision",
    "guard",
    "classify",
    "detect",
    "speech",
    "voxtral",
)


def is_chat_model(model_ref: str) -> bool:
    """Return whether a ``provider/model`` ref looks like a chat model."""

    ref = model_ref.lower()
    return not any(marker in ref for marker in _NON_CHAT_MARKERS)


def rank_potency(model_ref: str) -> int:
    """Heuristic potency score for ordering fallback candidates (higher = stronger)."""

    ref = model_ref.lower()
    for score, keywords in _POTENCY_TIERS:
        if any(keyword in ref for keyword in keywords):
            return score
    return 50


def build_fallback_chain(
    primary_ref: str,
    model_infos: Iterable[ProviderModelInfo],
) -> list[str]:
    """Return ``provider/model`` refs to try when ``primary_ref`` fails.

    ``primary_ref`` (whatever model the request originally resolved to - the
    default, opus/sonnet/haiku role, or a direct ``provider/model`` request) is
    always first; remaining discovered chat models (those reachable with
    configured API keys) follow in descending heuristic potency. Non-chat
    models are excluded.
    """

    seen: set[str] = {primary_ref}
    rest: list[str] = []
    for info in model_infos:
        ref = info.model_id
        if ref in seen:
            continue
        if not is_chat_model(ref):
            continue
        seen.add(ref)
        rest.append(ref)

    rest.sort(key=lambda ref: (-rank_potency(ref), ref))
    chain = [primary_ref, *rest]
    logger.debug("MODEL FALLBACK CHAIN ({} candidates): {}", len(chain), chain)
    return chain
