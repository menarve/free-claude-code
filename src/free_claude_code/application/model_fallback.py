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

import re
from collections.abc import Iterable

from loguru import logger

from free_claude_code.application.model_metadata import ProviderModelInfo

# Potency is name-based only: ProviderModelInfo carries no capability metadata.
# A model's size CLASS dominates the score; version and parameter count only
# break ties within a class. Markers are matched against whole letter-tokens
# (not substrings) so "flash-lite" reads as small and "gemini" is never
# mistaken for "mini".
# "gemma" is intentionally NOT a small marker: Gemma 4 26B/31B are mid-size
# models whose parameter count places them correctly. Their class is decided by
# the "26b"/"31b" in the name, not the family word.
_SMALL_TOKENS = frozenset(
    {
        "lite",
        "mini",
        "nano",
        "micro",
        "tiny",
        "small",
        "haiku",
        "xs",
        "phi",
        "oss",
    }
)
_LARGE_TOKENS = frozenset({"opus", "ultra", "max", "pro", "large", "nemotron"})
_MEDIUM_TOKENS = frozenset(
    {"flash", "sonnet", "plus", "medium", "coder", "command", "mistral"}
)

_CLASS_SMALL = 1
_CLASS_MEDIUM = 2
_CLASS_LARGE = 3

_WORD_RE = re.compile(r"[a-z]+")
# Explicit parameter count in the name, e.g. "550b", "120b", "8b".
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])")
# Family version, e.g. gemini-3.1, gpt-5, claude-opus-4.8, qwen3.
_VERSION_RE = re.compile(
    r"(?:gpt-|gemini-|claude-[a-z]+-|grok-|llama-|qwen-?|deepseek-|"
    r"mistral-|hy|nemotron-|command-r-?|-v|/v)(\d+(?:\.\d+)?)"
)


def _size_class(ref: str, params: float) -> int:
    """Coarse capability class from name tokens and any explicit param count."""
    words = set(_WORD_RE.findall(ref))
    if words & _SMALL_TOKENS:
        return _CLASS_SMALL
    if params >= 100 or (words & _LARGE_TOKENS):
        return _CLASS_LARGE
    if params and params <= 9:
        return _CLASS_SMALL
    if words & _MEDIUM_TOKENS:
        return _CLASS_MEDIUM
    return _CLASS_MEDIUM


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
    "lyria",
    "veo",
    # Agent/specialized Gemini models that only serve the Interactions or Live
    # API, not chat completions - unusable as derivation candidates.
    "antigravity",
    "deep-research",
    "computer-use",
    "robotics",
    "-live",
    "live-",
)


def is_chat_model(model_ref: str) -> bool:
    """Return whether a ``provider/model`` ref looks like a chat model."""

    ref = model_ref.lower()
    return not any(marker in ref for marker in _NON_CHAT_MARKERS)


def is_free_candidate(model_ref: str) -> bool:
    """Return whether a ref is safe to try automatically without user consent.

    OpenRouter's catalog (the source of ``model_infos``) mixes free and paid
    frontier models with no price field - ``:free`` in the id is the only
    signal - so an unfiltered chain could silently spend real money on a paid
    model the user never configured. Other providers here are single-tier
    (free or bring-your-own-key at whatever rate the user already accepted),
    so only OpenRouter needs this extra check.
    """

    if not model_ref.startswith("open_router/"):
        return True
    return model_ref.endswith(":free")


def rank_potency(model_ref: str) -> int:
    """Heuristic score for ordering candidates strongest -> weakest.

    Size class dominates (large > medium > small); the family version number
    and any explicit parameter count only break ties within a class. Name-based
    only, since ProviderModelInfo carries no capability metadata.
    """

    ref = model_ref.lower()
    sizes = [float(match) for match in _SIZE_RE.findall(ref)]
    params = max(sizes) if sizes else 0.0
    size_class = _size_class(ref, params)
    version_match = _VERSION_RE.search(ref)
    version = float(version_match.group(1)) if version_match else 0.0
    # Class dominates; version then parameter count break within-class ties.
    return int(size_class * 10000 + min(version, 99) * 100 + min(params, 999))


def eligible_candidate_refs(
    model_infos: Iterable[ProviderModelInfo],
) -> list[str]:
    """Discovered chat models the fallback may auto-substitute, most potent first.

    Excludes non-chat models (embeddings, image, TTS, ...) and paid OpenRouter
    models - an automatic substitution the user never requested must never
    reach a model that could cost money. This is the single source of truth
    for "models the derivation system can use", shared by the fallback chain
    and the admin Usage tab so they never drift apart.
    """

    seen: set[str] = set()
    refs: list[str] = []
    for info in model_infos:
        ref = info.model_id
        if ref in seen:
            continue
        if not is_chat_model(ref) or not is_free_candidate(ref):
            continue
        seen.add(ref)
        refs.append(ref)

    refs.sort(key=lambda ref: (-rank_potency(ref), ref))
    return refs


def build_fallback_chain(
    primary_ref: str,
    model_infos: Iterable[ProviderModelInfo],
) -> list[str]:
    """Return ``provider/model`` refs to try when ``primary_ref`` fails.

    ``primary_ref`` (whatever model the request originally resolved to - the
    default, opus/sonnet/haiku role, or a direct ``provider/model`` request) is
    always first; the eligible discovered candidates follow in descending
    heuristic potency.
    """

    rest = [ref for ref in eligible_candidate_refs(model_infos) if ref != primary_ref]
    chain = [primary_ref, *rest]
    logger.debug("MODEL FALLBACK CHAIN ({} candidates): {}", len(chain), chain)
    return chain
