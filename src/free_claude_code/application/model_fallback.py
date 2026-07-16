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
# "oss" (open-source) is likewise NOT a size marker: gpt-oss-120b is a large
# model and must be classed by its "120b", not sunk as if it were small.
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
    "orpheus",
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
    """Return whether a ref is a good automatic derivation candidate.

    Automatic derivation must never spend money nor waste attempts on models
    that reliably reject real requests:

    - OpenRouter mixes free and paid models with no price field; only ``:free``
      ids are safe.
    - Gemini serves Flash/Flash-Lite free even with billing enabled, but its
      ``pro`` models are paid-only (excluded), and its ``gemma`` models cap
      input at 16K tokens/minute - far below a real coding request - so they
      reject almost everything and are excluded too.

    Other providers are single-tier (free, or bring-your-own-key at a rate the
    user already accepted), so they pass through.
    """

    if model_ref.startswith("open_router/"):
        return model_ref.endswith(":free")
    if model_ref.startswith("gemini/"):
        ref = model_ref.lower()
        return "pro" not in ref and "gemma" not in ref
    return True


# Curated coding-capability order, BEST first. A model's rank is the position
# of the first family it matches; parameter count must not push a big-but-weaker
# model above a frontier one (e.g. gpt-oss-120b or llama-3.1-405b above gpt-5).
# This is opinionated and time-sensitive - update it as frontier models change.
# Anything not listed falls back to the name/size heuristic, always ranked below
# the known families. `-mini/-nano/-lite` variants are demoted below their
# full-size siblings.
_CODING_ORDER = (
    r"claude-opus",
    r"gpt-5",
    r"claude-sonnet",
    r"deepseek-r\d",
    r"(?:^|[/-])o[1-9](?:[/-]|$)",
    r"gpt-4\.1",
    r"gpt-4o",
    r"deepseek-v\d",
    r"codestral|devstral|qwen[\d.]*-?coder",
    r"magistral",
    r"grok-[4-9]",
    r"gemini-3(?:\.\d+)?-pro",
    r"llama-4",
    r"llama-3\.3",
    r"llama-3\.1",
    r"nemotron",
    r"mistral-(?:large|medium)",
    r"gemini-3(?:\.\d+)?-flash",
    r"qwen-?3",
    r"kimi",
    r"minimax",
    r"gpt-oss",
    r"gemini-2\.5-pro",
    r"gemini-2\.5-flash",
    r"command-a|command-r",
    r"gemma-4",
    r"glm-4|zai-",
    r"gemini-2\.0-flash",
    r"gemini[\w.-]*flash",
)
_CODING_PATTERNS = tuple(re.compile(pattern) for pattern in _CODING_ORDER)


def _size_score(ref: str) -> int:
    """Name/size heuristic used to rank models the curated table does not know."""

    sizes = [float(match) for match in _SIZE_RE.findall(ref)]
    params = max(sizes) if sizes else 0.0
    size_class = _size_class(ref, params)
    version_match = _VERSION_RE.search(ref)
    version = float(version_match.group(1)) if version_match else 0.0
    return int(size_class * 10000 + min(version, 99) * 100 + min(params, 999))


def _coding_index(ref: str) -> int | None:
    """Index of the first curated family a ref matches, or None if unknown."""

    for index, pattern in enumerate(_CODING_PATTERNS):
        if pattern.search(ref):
            return index
    return None


def rank_potency(model_ref: str) -> int:
    """Score for ordering candidates strongest -> weakest for coding.

    The curated `_CODING_ORDER` table decides the order of known frontier
    families so a big open-weight model never outranks a frontier one. A
    `-mini/-nano/-lite` variant of a known family sits below its full-size
    siblings, and families the table does not know fall back to the pure size
    heuristic, always below the curated band.
    """

    ref = model_ref.lower()
    index = _coding_index(ref)
    if index is None:
        return _size_score(ref)
    # Within a curated tier, break ties by real parameter count (0-999b) so a
    # bigger member of a family (nemotron-ultra-550b) outranks a smaller one
    # (nemotron-super-120b). The tier itself always dominates the tiebreak.
    sizes = [float(match) for match in _SIZE_RE.findall(ref)]
    params = int(max(sizes)) if sizes else 0
    curated = (len(_CODING_PATTERNS) - index) * 1000 + min(params, 999)
    is_small_variant = bool(set(_WORD_RE.findall(ref)) & _SMALL_TOKENS)
    return (1_000_000 if is_small_variant else 2_000_000) + curated


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
