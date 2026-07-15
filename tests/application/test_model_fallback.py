from free_claude_code.application.model_fallback import (
    build_fallback_chain,
    eligible_candidate_refs,
    is_chat_model,
    is_free_candidate,
    rank_potency,
)
from free_claude_code.application.model_metadata import ProviderModelInfo


def _info(model_id: str) -> ProviderModelInfo:
    return ProviderModelInfo(model_id)


def test_eligible_candidate_refs_excludes_paid_and_non_chat_ordered_by_potency():
    refs = eligible_candidate_refs(
        [
            _info("open_router/small-model:free"),
            _info("open_router/opus-tier-model:free"),
            _info("open_router/anthropic/claude-opus-4.8"),
            _info("open_router/whisper-large:free"),
            _info("gemini/gemini-3.1-flash-lite"),
        ]
    )

    assert "open_router/anthropic/claude-opus-4.8" not in refs
    assert "open_router/whisper-large:free" not in refs
    assert refs.index("open_router/opus-tier-model:free") < refs.index(
        "open_router/small-model:free"
    )
    assert "gemini/gemini-3.1-flash-lite" in refs


def test_build_fallback_chain_always_keeps_primary_first():
    chain = build_fallback_chain(
        "gemini/gemini-3.1-flash-lite",
        [_info("gemini/gemini-3.1-flash-lite"), _info("open_router/some/model:free")],
    )

    assert chain[0] == "gemini/gemini-3.1-flash-lite"
    assert chain.count("gemini/gemini-3.1-flash-lite") == 1


def test_build_fallback_chain_orders_remaining_candidates_by_potency():
    chain = build_fallback_chain(
        "open_router/primary:free",
        [
            _info("open_router/small-model:free"),
            _info("open_router/opus-tier-model:free"),
            _info("open_router/sonnet-tier-model:free"),
        ],
    )

    assert chain == [
        "open_router/primary:free",
        "open_router/opus-tier-model:free",
        "open_router/sonnet-tier-model:free",
        "open_router/small-model:free",
    ]


def test_build_fallback_chain_excludes_non_chat_models():
    chain = build_fallback_chain(
        "open_router/primary:free",
        [
            _info("open_router/primary:free"),
            _info("open_router/whisper-large:free"),
            _info("open_router/text-embedding-3:free"),
        ],
    )

    assert chain == ["open_router/primary:free"]


def test_build_fallback_chain_excludes_paid_openrouter_models():
    """A fallback the user never configured must never be able to cost money."""
    chain = build_fallback_chain(
        "open_router/openai/gpt-oss-20b:free",
        [
            _info("open_router/openai/gpt-oss-20b:free"),
            _info("open_router/anthropic/claude-opus-4.8"),
            _info("open_router/tencent/hy3:free"),
        ],
    )

    assert chain == [
        "open_router/openai/gpt-oss-20b:free",
        "open_router/tencent/hy3:free",
    ]


def test_build_fallback_chain_keeps_non_openrouter_candidates_regardless_of_suffix():
    chain = build_fallback_chain(
        "open_router/openai/gpt-oss-20b:free",
        [
            _info("open_router/openai/gpt-oss-20b:free"),
            _info("gemini/gemini-3.1-flash-lite"),
            _info("nvidia_nim/nvidia/nemotron-3-super-120b-a12b"),
        ],
    )

    assert set(chain) == {
        "open_router/openai/gpt-oss-20b:free",
        "gemini/gemini-3.1-flash-lite",
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
    }


def test_build_fallback_chain_allows_an_explicitly_configured_paid_primary():
    """Filtering only applies to auto-added candidates, never the user's own choice."""
    chain = build_fallback_chain(
        "open_router/anthropic/claude-opus-4.8",
        [_info("open_router/anthropic/claude-opus-4.8")],
    )

    assert chain == ["open_router/anthropic/claude-opus-4.8"]


def test_is_free_candidate_requires_free_suffix_only_for_openrouter():
    assert is_free_candidate("open_router/some/model:free") is True
    assert is_free_candidate("open_router/anthropic/claude-opus-4.8") is False
    assert is_free_candidate("gemini/gemini-3.1-flash-lite") is True
    assert is_free_candidate("nvidia_nim/nvidia/nemotron") is True


def test_is_free_candidate_excludes_paid_gemini_pro_models():
    # Gemini Pro models are paid-only, even with billing enabled.
    assert is_free_candidate("gemini/models/gemini-2.5-pro") is False
    assert is_free_candidate("gemini/models/gemini-3.1-pro-preview") is False
    assert is_free_candidate("gemini/models/gemini-pro-latest") is False
    # Flash/Flash-Lite/Gemma stay free.
    assert is_free_candidate("gemini/models/gemini-3.5-flash") is True
    assert is_free_candidate("gemini/models/gemma-4-31b-it") is True


def test_gemma_outranks_flash_so_derivation_prefers_the_high_quota_model():
    assert rank_potency("gemini/models/gemma-4-31b-it") > rank_potency(
        "gemini/models/gemini-3.5-flash"
    )


def test_is_chat_model_excludes_non_chat_markers():
    assert is_chat_model("open_router/openai/gpt-oss-20b:free") is True
    assert is_chat_model("open_router/openai/whisper-large-v3") is False
    assert is_chat_model("open_router/google/text-embedding-3") is False


def test_rank_potency_orders_known_families_above_unknown():
    assert rank_potency("open_router/anthropic/claude-opus-4.8") > rank_potency(
        "open_router/some/mystery-model"
    )
    assert rank_potency("open_router/anthropic/claude-sonnet-4-5") > rank_potency(
        "open_router/openai/gpt-oss-20b:free"
    )


def test_rank_potency_orders_pro_above_flash_above_flash_lite():
    pro = rank_potency("gemini/gemini-3.1-pro-preview")
    flash = rank_potency("gemini/gemini-3.5-flash")
    flash_lite = rank_potency("gemini/gemini-3.1-flash-lite")

    assert pro > flash > flash_lite


def test_rank_potency_does_not_mistake_gemini_for_mini():
    # "gemini" contains "mini" as a substring but must not read as a small model.
    assert rank_potency("gemini/gemini-2.5-pro") > rank_potency(
        "gemini/gemini-2.5-flash"
    )


def test_gemma_ranks_by_size_not_by_family_name():
    # Gemma 4 26B/31B are mid-size, not small: they must outrank flash-lite.
    assert rank_potency("gemini/models/gemma-4-31b-it") > rank_potency(
        "gemini/models/gemini-3.1-flash-lite"
    )
    assert rank_potency("gemini/models/gemma-4-26b-a4b-it") > rank_potency(
        "gemini/models/gemini-3.1-flash-lite"
    )


def test_music_and_video_models_are_excluded_as_non_chat():
    assert is_chat_model("gemini/models/lyria-3-pro-preview") is False
    assert is_chat_model("gemini/models/veo-3-generate") is False


def test_rank_potency_breaks_ties_by_version_then_params():
    assert rank_potency("gemini/gemini-3.5-flash") > rank_potency(
        "gemini/gemini-2.0-flash"
    )
    assert rank_potency("open_router/nvidia/nemotron-3-ultra-550b:free") > rank_potency(
        "open_router/nvidia/nemotron-3-super-120b:free"
    )
