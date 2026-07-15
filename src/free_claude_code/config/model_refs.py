"""Provider-prefixed model reference helpers."""

from dataclasses import dataclass
from typing import Protocol

# Sentinel a model role can point to instead of a concrete provider/model. It
# means "don't pin a model - run the derivation chain, trying every accessible
# model strongest-first until one responds". It is not a real provider/model,
# so it is excluded from configured-model validation and discovery.
DERIVATION_MODEL_REF = "menarve/derivation"
DERIVATION_DISPLAY_NAME = "Derivación Menarve"


def is_derivation_ref(model_ref: str | None) -> bool:
    """Return whether a configured role points at the derivation chain."""

    return model_ref == DERIVATION_MODEL_REF


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference and the env keys that set it."""

    model_ref: str
    provider_id: str
    model_id: str
    sources: tuple[str, ...]


class ChatModelConfig(Protocol):
    model: str
    model_fable: str | None
    model_opus: str | None
    model_sonnet: str | None
    model_haiku: str | None


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return model_ref.split("/", 1)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return model_ref.split("/", 1)[1]


def configured_chat_model_refs(
    settings: ChatModelConfig,
) -> tuple[ConfiguredChatModelRef, ...]:
    """Return unique configured chat provider/model refs with source env keys."""

    candidates = (
        ("MODEL", settings.model),
        ("MODEL_FABLE", settings.model_fable),
        ("MODEL_OPUS", settings.model_opus),
        ("MODEL_SONNET", settings.model_sonnet),
        ("MODEL_HAIKU", settings.model_haiku),
    )
    sources_by_ref: dict[str, list[str]] = {}
    for source, model_ref in candidates:
        # The derivation sentinel is not a concrete provider/model; excluding it
        # here keeps validation, discovery, and the model catalog from treating
        # "menarve" as a real provider.
        if model_ref is None or is_derivation_ref(model_ref):
            continue
        sources_by_ref.setdefault(model_ref, []).append(source)

    return tuple(
        ConfiguredChatModelRef(
            model_ref=model_ref,
            provider_id=parse_provider_type(model_ref),
            model_id=parse_model_name(model_ref),
            sources=tuple(sources),
        )
        for model_ref, sources in sources_by_ref.items()
    )
