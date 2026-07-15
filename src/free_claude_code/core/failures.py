"""Protocol-neutral execution failure semantics."""

from dataclasses import FrozenInstanceError, dataclass
from enum import StrEnum


class FailureKind(StrEnum):
    """Stable failure categories shared across execution and wire adapters."""

    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    UPSTREAM = "upstream"
    UNAVAILABLE = "unavailable"


@dataclass(eq=False)
class ExecutionFailure(Exception):
    """A finalized provider-execution failure independent of any wire protocol."""

    kind: FailureKind
    status_code: int
    message: str
    retryable: bool
    # Distinct from `retryable` (same-provider backoff): set when a candidate
    # in a different model's fallback chain might still succeed, e.g. a
    # context-length-exceeded 400 that a larger-context model could accept.
    model_fallback_eligible: bool = False

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    _FIELD_NAMES = (
        "kind",
        "status_code",
        "message",
        "retryable",
        "model_fallback_eligible",
    )

    def __setattr__(self, name: str, value: object) -> None:
        # Exception machinery must be able to update __traceback__, __cause__,
        # and __context__ while semantic failure fields remain immutable.
        # `name in self.__dict__` (not `hasattr`) so defaulted fields - whose
        # default value lives on the class until first assigned - aren't
        # mistaken for already-set instance state during __init__.
        if name in self._FIELD_NAMES and name in self.__dict__:
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        Exception.__setattr__(self, name, value)


def find_execution_failure(exc: BaseException) -> ExecutionFailure | None:
    """Return the first canonical failure in an exception or nested group."""
    pending = [exc]
    while pending:
        current = pending.pop()
        if isinstance(current, ExecutionFailure):
            return current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
    return None
