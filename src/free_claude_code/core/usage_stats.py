"""Per-model request/token/error counters shown in the admin Usage tab."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from free_claude_code.core.json_persistence import DebouncedJsonPersistence


@dataclass(frozen=True, slots=True)
class ModelUsageStats:
    """Accumulated counters for one ``provider_id/provider_model`` key."""

    requests: int = 0
    errors: int = 0
    input_tokens: int = 0
    last_used_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_json(cls, data: Any) -> "ModelUsageStats":
        if not isinstance(data, dict):
            return cls()
        requests = data.get("requests")
        errors = data.get("errors")
        input_tokens = data.get("input_tokens")
        last_used_at = data.get("last_used_at")
        return cls(
            requests=requests if isinstance(requests, int) else 0,
            errors=errors if isinstance(errors, int) else 0,
            input_tokens=input_tokens if isinstance(input_tokens, int) else 0,
            last_used_at=last_used_at if isinstance(last_used_at, str) else None,
        )


class UsageStatsTracker:
    """Process-lifetime aggregator, debounce-persisted to ``storage_path``."""

    def __init__(self, storage_path: str) -> None:
        self._stats: dict[str, ModelUsageStats] = {}
        self._dirty = False
        self._persistence = DebouncedJsonPersistence(
            storage_path,
            snapshot=self._snapshot_for_persistence,
            on_dirty=self._set_dirty,
        )

    def load(self) -> None:
        """Restore counters saved by a previous process, if any."""
        data = self._persistence.load_json()
        models = data.get("models")
        if not isinstance(models, dict):
            return
        for key, value in models.items():
            if isinstance(key, str):
                self._stats[key] = ModelUsageStats.from_json(value)

    def record_success(
        self, provider_id: str, provider_model: str, *, input_tokens: int
    ) -> None:
        key = f"{provider_id}/{provider_model}"
        current = self._stats.get(key, ModelUsageStats())
        self._stats[key] = replace(
            current,
            requests=current.requests + 1,
            input_tokens=current.input_tokens + input_tokens,
            last_used_at=_now_iso(),
        )
        self._persistence.schedule_save()

    def record_error(self, provider_id: str, provider_model: str) -> None:
        key = f"{provider_id}/{provider_model}"
        current = self._stats.get(key, ModelUsageStats())
        self._stats[key] = replace(
            current,
            errors=current.errors + 1,
            last_used_at=_now_iso(),
        )
        self._persistence.schedule_save()

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_for_persistence()

    def _snapshot_for_persistence(self) -> dict[str, Any]:
        return {
            "models": {
                key: stats.to_json() for key, stats in sorted(self._stats.items())
            }
        }

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
