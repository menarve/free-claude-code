"""Shared filesystem paths for Free Claude Code configuration."""

from pathlib import Path

FCC_CONFIG_DIRNAME = ".fcc"
FCC_ENV_FILENAME = ".env"
LEGACY_REPO_DIRNAME = "free-claude-code"
LEGACY_XDG_CONFIG_DIRNAME = ".config"
MESSAGING_STATE_DIRNAME = "agent_workspace"
FCC_LOGS_DIRNAME = "logs"
SERVER_LOG_FILENAME = "server.log"
CODEX_MODEL_CATALOG_FILENAME = "codex-model-catalog.json"
USAGE_STATS_FILENAME = "usage_stats.json"
ACTIVE_MODEL_FILENAME = "active_model"
CLAUDE_SETTINGS_OVERRIDE_FILENAME = "claude-settings.json"


def config_dir_path() -> Path:
    """Return the default user config directory."""

    return Path.home() / FCC_CONFIG_DIRNAME


def managed_env_path() -> Path:
    """Return the default user-managed env file path."""

    return config_dir_path() / FCC_ENV_FILENAME


def legacy_env_paths() -> tuple[Path, ...]:
    """Return legacy user env paths that can be migrated to ~/.fcc/.env."""

    home = Path.home()
    return (
        home / LEGACY_REPO_DIRNAME / FCC_ENV_FILENAME,
        home / LEGACY_XDG_CONFIG_DIRNAME / LEGACY_REPO_DIRNAME / FCC_ENV_FILENAME,
    )


def messaging_state_dir_path() -> Path:
    """Return the managed messaging state directory."""

    return config_dir_path() / MESSAGING_STATE_DIRNAME


def server_log_path() -> Path:
    """Return the canonical server log path."""

    return config_dir_path() / FCC_LOGS_DIRNAME / SERVER_LOG_FILENAME


def codex_model_catalog_path() -> Path:
    """Return the generated Codex model catalog path."""

    return config_dir_path() / CODEX_MODEL_CATALOG_FILENAME


def usage_stats_path() -> Path:
    """Return the persisted per-model usage stats path."""

    return config_dir_path() / USAGE_STATS_FILENAME


def active_model_path() -> Path:
    """Return the path holding the model that last served a response.

    A status-line command reads this so the user can see which model the
    derivation chose for the current turn.
    """

    return config_dir_path() / ACTIVE_MODEL_FILENAME


def claude_settings_override_path() -> Path:
    """Return the fcc-only Claude Code settings file.

    Passed to Claude Code via ``--settings`` so fcc-claude carries its own
    model and status line WITHOUT touching the user's global
    ``~/.claude/settings.json`` (which their real ``claude`` plan uses).
    """

    return config_dir_path() / CLAUDE_SETTINGS_OVERRIDE_FILENAME
