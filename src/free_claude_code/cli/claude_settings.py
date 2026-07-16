"""fcc-only Claude Code settings, isolated from the user's global config.

fcc-claude launches the official Claude Code binary with
``--settings <this file>``, which loads ADDITIONAL settings that merge on top
of (and win over) the user's ``~/.claude/settings.json``. This lets fcc-claude
carry its own model and status line WITHOUT contaminating the plain ``claude``
command the user runs on their paid plan.

- ``model: haiku`` caps the Claude Code context window at 200K (vs 1M for
  Opus/Sonnet), forcing earlier auto-compaction. The derivation still routes
  the request to the best available free model server-side; only the client
  bookkeeping tracks the smaller window.
- ``statusLine`` prints the model the derivation actually served, read from the
  file ``active_model`` writes on the first streamed chunk.
"""

import json

from free_claude_code.config.paths import (
    active_model_path,
    claude_settings_override_path,
)


def _fcc_claude_settings() -> dict[str, object]:
    """Build the isolated settings, resolving paths at call time."""

    # Read the model the derivation served this turn; fall back to a neutral
    # label before the first response has committed a model.
    status_command = (
        f'cat "{active_model_path()}" 2>/dev/null || printf "derivación Menarve"'
    )
    return {
        "model": "haiku",
        "statusLine": {
            "type": "command",
            "command": status_command,
        },
    }


def ensure_fcc_claude_settings() -> str:
    """Write the fcc-only settings file and return its path.

    Rewritten on every launch so upgrades to the isolated settings take effect
    without the user deleting a stale file.
    """

    path = claude_settings_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_fcc_claude_settings(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)
