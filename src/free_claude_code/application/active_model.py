"""Record the model that most recently served a response.

A status-line command (`cat ~/.fcc/active_model`) reads the file this writes so
the user can see which model the derivation chose for the current turn.
"""

from free_claude_code.config.paths import active_model_path


def write_active_model(provider_model: str) -> None:
    """Persist the model that just committed a response, best-effort.

    Called on the first streamed chunk, so a failure here must never break the
    response: all filesystem errors are swallowed. The stored value is the bare
    model name (any ``provider/`` or ``models/`` prefix stripped) with a small
    marker, e.g. ``⚡ gemini-3-flash-preview``.
    """

    short = provider_model.rsplit("/", 1)[-1]
    try:
        path = active_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"⚡ {short}", encoding="utf-8")
    except OSError:
        pass
