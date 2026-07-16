import json

from free_claude_code.cli.claude_settings import ensure_fcc_claude_settings
from free_claude_code.cli.launchers.claude import build_claude_launcher_command
from free_claude_code.config import paths


def test_ensure_fcc_claude_settings_writes_isolated_model_and_status_line(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paths, "config_dir_path", lambda: tmp_path)

    path = ensure_fcc_claude_settings()

    data = json.loads(tmp_path.joinpath("claude-settings.json").read_text("utf-8"))
    assert path == str(tmp_path / "claude-settings.json")
    # Haiku caps the client context window at 200K without touching the global.
    assert data["model"] == "haiku"
    assert data["statusLine"]["type"] == "command"
    # The status line reads the model the derivation actually served.
    assert str(tmp_path / paths.ACTIVE_MODEL_FILENAME) in data["statusLine"]["command"]


def test_build_claude_launcher_command_prepends_isolated_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paths, "config_dir_path", lambda: tmp_path)

    command = build_claude_launcher_command(
        binary_path="/usr/bin/claude", argv=["--resume", "prompt text"]
    )

    override = str(tmp_path / "claude-settings.json")
    assert command == [
        "/usr/bin/claude",
        "--settings",
        override,
        "--resume",
        "prompt text",
    ]
