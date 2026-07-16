"""Tests for application/active_model.py — status-line model recorder."""

from pathlib import Path
from unittest.mock import patch

from free_claude_code.application.active_model import write_active_model


def test_write_active_model_strips_provider_prefix_and_marks(tmp_path: Path) -> None:
    target = tmp_path / ".fcc" / "active_model"
    with patch(
        "free_claude_code.application.active_model.active_model_path",
        return_value=target,
    ):
        write_active_model("models/gemini-3-flash-preview")

    assert target.read_text(encoding="utf-8") == "⚡ gemini-3-flash-preview"


def test_write_active_model_keeps_bare_model_name(tmp_path: Path) -> None:
    target = tmp_path / ".fcc" / "active_model"
    with patch(
        "free_claude_code.application.active_model.active_model_path",
        return_value=target,
    ):
        write_active_model("llama-3.3-70b-versatile")

    assert target.read_text(encoding="utf-8") == "⚡ llama-3.3-70b-versatile"


def test_write_active_model_swallows_filesystem_errors(tmp_path: Path) -> None:
    target = tmp_path / ".fcc" / "active_model"
    with (
        patch(
            "free_claude_code.application.active_model.active_model_path",
            return_value=target,
        ),
        patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
    ):
        # A write failure on the hot response path must never raise.
        write_active_model("gpt-5")
