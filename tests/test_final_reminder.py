"""Tests for :mod:`familiar_connect.context.final_reminder`."""

from __future__ import annotations

from datetime import UTC, datetime

from familiar_connect.context.final_reminder import build_final_reminder


def _at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class TestBuildFinalReminder:
    def test_text_mode_lists_all_sentinels(self) -> None:
        out = build_final_reminder(viewer_mode="text", now=_at(2026, 5, 4, 14, 30))
        assert "It is now: 2026-05-04 2:30PM UTC" in out
        assert "<silent>" in out
        assert "[@DisplayName]" in out
        assert "[↩ <message_id>]" in out

    def test_voice_mode_only_silent_sentinel(self) -> None:
        out = build_final_reminder(viewer_mode="voice", now=_at(2026, 1, 1, 9, 5))
        assert "It is now: 2026-01-01 9:05AM UTC" in out
        assert "<silent>" in out
        assert "[@DisplayName]" not in out
        assert "message_id" not in out

    def test_starts_with_horizontal_rule(self) -> None:
        out = build_final_reminder(viewer_mode="text", now=_at(2026, 5, 4, 0, 0))
        assert out.startswith("---")

    def test_voice_mode_instruction_appended_when_requested(self) -> None:
        out = build_final_reminder(
            viewer_mode="voice",
            now=_at(2026, 5, 4, 14, 30),
            include_mode_instruction=True,
        )
        assert "You are speaking aloud" in out
        assert "Avoid markdown" in out

    def test_text_mode_instruction_appended_when_requested(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            include_mode_instruction=True,
        )
        assert "chatting in a text channel" in out
        assert "Markdown" in out

    def test_mode_instruction_omitted_by_default(self) -> None:
        out = build_final_reminder(viewer_mode="voice", now=_at(2026, 5, 4, 14, 30))
        assert "You are speaking aloud" not in out

    def test_unknown_mode_with_instruction_flag_is_silent(self) -> None:
        out = build_final_reminder(
            viewer_mode="other",
            now=_at(2026, 5, 4, 14, 30),
            include_mode_instruction=True,
        )
        assert "You are speaking aloud" not in out
        assert "Markdown" not in out
