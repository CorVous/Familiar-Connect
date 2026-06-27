"""Tests for :mod:`familiar_connect.context.final_reminder`."""

from __future__ import annotations

from datetime import UTC, datetime

from familiar_connect.context.final_reminder import build_final_reminder


def _at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class TestBuildFinalReminder:
    def test_text_mode_lists_ping_and_reply_sentinels(self) -> None:
        out = build_final_reminder(viewer_mode="text", now=_at(2026, 5, 4, 14, 30))
        assert "It is now: 2026-05-04 2:30PM UTC" in out
        assert "[@DisplayName]" in out
        assert "[↩ <message_id>]" in out

    def test_text_mode_no_silent_sentinel(self) -> None:
        out = build_final_reminder(viewer_mode="text", now=_at(2026, 5, 4, 14, 30))
        assert "`<silent>`" not in out

    def test_voice_mode_no_sentinels(self) -> None:
        out = build_final_reminder(viewer_mode="voice", now=_at(2026, 1, 1, 9, 5))
        assert "It is now: 2026-01-01 9:05AM UTC" in out
        assert "[@DisplayName]" not in out
        assert "message_id" not in out
        assert "`<silent>`" not in out

    def test_display_tz_converts_clock_and_abbrev(self) -> None:
        # 21:30 UTC -> 14:30 PDT in summer Los Angeles
        out = build_final_reminder(
            viewer_mode="voice",
            now=_at(2026, 5, 4, 21, 30),
            display_tz="America/Los_Angeles",
        )
        assert "It is now: 2026-05-04 2:30PM PDT" in out

    def test_display_tz_defaults_to_utc(self) -> None:
        out = build_final_reminder(viewer_mode="voice", now=_at(2026, 5, 4, 21, 30))
        assert "It is now: 2026-05-04 9:30PM UTC" in out

    def test_starts_with_horizontal_rule(self) -> None:
        out = build_final_reminder(viewer_mode="text", now=_at(2026, 5, 4, 0, 0))
        assert out.startswith("---")

    def test_include_time_false_omits_timestamp(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            include_time=False,
        )
        assert "It is now:" not in out
        assert "[@DisplayName]" in out

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

    def test_post_history_instructions_appended_when_provided(self) -> None:
        out = build_final_reminder(
            viewer_mode="voice",
            now=_at(2026, 5, 4, 14, 30),
            post_history_instructions="# Etiquette\n\nBe terse.",
        )
        assert "# Etiquette" in out
        assert "Be terse." in out

    def test_post_history_instructions_land_at_tail(self) -> None:
        out = build_final_reminder(
            viewer_mode="voice",
            now=_at(2026, 5, 4, 14, 30),
            include_mode_instruction=True,
            post_history_instructions="ETIQUETTE_MARKER",
        )
        # deepest position — after the per-mode operating directive
        assert out.rstrip().endswith("ETIQUETTE_MARKER")
        assert out.index("You are speaking aloud") < out.index("ETIQUETTE_MARKER")

    def test_post_history_instructions_omitted_by_default(self) -> None:
        out = build_final_reminder(viewer_mode="voice", now=_at(2026, 5, 4, 14, 30))
        assert "Etiquette" not in out

    def test_blank_post_history_instructions_appends_nothing(self) -> None:
        out = build_final_reminder(
            viewer_mode="voice",
            now=_at(2026, 5, 4, 14, 30),
            post_history_instructions="   ",
        )
        # whitespace-only must not leave a dangling separator
        assert not out.endswith("\n\n")


class TestBuildFinalReminderGuildName:
    """guild_name names the current server on the focus line (multi-server bot)."""

    def test_guild_name_named_alongside_channel_on_focus_line(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=123,
            channel_names={123: "general"},
            guild_name="My Server",
        )
        focus_line = next(
            ln for ln in out.splitlines() if "attention is currently on" in ln
        )
        assert "#general" in focus_line
        assert "My Server" in focus_line

    def test_no_guild_name_output_byte_for_byte_unchanged(self) -> None:
        """Regression guard: omitting guild_name must reproduce today's output."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=123,
            channel_names={123: "general"},
        )
        expected = (
            "---\n"
            "\n"
            "It is now: 2026-05-04 2:30PM UTC\n"
            "\n"
            "Special input:\n"
            "\n"
            "* `[@DisplayName]` - ping user\n"
            "* `[↩ <message_id>]` - reply to message\n"
            "\n"
            "Your attention is currently on #general."
        )
        assert out == expected
        assert "server" not in out

    def test_guild_name_none_keeps_plain_focus_line(self) -> None:
        """DM / unknown server: guild_name=None renders the unchanged focus line."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=123,
            channel_names={123: "general"},
            guild_name=None,
        )
        assert "Your attention is currently on #general." in out
        assert "server" not in out

    def test_guild_name_without_focus_channel_leaks_no_server_text(self) -> None:
        """Boundary: server name rides the focus line; no focus line, no server."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=None,
            guild_name="My Server",
        )
        assert "My Server" not in out
        assert "server" not in out

    def test_server_clause_stays_on_focus_sentence_not_unread(self) -> None:
        """Server clause stays on the focus sentence, not the unread one.

        With focus + unread + guild all set, the server text must attach to
        the focus sentence and not tangle into the unread sentence.
        """
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=123,
            unread_digest={20: 2},
            channel_names={123: "general", 20: "random"},
            guild_name="My Server",
        )
        # focus sentence ends at "server." then the unread sentence begins
        assert "server. There " in out
        assert out.index("My Server") < out.index("There are")

    def test_empty_guild_name_identical_to_none(self) -> None:
        """Falsy contract: guild_name="" behaves exactly like guild_name=None."""
        common = {
            "viewer_mode": "text",
            "now": _at(2026, 5, 4, 14, 30),
            "focus_channel_id": 123,
            "channel_names": {123: "general"},
        }
        assert build_final_reminder(**common, guild_name="") == build_final_reminder(
            **common, guild_name=None
        )

    def test_server_clause_exact_wording(self) -> None:
        """AC1 tightened: the quotes and "the" in the server clause are pinned."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=123,
            channel_names={123: "general"},
            guild_name="My Server",
        )
        focus_line = next(
            ln for ln in out.splitlines() if "attention is currently on" in ln
        )
        assert 'in the "My Server" server.' in focus_line
