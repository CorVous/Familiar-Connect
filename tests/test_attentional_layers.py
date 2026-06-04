"""Tests for attentional stream changes (issue #107).

Covers:
- RecentHistoryLayer.recent_messages uses recent_cross_channel
- Turn rendering includes #channel_id tag
- build_final_reminder with focus_channel_id renders directive
- build_final_reminder with unread_digest renders unreads block
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.context.layers import RecentHistoryLayer
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author


def _ctx(*, channel_id: int = 1, viewer_mode: str = "voice") -> AssemblyContext:
    return AssemblyContext(
        familiar_id="fam", channel_id=channel_id, viewer_mode=viewer_mode
    )


def _at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class TestRecentHistoryLayerCrossChannel:
    """RecentHistoryLayer must use recent_cross_channel instead of recent."""

    @pytest.mark.asyncio
    async def test_uses_recent_cross_channel_method(self) -> None:
        """recent_messages calls store.recent_cross_channel, not recent."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        # consumed=True is the default; turns are immediately consumed
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="message in channel 1",
            author=alice,
            consumed=True,
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=2,
            role="user",
            content="message in channel 2",
            author=alice,
            consumed=True,
        )

        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        # Both channels' turns should appear
        all_content = " ".join(m.content_str for m in msgs)
        assert "channel 1" in all_content
        assert "channel 2" in all_content

    @pytest.mark.asyncio
    async def test_only_consumed_turns_appear(self) -> None:
        """Staged (not consumed) turns must be excluded."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        # Consumed turn
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="consumed message",
            author=alice,
            consumed=True,
        )
        # Staged turn
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="staged message",
            author=alice,
            consumed=False,
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        all_content = " ".join(m.content_str for m in msgs)
        assert "consumed message" in all_content
        assert "staged message" not in all_content


class TestTurnRenderingChannelTag:
    """Turn rendering must include #channel_id in prefix."""

    @pytest.mark.asyncio
    async def test_user_turn_includes_channel_id_in_prefix(self) -> None:
        """User turns get [HH:MM speaker #channel_id] format."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="42", username="alice", display_name="Alice"
        )
        # consumed=True is the default
        store.append_turn(
            familiar_id="fam",
            channel_id=10,
            role="user",
            content="hello",
            author=alice,
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=10))
        user_msg = next(m for m in messages if m.role == "user")
        # Format: [HH:MM Alice #10] hello
        assert re.match(r"^\[\d{2}:\d{2} Alice #10\] hello$", user_msg.content_str), (
            user_msg.content_str
        )

    @pytest.mark.asyncio
    async def test_channel_id_tag_appears_in_all_user_turns(self) -> None:
        """All user turns from any channel carry the #channel_id tag."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="ch1 msg",
            author=alice,
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=2,
            role="user",
            content="ch2 msg",
            author=alice,
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        user_msgs = [m for m in msgs if m.role == "user"]
        for msg in user_msgs:
            # Each must match [HH:MM name #<id>] content
            assert re.match(r"^\[\d{2}:\d{2} \w+ #\d+\]", msg.content_str), (
                msg.content_str
            )

    @pytest.mark.asyncio
    async def test_message_id_tag_still_present_alongside_channel_tag(self) -> None:
        """platform_message_id still appears after speaker in prefix."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=5,
            role="user",
            content="hi",
            author=alice,
            platform_message_id="msg-99",
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=5))
        user_msg = next(m for m in msgs if m.role == "user")
        # Both channel and message id must be present
        assert "#5" in user_msg.content_str
        assert "#msg-99" in user_msg.content_str


class TestBuildFinalReminderFocusChannel:
    """build_final_reminder with focus_channel_id renders directive."""

    def test_focus_channel_directive_rendered(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=42,
        )
        assert "attention is currently on #42" in out
        # no unreads — no shift_focus mention needed
        assert "shift_focus" not in out

    def test_no_focus_channel_no_directive(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
        )
        assert "attention is currently on" not in out
        assert "shift_focus" not in out

    def test_focus_channel_id_zero_not_rendered(self) -> None:
        """None disables; providing focus_channel_id=None means no directive."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=None,
        )
        assert "shift_focus" not in out

    def test_focus_channel_before_post_history_instructions(self) -> None:
        """Focus directive appears before post_history_instructions."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=7,
            post_history_instructions="ETIQUETTE",
        )
        assert out.index("attention is currently on #7") < out.index("ETIQUETTE")


class TestBuildFinalReminderUnreadDigest:
    """build_final_reminder with unread_digest renders unreads block."""

    def test_unread_digest_rendered(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: 3, 20: 1},
        )
        assert "new message" in out
        assert "#10" in out
        assert "#20" in out
        assert "shift_focus" in out

    def test_empty_unread_digest_renders_nothing(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={},
        )
        assert "new message" not in out
        assert "shift_focus" not in out

    def test_zero_count_channels_excluded(self) -> None:
        """Channels with count=0 must not appear in the unreads line."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: 5, 20: 0, 30: 2},
        )
        assert "#10" in out
        assert "#30" in out
        assert "#20" not in out

    def test_none_unread_digest_renders_nothing(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest=None,
        )
        assert "new message" not in out
        assert "shift_focus" not in out

    def test_unread_digest_before_post_history_instructions(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={5: 2},
            post_history_instructions="TAIL_MARKER",
        )
        assert out.index("new message") < out.index("TAIL_MARKER")

    def test_focus_and_unread_both_present(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=3,
            unread_digest={10: 4},
        )
        assert "attention is currently on #3" in out
        assert "#10 (4)" in out
