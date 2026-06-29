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
from typing import TYPE_CHECKING

import pytest

from familiar_connect.budget import estimate_message_tokens
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.context.layers import RecentHistoryLayer
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author

if TYPE_CHECKING:
    from familiar_connect.llm import Message


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


class TestRecentHistoryLayerArchiveWatermark:
    """Window resets at the per-channel archive watermark (activities).

    Watermark hides ids at/below it for its own channel only —
    ``read_channel`` and other store callers stay unfiltered.
    """

    @staticmethod
    def _turn(store: HistoryStore, *, channel_id: int, content: str):  # noqa: ANN205
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        return store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user",
            content=content,
            author=alice,
            consumed=True,
        )

    @pytest.mark.asyncio
    async def test_no_watermark_includes_all_turns(self) -> None:
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="before departure")
        self._turn(store, channel_id=1, content="after return")
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        all_content = " ".join(m.content_str for m in msgs)
        assert "before departure" in all_content
        assert "after return" in all_content

    @pytest.mark.asyncio
    async def test_watermark_hides_pre_archive_turns(self) -> None:
        store = HistoryStore(":memory:")
        departure = self._turn(store, channel_id=1, content="before departure")
        store.set_archive_watermark(
            familiar_id="fam", channel_id=1, turn_id=departure.id
        )
        self._turn(store, channel_id=1, content="after return")
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        all_content = " ".join(m.content_str for m in msgs)
        # watermark sits at the departure turn id: it and earlier hidden
        assert "before departure" not in all_content
        assert "after return" in all_content

    @pytest.mark.asyncio
    async def test_watermark_does_not_leak_across_channels(self) -> None:
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=2, content="other channel chatter")
        departure = self._turn(store, channel_id=1, content="before departure")
        store.set_archive_watermark(
            familiar_id="fam", channel_id=1, turn_id=departure.id
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        all_content = " ".join(m.content_str for m in msgs)
        # channel 2's older turn survives channel 1's watermark
        assert "other channel chatter" in all_content
        assert "before departure" not in all_content


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


def _is_marker(msg: Message) -> bool:
    """Channel-change separators are bare lines; rendered turns are bracketed."""
    return not msg.content_str.startswith("[")


def _markers(msgs: list) -> list[str]:
    return [m.content_str for m in msgs if _is_marker(m)]


def _marker_indices(msgs: list) -> list[int]:
    return [i for i, m in enumerate(msgs) if _is_marker(m)]


class TestRecentHistoryChannelMarkers:
    """Standalone channel/server separator markers in cross-channel history."""

    @staticmethod
    def _turn(
        store: HistoryStore, *, channel_id: int, content: str, msg_id: str
    ) -> None:
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user",
            content=content,
            author=alice,
            platform_message_id=msg_id,
        )

    @pytest.mark.asyncio
    async def test_single_channel_emits_no_markers(self) -> None:
        """All turns in one channel ⇒ zero separator markers."""
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="one", msg_id="m1")
        self._turn(store, channel_id=1, content="two", msg_id="m2")
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        assert _markers(msgs) == []
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_multi_channel_leading_and_change_markers(self) -> None:
        """Channels [A, A, B, A] ⇒ markers at first A, before B, before return."""
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="a one", msg_id="m1")
        self._turn(store, channel_id=1, content="a two", msg_id="m2")
        self._turn(store, channel_id=2, content="b one", msg_id="m3")
        self._turn(store, channel_id=1, content="a three", msg_id="m4")
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        # 7 messages: 3 markers + 4 turns; no marker between the two A turns.
        assert _marker_indices(msgs) == [0, 3, 5]
        assert len(msgs) == 7
        assert msgs[1].content_str.endswith("a one")
        assert msgs[2].content_str.endswith("a two")
        assert msgs[4].content_str.endswith("b one")
        assert msgs[6].content_str.endswith("a three")

    @pytest.mark.asyncio
    async def test_marker_resolves_channel_and_server_names(self) -> None:
        """Resolvers present ⇒ marker names the channel and server."""
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="hi", msg_id="m1")
        self._turn(store, channel_id=2, content="yo", msg_id="m2")
        layer = RecentHistoryLayer(
            store=AsyncHistoryStore(store),
            window_size=20,
            channel_name_resolver={1: "general", 2: "random"}.get,
            guild_name_resolver={1: "My Server", 2: "My Server"}.get,
        )
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        assert "My Server/#general" in _markers(msgs)

    @pytest.mark.asyncio
    async def test_marker_falls_back_to_channel_id_without_resolvers(self) -> None:
        """No resolvers ⇒ marker uses #<channel_id> and omits server clause."""
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="hi", msg_id="m1")
        self._turn(store, channel_id=2, content="yo", msg_id="m2")
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        markers = _markers(msgs)
        assert "#1" in markers
        assert "#2" in markers

    @pytest.mark.asyncio
    async def test_marker_omits_server_when_guild_unknown(self) -> None:
        """Channel name known but no guild ⇒ marker omits the server clause."""
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="hi", msg_id="m1")
        self._turn(store, channel_id=2, content="yo", msg_id="m2")
        layer = RecentHistoryLayer(
            store=AsyncHistoryStore(store),
            window_size=20,
            channel_name_resolver={1: "general", 2: "random"}.get,
            guild_name_resolver=lambda _cid: None,
        )
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        markers = _markers(msgs)
        assert "#general" in markers
        assert all("/" not in m for m in markers)

    @pytest.mark.asyncio
    async def test_marker_is_distinct_user_message_without_name(self) -> None:
        """A marker is its own role=user message carrying no name field."""
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="hi", msg_id="m1")
        self._turn(store, channel_id=2, content="yo", msg_id="m2")
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        marker_msg = next(m for m in msgs if _is_marker(m))
        assert marker_msg.role == "user"
        assert marker_msg.name is None

    @staticmethod
    async def _per_turn_cost(store: HistoryStore) -> int:
        """Token cost of one rendered turn (all seeded turns are equal-cost)."""
        probe = await RecentHistoryLayer(
            store=AsyncHistoryStore(store), window_size=20
        ).recent_messages(_ctx(channel_id=1))
        return next(estimate_message_tokens(m) for m in probe if not _is_marker(m))

    @pytest.mark.asyncio
    async def test_token_trim_realigns_markers_to_surviving_window(self) -> None:
        """Markers track the post-trim tail, not the pre-trim head.

        Channels [9, 9, 1, 2] with a cap that keeps only the last 2
        turns ([1, 2]). The surviving window must emit a leading anchor
        + an A→B change marker naming the *surviving* channels. A
        head-slice (``turns[:len(rendered)]``) would pair the dropped
        [9, 9] head with the kept messages — a single channel, zero
        markers — so this locks the tail-slice realignment.
        """
        store = HistoryStore(":memory:")
        # Equal-cost turns (identical body/author; same-width ids).
        self._turn(store, channel_id=9, content="same body", msg_id="m1")
        self._turn(store, channel_id=9, content="same body", msg_id="m2")
        self._turn(store, channel_id=1, content="same body", msg_id="m3")
        self._turn(store, channel_id=2, content="same body", msg_id="m4")
        cost = await self._per_turn_cost(store)
        layer = RecentHistoryLayer(
            store=AsyncHistoryStore(store),
            window_size=20,
            max_tokens=2 * cost,  # keeps exactly the last 2 turns
        )
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        # 2 surviving turns + 2 markers; markers name the survivors (#1, #2),
        # never the trimmed #9 head.
        assert len(msgs) == 4
        assert _marker_indices(msgs) == [0, 2]
        assert _markers(msgs) == ["#1", "#2"]
        assert "#1" in msgs[1].content_str
        assert "#2" in msgs[3].content_str

    @pytest.mark.asyncio
    async def test_multi_before_trim_single_after_trim_emits_no_markers(self) -> None:
        """Multi-channel detection runs on post-trim survivors only.

        Channels [1, 1, 2, 2] but the cap keeps only the last 2 ([2, 2]):
        a single-channel surviving window ⇒ zero markers, even though the
        pre-trim window spanned two channels.
        """
        store = HistoryStore(":memory:")
        self._turn(store, channel_id=1, content="same body", msg_id="m1")
        self._turn(store, channel_id=1, content="same body", msg_id="m2")
        self._turn(store, channel_id=2, content="same body", msg_id="m3")
        self._turn(store, channel_id=2, content="same body", msg_id="m4")
        cost = await self._per_turn_cost(store)
        layer = RecentHistoryLayer(
            store=AsyncHistoryStore(store),
            window_size=20,
            max_tokens=2 * cost,  # keeps exactly the last 2 turns ([2, 2])
        )
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        assert _markers(msgs) == []
        assert len(msgs) == 2


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
            unread_digest={10: (3, 0), 20: (1, 0)},
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
            unread_digest={10: (5, 0), 20: (0, 0), 30: (2, 0)},
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
            unread_digest={5: (2, 0)},
            post_history_instructions="TAIL_MARKER",
        )
        assert out.index("new message") < out.index("TAIL_MARKER")

    def test_focus_and_unread_both_present(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            focus_channel_id=3,
            unread_digest={10: (4, 0)},
        )
        assert "attention is currently on #3" in out
        assert "#10 (4)" in out

    def test_ping_subset_with_higher_unread_count(self) -> None:
        """Mixed channel renders ``(unread, N ping)``."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: (3, 1)},
        )
        assert "#10 (3, 1 ping)" in out
        assert "shift_focus" in out

    def test_all_unreads_are_pings_singular(self) -> None:
        """When every unread is a ping, count isn't repeated."""
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: (1, 1)},
        )
        assert "#10 (1 ping)" in out

    def test_mixed_unread_with_multiple_pings_plural(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: (3, 2)},
        )
        assert "#10 (3, 2 pings)" in out

    def test_no_pings_renders_count_only(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: (2, 0)},
        )
        assert "#10 (2)" in out

    def test_single_unread_no_ping_has_no_suffix(self) -> None:
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={10: (1, 0)},
        )
        assert "#10 —" in out
        assert "#10 (" not in out

    def test_named_unread_channel_surfaces_numeric_id(self) -> None:
        """Named unread channel still exposes its id so shift_focus can target it.

        Digest renders #name; without the numeric id in context the model
        can't pass a valid channel_id to shift_focus (hallucinates → guard
        bounce → tool-loop). Id must ride alongside the name.
        """
        out = build_final_reminder(
            viewer_mode="text",
            now=_at(2026, 5, 4, 14, 30),
            unread_digest={422137955130408970: (2, 0)},
            channel_names={422137955130408970: "the-annex"},
        )
        assert "#the-annex" in out
        assert "422137955130408970" in out  # id available for shift_focus
        assert "shift_focus" in out
