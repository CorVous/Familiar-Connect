"""Tests for :class:`familiar_connect.processors.text_responder.TextResponder`.

End-to-end text turn: ``discord.text`` event in → assemble prompt →
stream LLM → call ``send_text(channel_id, reply)`` → append assistant
turn to history. Mirrors :class:`VoiceResponder` but without TTS.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

import asyncio

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.context import (
    Assembler,
    CoreInstructionsLayer,
    RecentHistoryLayer,
)
from familiar_connect.context.layers import _turn_to_message_with_context
from familiar_connect.history.store import HistoryStore, HistoryTurn
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.text_responder import (
    TextResponder,
    _strip_leaked_metadata_prefix,
)


class _ScriptedLLM(LLMClient):
    """Yield a fixed delta sequence; optional per-delta delay."""

    def __init__(self, *, deltas: list[str], delay_ms: int = 0) -> None:
        super().__init__(api_key="k", model="m")
        self._deltas = list(deltas)
        self._delay_ms = delay_ms

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        return Message(role="assistant", content="".join(self._deltas))

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],  # noqa: ARG002
    ) -> AsyncIterator[str]:
        for d in self._deltas:
            if self._delay_ms:
                await asyncio.sleep(self._delay_ms / 1000.0)
            yield d


class _CapturingSend:
    """Records ``send_text`` invocations for assertions.

    Matches the ``SendText`` callback signature: returns a fake
    platform message id so the responder can persist it on the
    assistant turn.
    """

    def __init__(self, *, returned_id: str | None = "bot-msg-1") -> None:
        self.calls: list[tuple[int, str, str | None, tuple[int, ...]]] = []
        self._returned_id = returned_id

    async def __call__(
        self,
        channel_id: int,
        content: str,
        reply_to_message_id: str | None = None,
        mention_user_ids: tuple[int, ...] = (),
    ) -> str | None:
        self.calls.append((channel_id, content, reply_to_message_id, mention_user_ids))
        return self._returned_id


def _discord_text_event(
    *,
    event_id: str = "e-1",
    channel_id: int = 42,
    content: str = "hi there",
    seq: int = 1,
) -> Event:
    return Event(
        event_id=event_id,
        turn_id=event_id,
        session_id=f"discord:{channel_id}",
        parent_event_ids=(),
        topic=TOPIC_DISCORD_TEXT,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload={
            "familiar_id": "fam",
            "channel_id": channel_id,
            "guild_id": 99,
            "author": Author(
                platform="discord",
                user_id="1",
                username="alice",
                display_name="Alice",
            ),
            "content": content,
        },
    )


def _make_responder(
    *,
    llm: LLMClient,
    send: _CapturingSend,
    tmp_path: Path,
    router: TurnRouter | None = None,
    store: HistoryStore | None = None,
) -> tuple[TextResponder, TurnRouter, HistoryStore]:
    core = tmp_path / "core.md"
    core.write_text("You are a familiar.\n")
    store = store or HistoryStore(":memory:")
    assembler = Assembler(
        layers=[
            CoreInstructionsLayer(path=core),
            RecentHistoryLayer(store=store, window_size=20),
        ]
    )
    router = router or TurnRouter()
    responder = TextResponder(
        assembler=assembler,
        llm_client=llm,
        send_text=send,
        history_store=store,
        router=router,
        familiar_id="fam",
    )
    return responder, router, store


class TestProcessorSurface:
    def test_topic_is_discord_text(self, tmp_path: Path) -> None:
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=_ScriptedLLM(deltas=["x"]),
            send=send,
            tmp_path=tmp_path,
        )
        assert responder.name == "text-responder"
        assert TOPIC_DISCORD_TEXT in responder.topics


class TestReply:
    @pytest.mark.asyncio
    async def test_streams_reply_and_calls_send_text(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_discord_text_event(content="hi there"), bus)
        await bus.shutdown()

        assert send.calls == [(42, "Hello, world.", None, ())]
        # assistant turn appended (user-turn write is HistoryWriter's job)
        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        roles_contents = [(t.role, t.content) for t in turns]
        assert ("assistant", "Hello, world.") in roles_contents

    @pytest.mark.asyncio
    async def test_skips_when_payload_missing_content(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["nope"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content=""), bus)
        await bus.shutdown()
        assert send.calls == []

    @pytest.mark.asyncio
    async def test_skips_other_familiars(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["nope"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        ev = _discord_text_event()
        payload = dict(ev.payload)
        payload["familiar_id"] = "other"
        other = Event(
            event_id="other",
            turn_id="other",
            session_id=ev.session_id,
            parent_event_ids=(),
            topic=ev.topic,
            timestamp=ev.timestamp,
            sequence_number=ev.sequence_number,
            payload=payload,
        )
        await responder.handle(other, bus)
        await bus.shutdown()
        assert send.calls == []

    @pytest.mark.asyncio
    async def test_begins_turn_for_session(self, tmp_path: Path) -> None:
        """Router should track an active scope for the discord session."""
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, router, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()
        # turn ended cleanly — no active scope after completion
        assert router.active_scope("discord:42") is None
        assert send.calls

    @pytest.mark.asyncio
    async def test_skips_when_llm_returns_empty(self, tmp_path: Path) -> None:
        """Discord rejects empty messages; bail before calling send_text."""
        llm = _ScriptedLLM(deltas=[])  # zero deltas → empty reply
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()
        assert send.calls == []
        # no assistant turn either — nothing to record
        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        assert all(t.role != "assistant" for t in turns)

    @pytest.mark.asyncio
    async def test_skips_when_llm_returns_whitespace(self, tmp_path: Path) -> None:
        """Whitespace-only replies also fail Discord; bail."""
        llm = _ScriptedLLM(deltas=["   ", "\n"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()
        assert send.calls == []

    @pytest.mark.asyncio
    async def test_user_turn_persisted_before_llm_stream(self, tmp_path: Path) -> None:
        """User turn must be in history *before* the LLM stream is read.

        Otherwise ``RecentHistoryLayer`` would miss the just-arrived
        message and the LLM would respond to a stale conversation.
        """
        store = HistoryStore(":memory:")
        observed_user_turns: list[str] = []

        class _ObservingLLM(LLMClient):
            def __init__(self) -> None:
                super().__init__(api_key="k", model="m")

            async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
                return Message(role="assistant", content="ack")

            async def chat_stream(  # type: ignore[override]
                self,
                messages: list[Message],  # noqa: ARG002
            ) -> AsyncIterator[str]:
                # snapshot history at the moment streaming begins
                turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
                observed_user_turns.extend(t.content for t in turns if t.role == "user")
                yield "ack"

        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=_ObservingLLM(), send=send, tmp_path=tmp_path, store=store
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hello"), bus)
        await bus.shutdown()

        assert "hello" in observed_user_turns, observed_user_turns

    @pytest.mark.asyncio
    async def test_silent_sentinel_skips_send_and_assistant_turn(
        self, tmp_path: Path
    ) -> None:
        """``<silent>`` reply gates the response.

        No Discord post, no assistant turn appended. User turn is
        still recorded.
        """
        llm = _ScriptedLLM(deltas=["<silent>"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi nobody"), bus)
        await bus.shutdown()

        assert send.calls == []
        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        assert all(t.role != "assistant" for t in turns)
        # user turn is still persisted (we observed the message)
        assert any(t.role == "user" and "hi nobody" in t.content for t in turns)

    @pytest.mark.asyncio
    async def test_silent_sentinel_with_leading_whitespace(
        self, tmp_path: Path
    ) -> None:
        """Leading whitespace before the sentinel is tolerated."""
        llm = _ScriptedLLM(deltas=["  ", "<silent>"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()
        assert send.calls == []

    @pytest.mark.asyncio
    async def test_sentinel_mid_reply_is_not_a_gate(self, tmp_path: Path) -> None:
        """A ``<silent>`` token after content is treated as content."""
        llm = _ScriptedLLM(deltas=["Sure! ", "<silent>", " — kidding."])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()
        assert send.calls == [(42, "Sure! <silent> — kidding.", None, ())]

    @pytest.mark.asyncio
    async def test_user_turn_recorded_with_author_and_guild(
        self, tmp_path: Path
    ) -> None:
        """Author + guild_id from the event payload reach ``HistoryStore``."""
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()

        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        user_turns = [t for t in turns if t.role == "user"]
        assert len(user_turns) == 1
        assert user_turns[0].content == "hi"
        assert user_turns[0].author is not None
        assert user_turns[0].author.display_name == "Alice"


def _discord_text_event_full(
    *,
    event_id: str = "e-1",
    channel_id: int = 42,
    guild_id: int | None = 99,
    content: str = "hi @bob",
    message_id: str = "msg-001",
    reply_to_message_id: str | None = None,
    mentions: tuple[Author, ...] = (),
    seq: int = 1,
) -> Event:
    """Like ``_discord_text_event`` but carries the new identity fields."""
    return Event(
        event_id=event_id,
        turn_id=event_id,
        session_id=f"discord:{channel_id}",
        parent_event_ids=(),
        topic=TOPIC_DISCORD_TEXT,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload={
            "familiar_id": "fam",
            "channel_id": channel_id,
            "guild_id": guild_id,
            "author": Author(
                platform="discord",
                user_id="1",
                username="alice",
                display_name="Alice",
                global_name="Alice Liddell",
                guild_nick="Aria",
            ),
            "content": content,
            "message_id": message_id,
            "reply_to_message_id": reply_to_message_id,
            "mentions": mentions,
        },
    )


class TestIdentityWritePath:
    """User-turn ingestion upserts accounts/guild_nicks and records mentions."""

    @pytest.mark.asyncio
    async def test_persists_platform_message_id(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(message_id="msg-9999"), bus)
        await bus.shutdown()

        looked = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="msg-9999"
        )
        assert looked is not None
        assert looked.role == "user"

    @pytest.mark.asyncio
    async def test_persists_reply_to_message_id(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(
                message_id="msg-2",
                reply_to_message_id="msg-1",
            ),
            bus,
        )
        await bus.shutdown()

        row = store._conn.execute(
            "SELECT reply_to_message_id FROM turns WHERE platform_message_id = ?",
            ("msg-2",),
        ).fetchone()
        assert row["reply_to_message_id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_upserts_author_into_accounts(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(), bus)
        await bus.shutdown()

        row = store._conn.execute(
            "SELECT canonical_key, global_name FROM accounts WHERE canonical_key = ?",
            ("discord:1",),
        ).fetchone()
        assert row is not None
        assert row["global_name"] == "Alice Liddell"

    @pytest.mark.asyncio
    async def test_upserts_author_guild_nick(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(guild_id=99), bus)
        await bus.shutdown()

        row = store._conn.execute(
            "SELECT nick FROM account_guild_nicks "
            "WHERE canonical_key = ? AND guild_id = ?",
            ("discord:1", 99),
        ).fetchone()
        assert row is not None
        assert row["nick"] == "Aria"

    @pytest.mark.asyncio
    async def test_upserts_each_mentioned_user(self, tmp_path: Path) -> None:
        bob = Author(
            platform="discord",
            user_id="2",
            username="bob",
            display_name="Bob",
            global_name="Robert",
        )
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(mentions=(bob,)),
            bus,
        )
        await bus.shutdown()

        row = store._conn.execute(
            "SELECT global_name FROM accounts WHERE canonical_key = ?",
            ("discord:2",),
        ).fetchone()
        assert row is not None
        assert row["global_name"] == "Robert"

    @pytest.mark.asyncio
    async def test_does_not_thread_by_default(self, tmp_path: Path) -> None:
        """No marker → no threading. Default is a normal post."""
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(message_id="incoming-msg-77"),
            bus,
        )
        await bus.shutdown()

        assert len(send.calls) == 1
        _, _, reply_to, _ = send.calls[0]
        assert reply_to is None

    @pytest.mark.asyncio
    async def test_threads_when_llm_emits_thread_marker(self, tmp_path: Path) -> None:
        """LLM emits ``[↩]`` → thread to the triggering message id."""
        llm = _ScriptedLLM(deltas=["[↩] sure thing"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(message_id="incoming-msg-77"),
            bus,
        )
        await bus.shutdown()

        _, content, reply_to, _ = send.calls[0]
        assert reply_to == "incoming-msg-77"
        # Marker stripped before posting.
        assert "[↩]" not in content
        assert content == "sure thing"

    @pytest.mark.asyncio
    async def test_thread_marker_with_explicit_id_targets_that_message(
        self, tmp_path: Path
    ) -> None:
        """``[↩ <id>]`` threads to that specific message when known."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=42,
            role="user",
            content="earlier",
            author=None,
            platform_message_id="older-msg-7",
        )
        llm = _ScriptedLLM(deltas=["[↩ older-msg-7] sure"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=llm, send=send, tmp_path=tmp_path, store=store
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(message_id="incoming-msg-99"),
            bus,
        )
        await bus.shutdown()

        _, content, reply_to, _ = send.calls[0]
        assert reply_to == "older-msg-7"
        assert "[↩" not in content
        assert content == "sure"

    @pytest.mark.asyncio
    async def test_thread_marker_with_hash_sigil_is_resolved(
        self, tmp_path: Path
    ) -> None:
        """``[↩ #<id>]`` strips the sigil before lookup.

        Recent history surfaces ids as ``#<id>``; models echo the
        sigil back inside the marker. Stripping it keeps the lookup
        from missing and silently falling back to the triggering
        message.
        """
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=42,
            role="user",
            content="earlier",
            author=None,
            platform_message_id="1500691573262782604",
        )
        llm = _ScriptedLLM(deltas=["[↩ #1500691573262782604] still thinking"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=llm, send=send, tmp_path=tmp_path, store=store
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(message_id="trigger-msg"),
            bus,
        )
        await bus.shutdown()

        _, content, reply_to, _ = send.calls[0]
        assert reply_to == "1500691573262782604"
        assert content == "still thinking"

    @pytest.mark.asyncio
    async def test_thread_marker_with_unknown_id_falls_back_to_triggering(
        self, tmp_path: Path
    ) -> None:
        """Unknown id → fall back to the triggering message id."""
        llm = _ScriptedLLM(deltas=["[↩ no-such-msg] ok"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(message_id="incoming-msg-77"),
            bus,
        )
        await bus.shutdown()

        _, content, reply_to, _ = send.calls[0]
        assert reply_to == "incoming-msg-77"
        assert content == "ok"

    @pytest.mark.asyncio
    async def test_thread_marker_reply_word_form_also_works(
        self, tmp_path: Path
    ) -> None:
        """``[reply]`` is the tokenizer-safe alias for ``[↩]``."""
        llm = _ScriptedLLM(deltas=["[reply] got it"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(message_id="msg-99"), bus)
        await bus.shutdown()

        _, content, reply_to, _ = send.calls[0]
        assert reply_to == "msg-99"
        assert "[reply]" not in content
        assert content == "got it"

    @pytest.mark.asyncio
    async def test_rewrites_llm_ping_marker_to_discord_mention(
        self, tmp_path: Path
    ) -> None:
        """``[@DisplayName]`` from the LLM becomes ``<@user_id>`` on Discord."""
        # Pre-seed the participant: bob exists in the channel.
        store = HistoryStore(":memory:")
        bob = Author(
            platform="discord",
            user_id="222",
            username="bob",
            display_name="Bob",
            global_name="Bob",
        )
        store.upsert_account(bob)
        store.append_turn(
            familiar_id="fam",
            channel_id=42,
            role="user",
            content="hi",
            author=bob,
            guild_id=99,
        )
        llm = _ScriptedLLM(deltas=["sure thing, [@Bob]"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=llm, send=send, tmp_path=tmp_path, store=store
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(content="hi bob"), bus)
        await bus.shutdown()

        _, content, _, mention_user_ids = send.calls[0]
        assert "<@222>" in content
        assert "[@Bob]" not in content  # marker was consumed
        assert 222 in mention_user_ids

    @pytest.mark.asyncio
    async def test_unknown_ping_marker_renders_as_plain_text(
        self, tmp_path: Path
    ) -> None:
        """Unknown ``[@X]`` markers degrade to plain ``@X`` (no ping)."""
        llm = _ScriptedLLM(deltas=["hi [@Nobody], welcome"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(), bus)
        await bus.shutdown()

        _, content, _, mention_user_ids = send.calls[0]
        assert "<@" not in content
        assert "@Nobody" in content  # rendered as plain @-prefix
        assert mention_user_ids == ()

    @pytest.mark.asyncio
    async def test_assistant_turn_persists_platform_message_id(
        self, tmp_path: Path
    ) -> None:
        """Assistant turn carries the Discord message id returned by send."""
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend(returned_id="bot-msg-12345")
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(message_id="user-msg-1"), bus)
        await bus.shutdown()

        looked = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="bot-msg-12345"
        )
        assert looked is not None
        assert looked.role == "assistant"
        # Default is non-threaded; reply_to_message_id only persists
        # when the LLM opted into threading via ``[↩]``.
        assert looked.reply_to_message_id is None

    @pytest.mark.asyncio
    async def test_assistant_turn_records_thread_target_when_threaded(
        self, tmp_path: Path
    ) -> None:
        """When the LLM threads, the assistant turn records the target id."""
        llm = _ScriptedLLM(deltas=["[↩] yep"])
        send = _CapturingSend(returned_id="bot-msg-67890")
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event_full(message_id="user-msg-1"), bus)
        await bus.shutdown()

        looked = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="bot-msg-67890"
        )
        assert looked is not None
        assert looked.reply_to_message_id == "user-msg-1"

    @pytest.mark.asyncio
    async def test_records_turn_mentions_for_each_pinged_user(
        self, tmp_path: Path
    ) -> None:
        bob = Author(
            platform="discord",
            user_id="2",
            username="bob",
            display_name="Bob",
        )
        carol = Author(
            platform="discord",
            user_id="3",
            username="carol",
            display_name="Carol",
        )
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(
            _discord_text_event_full(mentions=(bob, carol)),
            bus,
        )
        await bus.shutdown()

        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        user_turn = next(t for t in turns if t.role == "user")
        keys = store.mentions_for_turn(turn_id=user_turn.id)
        assert "discord:2" in keys
        assert "discord:3" in keys


class TestStripLeakedMetadataPrefix:
    """Defensively drop ``[#id] / [H:MMpm]``-shaped prefixes the LLM may echo."""

    def test_strips_leading_id_clump(self) -> None:
        leaked = "[#1500709436557445449] 4:03AM UTC. I actually know that now!"
        assert (
            _strip_leaked_metadata_prefix(leaked)
            == "4:03AM UTC. I actually know that now!"
        )

    def test_strips_leading_time_clump(self) -> None:
        assert _strip_leaked_metadata_prefix("[4:03AM] hello") == "hello"

    def test_strips_chained_metadata_clumps(self) -> None:
        leaked = "[14:32 Alice #abc] [↩ msg-1] hi"
        # only the metadata clump is dropped; the reply marker survives
        # for ``_consume_thread_marker`` to handle separately.
        assert _strip_leaked_metadata_prefix(leaked).startswith("[↩ msg-1] hi")

    def test_leaves_legitimate_bracketed_text_alone(self) -> None:
        assert _strip_leaked_metadata_prefix("[note] heads up") == "[note] heads up"

    def test_passes_through_clean_text(self) -> None:
        assert _strip_leaked_metadata_prefix("just a normal reply") == (
            "just a normal reply"
        )


class TestAssistantTurnRendering:
    """Assistant past turns must not surface ``[#id]`` (mimicry vector)."""

    def test_assistant_message_has_no_id_prefix(self) -> None:
        turn = HistoryTurn(
            id=1,
            timestamp=datetime(2026, 5, 4, 4, 3, tzinfo=UTC),
            role="assistant",
            author=None,
            content="hello world",
            channel_id=42,
            platform_message_id="1500709436557445449",
        )
        store = HistoryStore(":memory:")
        msg = _turn_to_message_with_context(
            turn=turn,
            store=store,
            familiar_id="fam",
            guild_id=None,
            in_window_msg_ids=set(),
        )
        assert msg.content == "hello world"
