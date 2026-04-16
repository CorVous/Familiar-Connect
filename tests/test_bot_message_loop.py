"""End-to-end tests for the rewritten bot message loop.

Exercises the new :class:`Familiar`-bundle-driven bot. Each test
constructs a real Familiar against a ``tmp_path`` skeleton and
fakes just the Discord objects (messages, application contexts,
voice clients) that the production :mod:`familiar_connect.bot`
module touches directly. The LLM client, side model, TTS client,
and all Discord surfaces are mocks.

Covers:

- ``on_message`` routes through the pipeline, post-processors,
  and the history store.
- ``on_message`` ignores bot messages and unsubscribed channels.
- ``/subscribe-text`` / ``/unsubscribe-text`` mutate the registry
  and persist to disk.
- ``/channel-*`` commands flip the channel mode.
- ``create_bot`` takes a Familiar bundle and registers every
  slash command.
"""

from __future__ import annotations

import asyncio
import re
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, PropertyMock, patch

import discord
import httpx
import pytest

from familiar_connect.bot import (
    _default_backdrop_placeholder,
    _make_backdrop_modal,
    _run_text_response,
    _run_voice_response,
    channel_backdrop,
    create_bot,
    dispatch_interruption_regen,
    on_message,
    set_channel_mode,
    show_context,
    subscribe_my_voice,
    subscribe_text,
    unsubscribe_text,
    unsubscribe_voice,
)
from familiar_connect.chattiness import BufferedMessage, ResponseTrigger
from familiar_connect.config import LLM_SLOT_NAMES, ChannelMode
from familiar_connect.familiar import Familiar
from familiar_connect.llm import LLMClient, Message
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.transcription import TranscriptionResult
from familiar_connect.tts import TTSResult, WordTimestamp
from familiar_connect.voice.interruption import (
    InterruptionClass,
    InterruptionDetector,
    ResponseState,
    ResponseTracker,
    ResponseTrackerRegistry,
)
from familiar_connect.voice_lull import VoiceLullMonitor

if TYPE_CHECKING:
    from collections.abc import Iterator

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROFILE_PATH = (
    _REPO_ROOT / "data" / "familiars" / "_default" / "character.toml"
)


@pytest.fixture(autouse=True)
def _fresh_loop() -> Iterator[None]:
    """Ensure every test has a current event loop.

    py-cord's :class:`discord.Bot` and ``asyncio.run`` both probe
    ``asyncio.get_event_loop`` at construction/invocation time; pytest
    doesn't create one by default under Python 3.13.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubLLMClient(LLMClient):
    """LLMClient subclass that records calls and returns a canned reply."""

    def __init__(self, reply: str = "I am here.") -> None:
        super().__init__(api_key="test-key", model="stub/test-model")
        self.reply = reply
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(messages)
        return Message(role="assistant", content=self.reply)


class _RaisingLLMClient(LLMClient):
    """LLMClient subclass whose ``chat`` always raises a configured exception.

    Used by the main-reply resilience tests to simulate a failing
    OpenRouter call without touching the network.
    """

    def __init__(self, exc: Exception) -> None:
        super().__init__(api_key="test-key", model="stub/test-model")
        self._exc = exc
        self.call_count = 0

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        self.call_count += 1
        raise self._exc


class _SlowLLMClient(LLMClient):
    """LLMClient whose ``chat`` blocks until ``release`` is set.

    Used by the cancellable-generation tests to observe the tracker
    while the LLM call is mid-flight and to exercise the cancel path
    without racing against the stub's return.
    """

    def __init__(self, reply: str = "I am here.") -> None:
        super().__init__(api_key="test-key", model="stub/test-model")
        self.reply = reply
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.was_cancelled = False

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.was_cancelled = True
            raise
        return Message(role="assistant", content=self.reply)


def _make_llm_clients(reply: str = "I am here.") -> dict[str, LLMClient]:
    """Return a ``slot_name -> LLMClient`` dict with a stub in every slot.

    All slots share the same canned ``reply`` so a test that cares
    about the end-to-end reply text can set it once and have the
    recast post-processor (which would otherwise overwrite the
    main-prose reply with its own stub response) return the same
    text. Tests that care about a specific call site's behaviour
    can still reach in through ``familiar.llm_clients[slot]``.
    """
    return {slot: _StubLLMClient(reply=reply) for slot in LLM_SLOT_NAMES}


def _make_familiar(tmp_path: Path, *, reply: str = "I am here.") -> Familiar:
    root = tmp_path / "aria"
    root.mkdir()
    return Familiar.load_from_disk(
        root,
        llm_clients=_make_llm_clients(reply=reply),
        defaults_path=_DEFAULT_PROFILE_PATH,
    )


def _make_message(
    *,
    content: str = "hello",
    channel_id: int = 12345,
    author_bot: bool = False,
    in_guild: bool = True,
    guild_id: int = 999,
) -> MagicMock:
    msg = MagicMock(spec=discord.Message)
    msg.content = content

    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.send = AsyncMock()
    typing_cm = MagicMock()
    typing_cm.__aenter__ = AsyncMock(return_value=None)
    typing_cm.__aexit__ = AsyncMock(return_value=False)
    channel.typing = MagicMock(return_value=typing_cm)
    msg.channel = channel

    author = MagicMock(spec=discord.Member)
    author.bot = author_bot
    author.display_name = "Alice"
    msg.author = author

    if in_guild:
        guild = MagicMock(spec=discord.Guild)
        guild.id = guild_id
        guild.voice_client = None
        msg.guild = guild
    else:
        msg.guild = None

    return msg


def _make_text_ctx(
    *,
    channel_id: int = 12345,
    guild_id: int = 999,
    view_channel: bool = True,
    send_messages: bool = True,
) -> MagicMock:
    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.respond = AsyncMock()
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.name = "general"

    # Wire up permissions_for so permission checks work.
    perms = MagicMock(spec=discord.Permissions)
    perms.view_channel = view_channel
    perms.send_messages = send_messages
    channel.permissions_for = MagicMock(return_value=perms)

    type(ctx).channel = PropertyMock(return_value=channel)
    type(ctx).channel_id = PropertyMock(return_value=channel_id)
    if guild_id is not None:
        guild = MagicMock(spec=discord.Guild)
        guild.id = guild_id
        bot_member = MagicMock(spec=discord.Member)
        type(guild).me = PropertyMock(return_value=bot_member)
        type(ctx).guild_id = PropertyMock(return_value=guild_id)
        type(ctx).guild = PropertyMock(return_value=guild)
    else:
        type(ctx).guild_id = PropertyMock(return_value=None)
        type(ctx).guild = PropertyMock(return_value=None)
    return ctx


def _make_thread_ctx(
    *,
    channel_id: int = 12345,
    parent_id: int = 54321,
    parent_name: str = "general",
    guild_id: int = 999,
    view_channel: bool = True,
    send_messages_in_threads: bool = True,
    parent_type: type = discord.TextChannel,
) -> MagicMock:
    """Build a fake ``discord.ApplicationContext`` whose channel is a Thread.

    ``parent_type`` switches between ``discord.TextChannel`` (regular
    channel thread) and ``discord.ForumChannel`` (forum post).
    """
    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.respond = AsyncMock()
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()

    parent = MagicMock(spec=parent_type)
    parent.id = parent_id
    parent.name = parent_name

    thread = MagicMock(spec=discord.Thread)
    thread.id = channel_id
    thread.name = "feature-brainstorm"
    thread.parent = parent
    thread.parent_id = parent_id
    thread.join = AsyncMock()

    perms = MagicMock(spec=discord.Permissions)
    perms.view_channel = view_channel
    perms.send_messages_in_threads = send_messages_in_threads
    perms.send_messages = True  # irrelevant for threads
    thread.permissions_for = MagicMock(return_value=perms)

    type(ctx).channel = PropertyMock(return_value=thread)
    type(ctx).channel_id = PropertyMock(return_value=channel_id)

    guild = MagicMock(spec=discord.Guild)
    guild.id = guild_id
    bot_member = MagicMock(spec=discord.Member)
    type(guild).me = PropertyMock(return_value=bot_member)
    type(ctx).guild_id = PropertyMock(return_value=guild_id)
    type(ctx).guild = PropertyMock(return_value=guild)
    return ctx


def _make_voice_ctx(
    *,
    channel_id: int = 9000,
    guild_id: int = 999,
    already_connected: bool = False,
    connect: bool = True,
    speak: bool = True,
) -> MagicMock:
    ctx = _make_text_ctx(channel_id=channel_id, guild_id=guild_id)
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()

    author = MagicMock(spec=discord.Member)
    type(ctx).author = PropertyMock(return_value=author)

    voice_state = MagicMock()
    voice_channel = MagicMock(spec=discord.VoiceChannel)
    voice_channel.id = channel_id
    voice_channel.name = "Voice"
    mock_vc = MagicMock(spec=discord.VoiceClient)
    mock_vc.play = MagicMock()
    mock_vc.is_playing = MagicMock(return_value=False)
    mock_vc.disconnect = AsyncMock()
    voice_channel.connect = AsyncMock(return_value=mock_vc)

    # Wire up voice channel permissions.
    voice_perms = MagicMock(spec=discord.Permissions)
    voice_perms.connect = connect
    voice_perms.speak = speak
    voice_channel.permissions_for = MagicMock(return_value=voice_perms)

    voice_state.channel = voice_channel
    type(author).voice = PropertyMock(return_value=voice_state)

    if already_connected:
        type(ctx).voice_client = PropertyMock(return_value=mock_vc)
    else:
        type(ctx).voice_client = PropertyMock(return_value=None)

    return ctx


# ---------------------------------------------------------------------------
# on_message — routes to monitor
# ---------------------------------------------------------------------------


def _make_channel(channel_id: int = 12345) -> MagicMock:
    """Build a minimal fake discord.TextChannel for _run_text_response tests."""
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.send = AsyncMock()
    typing_cm = MagicMock()
    typing_cm.__aenter__ = AsyncMock(return_value=None)
    typing_cm.__aexit__ = AsyncMock(return_value=False)
    channel.typing = MagicMock(return_value=typing_cm)
    return channel


class TestOnMessageMonitorRouting:
    """on_message now delegates to familiar.monitor.on_message."""

    def test_message_outside_subscription_is_ignored(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        mock = AsyncMock()
        familiar.monitor.on_message = mock  # ty: ignore[invalid-assignment]
        msg = _make_message()
        asyncio.run(on_message(msg, familiar))
        mock.assert_not_called()

    def test_bot_messages_are_ignored(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        mock = AsyncMock()
        familiar.monitor.on_message = mock  # ty: ignore[invalid-assignment]
        msg = _make_message(author_bot=True)
        asyncio.run(on_message(msg, familiar))
        mock.assert_not_called()

    def test_subscribed_message_calls_monitor(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        mock = AsyncMock()
        familiar.monitor.on_message = mock  # ty: ignore[invalid-assignment]
        msg = _make_message(content="hi", channel_id=12345)
        asyncio.run(on_message(msg, familiar))
        mock.assert_called_once()
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["channel_id"] == 12345
        assert call_kwargs["text"] == "hi"
        assert call_kwargs["speaker"] == "Alice"

    def test_on_message_passes_is_mention(self, tmp_path: Path) -> None:
        """Bot @mention is passed through to the monitor."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        mock = AsyncMock()
        familiar.monitor.on_message = mock  # ty: ignore[invalid-assignment]
        msg = _make_message(content="hey there", channel_id=12345)
        # Simulate a mention by adding bot_user to extras and message.mentions
        bot_user = MagicMock()
        familiar.extras["bot_user"] = bot_user
        msg.mentions = [bot_user]
        msg.guild = MagicMock()
        msg.guild.id = 999
        asyncio.run(on_message(msg, familiar))
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["is_mention"] is True


# ---------------------------------------------------------------------------
# _run_text_response — full pipeline path
# ---------------------------------------------------------------------------


class TestOnRespond:
    """_run_text_response exercises the full pipeline + LLM + reply path."""

    def test_subscribed_message_reaches_llm(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path, reply="Hello Alice.")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        channel = _make_channel(12345)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        asyncio.run(
            _run_text_response(
                channel_id=12345,
                guild_id=999,
                speaker="Alice",
                utterance="hi",
                buffer=buffer,
                familiar=familiar,
                channel=channel,
            )
        )

        llm = familiar.llm_clients["main_prose"]
        assert isinstance(llm, _StubLLMClient)
        main_calls = [c for c in llm.calls if c and c[0].role == "system"]
        assert len(main_calls) == 1
        messages = main_calls[0]
        assert messages[0].role == "system"
        assert messages[-1].role == "user"
        assert messages[-1].content == "Alice: hi"

    def test_reply_posted_to_channel(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path, reply="Hello Alice.")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        channel = _make_channel(12345)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        asyncio.run(
            _run_text_response(
                channel_id=12345,
                guild_id=999,
                speaker="Alice",
                utterance="hi",
                buffer=buffer,
                familiar=familiar,
                channel=channel,
            )
        )

        channel.send.assert_called_once_with("Hello Alice.")

    def test_turns_are_persisted_to_history(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path, reply="Hello.")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        channel = _make_channel(12345)
        buffer = [BufferedMessage(speaker="Alice", text="hi there", timestamp=0.0)]

        asyncio.run(
            _run_text_response(
                channel_id=12345,
                guild_id=999,
                speaker="Alice",
                utterance="hi there",
                buffer=buffer,
                familiar=familiar,
                channel=channel,
            )
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id,
            channel_id=12345,
            limit=10,
        )
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "hi there"
        assert turns[1].role == "assistant"
        assert turns[1].content == "Hello."

    def test_multiple_buffered_messages_all_persisted(self, tmp_path: Path) -> None:
        """All buffered user messages are persisted, not just the trigger."""
        familiar = _make_familiar(tmp_path, reply="Sure.")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        channel = _make_channel(12345)
        buffer = [
            BufferedMessage(speaker="Alice", text="msg1", timestamp=0.0),
            BufferedMessage(speaker="Bob", text="msg2", timestamp=1.0),
            BufferedMessage(speaker="Alice", text="msg3", timestamp=2.0),
        ]

        asyncio.run(
            _run_text_response(
                channel_id=12345,
                guild_id=999,
                speaker="Alice",
                utterance="msg3",
                buffer=buffer,
                familiar=familiar,
                channel=channel,
            )
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id,
            channel_id=12345,
            limit=20,
        )
        user_turns = [t for t in turns if t.role == "user"]
        assert len(user_turns) == 3
        assert user_turns[0].content == "msg1"
        assert user_turns[1].content == "msg2"
        assert user_turns[2].content == "msg3"

    def test_llm_request_log_shows_only_new_messages(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """LLM request log shows total message count and new messages only."""
        familiar = _make_familiar(tmp_path, reply="Hello.")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        channel = _make_channel(12345)
        buffer = [
            BufferedMessage(speaker="Alice", text="first message", timestamp=0.0),
            BufferedMessage(speaker="Bob", text="second message", timestamp=1.0),
        ]

        with caplog.at_level("INFO", logger="familiar_connect.bot"):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Bob",
                    utterance="second message",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        llm_records = [r for r in caplog.records if "Generating Text" in r.getMessage()]
        assert len(llm_records) == 1
        msg = _strip(llm_records[0].getMessage())
        # New format: "messages=N new=K"
        assert "new=2" in msg
        # New messages are shown
        assert "first message" in msg
        assert "second message" in msg

    def test_voice_llm_request_log_shows_only_new_message(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Voice LLM request log shows total count and the utterance only."""
        familiar = _make_familiar(tmp_path, reply="I hear you.")
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        buffer = [
            BufferedMessage(speaker="Alice", text="hello world", timestamp=0.0),
        ]

        with caplog.at_level("INFO", logger="familiar_connect.bot"):
            asyncio.run(
                _run_voice_response(
                    channel_id=9000,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hello world",
                    buffer=buffer,
                    familiar=familiar,
                    vc=vc,
                    trigger=ResponseTrigger.direct_address,
                )
            )

        llm_records = [
            r for r in caplog.records if "Generating Voice" in r.getMessage()
        ]
        assert len(llm_records) == 1
        msg = _strip(llm_records[0].getMessage())
        # New format: "[🧠 Generating Voice] channel=… messages=N new=K"
        assert "new=1" in msg
        # Utterance text is shown
        assert "hello world" in msg

    def test_run_text_response_skips_closed_thread(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Archived+locked thread: log error, do not call send or LLM."""
        familiar = _make_familiar(tmp_path, reply="should not be sent")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )

        thread = MagicMock(spec=discord.Thread)
        thread.id = 12345
        thread.archived = True
        thread.locked = True
        thread.send = AsyncMock()
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=False)
        thread.typing = MagicMock(return_value=typing_cm)

        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with caplog.at_level("ERROR", logger="familiar_connect.bot"):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=thread,
                )
            )

        thread.send.assert_not_called()
        llm = familiar.llm_clients["main_prose"]
        assert isinstance(llm, _StubLLMClient)
        assert llm.calls == []
        skipped = [r for r in caplog.records if "Send skipped" in r.getMessage()]
        assert len(skipped) == 1
        assert "thread_closed" in skipped[0].getMessage()


# ---------------------------------------------------------------------------
# /subscribe-text and friends
# ---------------------------------------------------------------------------


class TestSubscriptionCommands:
    def test_subscribe_text_registers_channel(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=42, guild_id=999)

        asyncio.run(subscribe_text(ctx, familiar))

        assert (
            familiar.subscriptions.get(channel_id=42, kind=SubscriptionKind.text)
            is not None
        )
        ctx.respond.assert_called_once_with(ANY, ephemeral=True)

    def test_subscribe_text_persists_across_reload(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=42, guild_id=999)
        asyncio.run(subscribe_text(ctx, familiar))

        # Reload the Familiar bundle; the subscription should come back.
        reloaded = Familiar.load_from_disk(
            familiar.root,
            llm_clients=_make_llm_clients(),
            defaults_path=_DEFAULT_PROFILE_PATH,
        )
        assert (
            reloaded.subscriptions.get(channel_id=42, kind=SubscriptionKind.text)
            is not None
        )

    def test_unsubscribe_text_removes_channel(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=42,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        ctx = _make_text_ctx(channel_id=42, guild_id=999)

        asyncio.run(unsubscribe_text(ctx, familiar))

        assert (
            familiar.subscriptions.get(channel_id=42, kind=SubscriptionKind.text)
            is None
        )
        # deferred + followup because flush() can exceed Discord's 3s window
        ctx.defer.assert_called_once_with(ephemeral=True)
        ctx.followup.send.assert_called_once_with(ANY, ephemeral=True)

    def test_unsubscribe_text_defers_before_flush(self, tmp_path: Path) -> None:
        """Defer must happen before flush so slow flush can't expire interaction."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=42,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        call_order: list[str] = []
        ctx = _make_text_ctx(channel_id=42, guild_id=999)
        ctx.defer.side_effect = lambda **_: call_order.append("defer")
        flush_mock = AsyncMock(side_effect=lambda: call_order.append("flush"))
        familiar.memory_writer_scheduler.flush = flush_mock  # ty: ignore[invalid-assignment]

        asyncio.run(unsubscribe_text(ctx, familiar))

        assert call_order == ["defer", "flush"]

    def test_unsubscribe_text_clears_monitor_state(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=42,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        mock_clear = MagicMock()
        familiar.monitor.clear_channel = mock_clear  # ty: ignore[invalid-assignment]
        ctx = _make_text_ctx(channel_id=42, guild_id=999)

        asyncio.run(unsubscribe_text(ctx, familiar))

        mock_clear.assert_called_once_with(42)

    def test_subscribe_text_no_send_permission_returns_ephemeral(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=42, guild_id=999, send_messages=False)

        asyncio.run(subscribe_text(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True
        assert (
            familiar.subscriptions.get(channel_id=42, kind=SubscriptionKind.text)
            is None
        )

    def test_subscribe_text_no_view_permission_returns_ephemeral(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=42, guild_id=999, view_channel=False)

        asyncio.run(subscribe_text(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True
        assert (
            familiar.subscriptions.get(channel_id=42, kind=SubscriptionKind.text)
            is None
        )

    def test_subscribe_text_in_thread_registers_channel(self, tmp_path: Path) -> None:
        """Thread subscription is accepted + thread.join is awaited."""
        familiar = _make_familiar(tmp_path)
        ctx = _make_thread_ctx(channel_id=77, guild_id=999)

        asyncio.run(subscribe_text(ctx, familiar))

        assert (
            familiar.subscriptions.get(
                channel_id=77,
                kind=SubscriptionKind.text,
            )
            is not None
        )
        ctx.channel.join.assert_awaited_once()
        ctx.respond.assert_called_once_with(ANY, ephemeral=True)

    def test_subscribe_text_in_thread_records_thread_context(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_thread_ctx(channel_id=77, parent_name="general")
        asyncio.run(subscribe_text(ctx, familiar))

        label = familiar.monitor.format_channel_context(77)
        assert label == "#general -> feature-brainstorm"

    def test_subscribe_text_in_forum_post_records_forum_context(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_thread_ctx(
            channel_id=77,
            parent_name="announcements",
            parent_type=discord.ForumChannel,
        )
        asyncio.run(subscribe_text(ctx, familiar))

        label = familiar.monitor.format_channel_context(77)
        assert label == "forum:announcements -> feature-brainstorm"

    def test_subscribe_text_in_thread_rejects_without_send_perm(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_thread_ctx(
            channel_id=77,
            send_messages_in_threads=False,
        )
        asyncio.run(subscribe_text(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True
        assert (
            familiar.subscriptions.get(
                channel_id=77,
                kind=SubscriptionKind.text,
            )
            is None
        )
        ctx.channel.join.assert_not_called()

    def test_subscribe_text_rejects_forum_root(self, tmp_path: Path) -> None:
        """The forum channel itself has no messages — reject, explain."""
        familiar = _make_familiar(tmp_path)
        ctx = MagicMock(spec=discord.ApplicationContext)
        ctx.respond = AsyncMock()
        forum = MagicMock(spec=discord.ForumChannel)
        forum.id = 77
        forum.name = "announcements"
        type(ctx).channel = PropertyMock(return_value=forum)
        type(ctx).channel_id = PropertyMock(return_value=77)
        guild = MagicMock(spec=discord.Guild)
        guild.id = 999
        type(guild).me = PropertyMock(return_value=MagicMock(spec=discord.Member))
        type(ctx).guild = PropertyMock(return_value=guild)
        type(ctx).guild_id = PropertyMock(return_value=999)

        asyncio.run(subscribe_text(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True
        assert (
            familiar.subscriptions.get(
                channel_id=77,
                kind=SubscriptionKind.text,
            )
            is None
        )

    def test_subscribe_text_thread_join_http_error_is_swallowed(
        self, tmp_path: Path
    ) -> None:
        """Transient join failure still registers the subscription."""
        familiar = _make_familiar(tmp_path)
        ctx = _make_thread_ctx(channel_id=77)
        ctx.channel.join = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "boom"),
        )

        asyncio.run(subscribe_text(ctx, familiar))

        assert (
            familiar.subscriptions.get(
                channel_id=77,
                kind=SubscriptionKind.text,
            )
            is not None
        )


# ---------------------------------------------------------------------------
# /subscribe-my-voice
# ---------------------------------------------------------------------------


class TestVoiceSubscription:
    def test_subscribe_my_voice_joins_and_registers(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_voice_ctx()

        asyncio.run(subscribe_my_voice(ctx, familiar))

        voice_channel = ctx.author.voice.channel
        voice_channel.connect.assert_called_once()
        assert familiar.subscriptions.voice_in_guild(999) is not None
        ctx.defer.assert_called_once_with(ephemeral=True)
        ctx.followup.send.assert_called_once_with(ANY, ephemeral=True)

    def test_subscribe_my_voice_skips_transcription_when_no_transcriber(
        self,
        tmp_path: Path,
    ) -> None:
        """Without a transcriber, the bot joins and greets but doesn't record."""
        familiar = _make_familiar(tmp_path)
        assert familiar.transcriber is None
        ctx = _make_voice_ctx()

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
        ) as mock_start:
            asyncio.run(subscribe_my_voice(ctx, familiar))

        mock_start.assert_not_called()
        voice_channel = ctx.author.voice.channel
        voice_channel.connect.return_value.start_recording.assert_not_called()

    def test_subscribe_my_voice_starts_pipeline_when_transcriber_present(
        self,
        tmp_path: Path,
    ) -> None:
        """``familiar.transcriber`` being set fires ``start_pipeline``.

        The voice client begins recording against a ``RecordingSink``
        backed by the pipeline's tagged queue. This is the PR #17
        wiring, retargeted at the subscription surface.
        """
        familiar = _make_familiar(tmp_path)
        familiar.transcriber = MagicMock(name="transcriber")
        ctx = _make_voice_ctx()

        mock_pipeline = MagicMock()
        mock_pipeline.tagged_audio_queue = MagicMock(name="audio_queue")

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline,
        ) as mock_start:
            asyncio.run(subscribe_my_voice(ctx, familiar))

        mock_start.assert_called_once()
        # The transcriber we put on the Familiar reached start_pipeline.
        assert mock_start.call_args.args[0] is familiar.transcriber
        # user_names is keyed by member id — discord.Member mocks have .id,
        # and _make_voice_ctx doesn't attach members, so the dict is empty
        # but the key kwarg is present.
        assert "user_names" in mock_start.call_args.kwargs
        # start_recording was actually called on the voice client.
        voice_channel = ctx.author.voice.channel
        voice_channel.connect.return_value.start_recording.assert_called_once()

    def test_unsubscribe_voice_disconnects(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000,
            kind=SubscriptionKind.voice,
            guild_id=999,
        )
        # Pretend the bot is currently connected.
        ctx = _make_voice_ctx(already_connected=True)

        asyncio.run(unsubscribe_voice(ctx, familiar))

        ctx.voice_client.disconnect.assert_called_once()
        assert familiar.subscriptions.voice_in_guild(999) is None
        # deferred + followup because flush() can exceed Discord's 3s window
        ctx.defer.assert_called_once_with(ephemeral=True)
        ctx.followup.send.assert_called_once_with(ANY, ephemeral=True)

    def test_unsubscribe_voice_defers_before_flush(self, tmp_path: Path) -> None:
        """Defer must precede disconnect+flush so they can't expire interaction."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000,
            kind=SubscriptionKind.voice,
            guild_id=999,
        )
        call_order: list[str] = []
        ctx = _make_voice_ctx(already_connected=True)
        ctx.defer.side_effect = lambda **_: call_order.append("defer")
        ctx.voice_client.disconnect = AsyncMock(
            side_effect=lambda: call_order.append("disconnect"),
        )
        flush_mock = AsyncMock(side_effect=lambda: call_order.append("flush"))
        familiar.memory_writer_scheduler.flush = flush_mock  # ty: ignore[invalid-assignment]

        asyncio.run(unsubscribe_voice(ctx, familiar))

        assert call_order[0] == "defer"
        assert "flush" in call_order
        assert call_order.index("defer") < call_order.index("flush")

    def test_unsubscribe_voice_stops_active_pipeline(self, tmp_path: Path) -> None:
        """Active voice pipeline is torn down before the voice client disconnects.

        Mirrors the intent of main's ``TestSleepPipelineTeardown``,
        retargeted at ``/unsubscribe-voice``.
        """
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000,
            kind=SubscriptionKind.voice,
            guild_id=999,
        )
        ctx = _make_voice_ctx(already_connected=True)
        # Mark the voice client as currently recording.
        ctx.voice_client.recording = True
        ctx.voice_client.stop_recording = MagicMock()

        with (
            patch(
                "familiar_connect.bot.get_pipeline",
                return_value=MagicMock(),
            ),
            patch(
                "familiar_connect.bot.stop_pipeline",
                new_callable=AsyncMock,
            ) as mock_stop,
        ):
            asyncio.run(unsubscribe_voice(ctx, familiar))

        ctx.voice_client.stop_recording.assert_called_once()
        mock_stop.assert_called_once()
        ctx.voice_client.disconnect.assert_called_once()

    def test_unsubscribe_voice_not_connected_returns_ephemeral(
        self, tmp_path: Path
    ) -> None:
        """Trying to leave voice when not in a channel should report an error."""
        familiar = _make_familiar(tmp_path)
        ctx = _make_voice_ctx(already_connected=False)

        asyncio.run(unsubscribe_voice(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True

    def test_subscribe_my_voice_connect_failure_does_not_register(
        self, tmp_path: Path
    ) -> None:
        """If channel.connect() raises, no subscription should be created."""
        familiar = _make_familiar(tmp_path)
        ctx = _make_voice_ctx()
        ctx.author.voice.channel.connect = AsyncMock(
            side_effect=discord.errors.ClientException("cannot connect"),
        )

        asyncio.run(subscribe_my_voice(ctx, familiar))

        # Should have sent an ephemeral error via followup (after defer).
        ctx.followup.send.assert_called_once()
        _, kwargs = ctx.followup.send.call_args
        assert kwargs.get("ephemeral") is True
        assert familiar.subscriptions.voice_in_guild(999) is None

    def test_subscribe_my_voice_no_connect_permission_returns_ephemeral(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_voice_ctx(connect=False)

        asyncio.run(subscribe_my_voice(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True
        # Bot should NOT have attempted to connect.
        ctx.author.voice.channel.connect.assert_not_called()
        assert familiar.subscriptions.voice_in_guild(999) is None

    def test_subscribe_my_voice_no_speak_permission_returns_ephemeral(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_voice_ctx(speak=False)

        asyncio.run(subscribe_my_voice(ctx, familiar))

        ctx.respond.assert_called_once()
        _, kwargs = ctx.respond.call_args
        assert kwargs.get("ephemeral") is True
        ctx.author.voice.channel.connect.assert_not_called()
        assert familiar.subscriptions.voice_in_guild(999) is None


# ---------------------------------------------------------------------------
# /channel-* mode commands
# ---------------------------------------------------------------------------


class TestChannelModeCommands:
    def test_set_channel_mode_writes_sidecar(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=77)

        asyncio.run(set_channel_mode(ctx, familiar, ChannelMode.full_rp))

        assert familiar.channel_configs.get(channel_id=77).mode is ChannelMode.full_rp
        ctx.respond.assert_called_once_with(ANY, ephemeral=True)

    def test_set_channel_mode_persists_across_reload(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=77)
        asyncio.run(set_channel_mode(ctx, familiar, ChannelMode.imitate_voice))

        reloaded = Familiar.load_from_disk(
            familiar.root,
            llm_clients=_make_llm_clients(),
            defaults_path=_DEFAULT_PROFILE_PATH,
        )
        cfg = reloaded.channel_configs.get(channel_id=77)
        assert cfg.mode is ChannelMode.imitate_voice


# ---------------------------------------------------------------------------
# create_bot
# ---------------------------------------------------------------------------


class TestCreateBot:
    def test_create_bot_returns_discord_bot(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        bot = create_bot(familiar)
        assert isinstance(bot, discord.Bot)

    @pytest.mark.parametrize(
        "expected",
        [
            "subscribe-text",
            "unsubscribe-text",
            "subscribe-my-voice",
            "unsubscribe-voice",
            "channel-full-rp",
            "channel-text-conversation-rp",
            "channel-imitate-voice",
            "channel-backdrop",
        ],
    )
    def test_create_bot_registers_slash_command(
        self,
        tmp_path: Path,
        expected: str,
    ) -> None:
        familiar = _make_familiar(tmp_path)
        bot = create_bot(familiar)
        names = [cmd.name for cmd in bot.pending_application_commands]
        assert expected in names

    def test_create_bot_does_not_register_awaken_or_sleep(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        bot = create_bot(familiar)
        names = [cmd.name for cmd in bot.pending_application_commands]
        assert "awaken" not in names
        assert "sleep" not in names

    def test_channel_backdrop_has_no_options(self, tmp_path: Path) -> None:
        """``/channel-backdrop`` takes no slash-command options.

        All input flows through the Discord modal; submitting blank clears.
        """
        familiar = _make_familiar(tmp_path)
        bot = create_bot(familiar)
        cmd = next(
            c for c in bot.pending_application_commands if c.name == "channel-backdrop"
        )
        assert cmd.options == []


class TestBackdropPlaceholder:
    """Preview of the mode default shown as Discord modal placeholder text.

    Discord caps InputText placeholder at 100 chars, so longer defaults
    are collapsed to a single line and truncated.
    """

    def test_none_returns_generic_fallback(self) -> None:
        out = _default_backdrop_placeholder(None)
        assert "mode default" in out
        assert len(out) <= 100

    def test_short_default_passes_through(self) -> None:
        out = _default_backdrop_placeholder("Talk like a pirate.")
        assert out == "Talk like a pirate."

    def test_long_default_truncated_with_ellipsis(self) -> None:
        long_text = "a" * 200
        out = _default_backdrop_placeholder(long_text)
        assert len(out) == 100
        assert out.endswith("\u2026")

    def test_multiline_default_collapsed_to_single_line(self) -> None:
        out = _default_backdrop_placeholder("line one\n\n  line two")
        assert "\n" not in out
        assert out == "line one line two"


class TestBackdropModalRequiredFlag:
    """Modal must send ``required=false`` so emptied text clears the backdrop.

    Regression against a py-cord quirk where
    ``InputText._generate_underlying`` collapses ``required=False`` to
    ``None`` via ``False or self.required``, which Discord then defaults
    to ``required=true``.
    """

    @pytest.mark.asyncio
    async def test_required_false_reaches_the_component_payload(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path)
        modal = _make_backdrop_modal(
            familiar,
            channel_id=1,
            channel_name="general",
            current="Talk like a pirate.",
            mode_default="Default text.",
        )
        field = modal.children[0]
        assert field.required is False
        assert field._underlying.to_dict()["required"] is False


# ---------------------------------------------------------------------------
# /channel-backdrop in a thread writes a labelled channel_name to the sidecar
# ---------------------------------------------------------------------------


class TestChannelBackdropInThread:
    @pytest.mark.asyncio
    async def test_backdrop_in_thread_writes_labelled_channel_name(
        self, tmp_path: Path
    ) -> None:
        """Sidecar ``channel_name`` shows ``#parent -> thread`` label.

        Guards that ``channel_backdrop`` → modal callback writes the
        human-readable label rather than the raw integer thread id.
        """
        familiar = _make_familiar(tmp_path)
        ctx = _make_thread_ctx(channel_id=77, parent_name="general")
        ctx.send_modal = AsyncMock()

        await channel_backdrop(ctx, familiar)

        modal = ctx.send_modal.call_args[0][0]
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        modal.children[0].value = "Speak like a wizard."
        await modal.callback(interaction)

        sidecar = familiar.root / "channels" / "77.toml"
        assert sidecar.exists()
        cfg = familiar.channel_configs.get(channel_id=77)
        assert cfg.channel_name == "#general -> feature-brainstorm"


# ---------------------------------------------------------------------------
# Main-reply resilience — failing LLMClient.chat does not poison the channel
# ---------------------------------------------------------------------------


def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build a realistic ``httpx.HTTPStatusError`` for a synthetic response."""
    request = httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"{status_code} server error",
        request=request,
        response=response,
    )


class TestMainReplyResilience:
    """A failing main-reply ``LLMClient.chat`` must not propagate.

    The caller catches the closed raise set
    ``(httpx.HTTPError, ValueError, KeyError)``, logs a warning, and
    returns early — no post-processing, no history write, no Discord
    send, no TTS. The Discord event callback stays alive for the next
    message. History is left untouched so the user can simply retry.
    """

    def test_text_main_reply_llm_failure_does_not_crash(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """httpx.ConnectTimeout from the main LLM is caught and logged."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        failing = _RaisingLLMClient(httpx.ConnectTimeout("boom"))
        familiar.llm_clients["main_prose"] = failing
        channel = _make_channel(12345)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with caplog.at_level("WARNING", logger="familiar_connect.bot"):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        # Main LLM was called exactly once and raised.
        assert failing.call_count == 1
        # No reply reached Discord.
        channel.send.assert_not_called()
        # No history was persisted for this failed turn.
        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=12345, limit=10
        )
        assert turns == []
        # A warning was logged naming the failure site.
        assert any(
            "main reply" in record.getMessage() and record.levelname == "WARNING"
            for record in caplog.records
        )

    def test_text_main_reply_http_500_does_not_crash(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A 5xx HTTPStatusError from the main LLM is caught and logged."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.imitate_voice
        )
        failing = _RaisingLLMClient(_make_http_status_error(500))
        familiar.llm_clients["main_prose"] = failing
        channel = _make_channel(12345)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with caplog.at_level("WARNING", logger="familiar_connect.bot"):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        channel.send.assert_not_called()
        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=12345, limit=10
        )
        assert turns == []
        assert any(
            "main reply" in record.getMessage() and record.levelname == "WARNING"
            for record in caplog.records
        )

    def test_voice_main_reply_failure_does_not_crash(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing main LLM on the voice path returns cleanly.

        No TTS call, no history write, and the handler does not raise
        — the transcriber's callback loop stays alive.
        """
        familiar = _make_familiar(tmp_path)
        # No tts_client on the stub Familiar, so TTS is implicitly not
        # exercised; the assertion is instead that we never reach it.
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        failing = _RaisingLLMClient(_make_http_status_error(503))
        familiar.llm_clients["main_prose"] = failing

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        buffer = [BufferedMessage(speaker="Alice", text="hello there", timestamp=0.0)]

        with caplog.at_level("WARNING", logger="familiar_connect.bot"):
            asyncio.run(
                _run_voice_response(
                    channel_id=9000,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hello there",
                    buffer=buffer,
                    familiar=familiar,
                    vc=vc,
                    trigger=ResponseTrigger.direct_address,
                )
            )

        # Main LLM was invoked and raised.
        assert failing.call_count == 1
        # No audio was played (no TTS fan-out).
        vc.play.assert_not_called()
        # No history turns were written for the failed voice turn.
        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=10
        )
        assert turns == []
        # A warning was logged naming the failure site.
        assert any(
            "main reply" in record.getMessage() and record.levelname == "WARNING"
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Voice pre/post-processor suppression
# ---------------------------------------------------------------------------


class TestVoicePreProcessorsSuppressed:
    """Voice turns must not invoke preprocessing or postprocessing LLMs.

    stepped_thinking (reasoning_context) and recast (post_process_style)
    are disabled for voice to reduce real-time latency. This test pins
    that contract so it cannot be silently regressed.
    """

    def test_voice_handler_does_not_call_reasoning_context_or_post_process_style(
        self,
        tmp_path: Path,
    ) -> None:
        """Neither side-model slot is called when handling a voice turn.

        Uses full_rp mode — which normally enables both stepped_thinking
        (preprocessor) and recast (postprocessor) — so the suppression is
        confirmed against the most permissive baseline.
        """
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        # full_rp enables preprocessors_enabled={"stepped_thinking"} and
        # postprocessors_enabled={"recast"} — worst-case baseline.
        familiar.channel_configs.set_mode(channel_id=9000, mode=ChannelMode.full_rp)

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        buffer = [BufferedMessage(speaker="Alice", text="hello there", timestamp=0.0)]

        asyncio.run(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="hello there",
                buffer=buffer,
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )

        # No LLM slot beyond main_prose should have been touched.
        # interjection_decision is excluded from _run_voice_response —
        # the YES/NO gate fires upstream in ConversationMonitor, before
        # the voice response path runs. This assertion covers the
        # response path alone, where the side-model must not be called.
        llm_only_slots = (
            "reasoning_context",
            "post_process_style",
            "memory_search",
            "history_summary",
            "interjection_decision",
        )
        for slot in llm_only_slots:
            client = familiar.llm_clients[slot]
            assert isinstance(client, _StubLLMClient)
            assert client.calls == [], f"Expected no calls on {slot!r}"


# ---------------------------------------------------------------------------
# Voice routes through ConversationMonitor
# ---------------------------------------------------------------------------


class TestVoiceInterjectionRouting:
    """Voice utterances flow through the same ConversationMonitor as text.

    Pins the design decision that voice turns get a YES/NO side-model
    gate (direct address, counter-based interjection, conversational
    lull) before the response pipeline runs, mirroring text behaviour.
    """

    def test_voice_lull_merged_utterance_routed_to_monitor(
        self, tmp_path: Path
    ) -> None:
        """VoiceLullMonitor fires → ConversationMonitor.on_message(voice_id, …).

        The merged transcript lands at the monitor with the voice
        channel id, the speaker's sanitised display name, and
        ``is_mention=False``. The monitor's own direct-address scan
        picks up name/alias hits from the transcript text.
        """
        familiar = _make_familiar(tmp_path)
        familiar.transcriber = MagicMock(name="transcriber")
        ctx = _make_voice_ctx(channel_id=9000)

        monitor_spy = AsyncMock()
        familiar.monitor.on_message = monitor_spy  # ty: ignore[invalid-assignment]

        mock_pipeline = MagicMock()
        mock_pipeline.tagged_audio_queue = MagicMock(name="audio_queue")

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline,
        ):
            asyncio.run(subscribe_my_voice(ctx, familiar))

        lull_monitor = familiar.extras["voice_lull_monitor"]
        assert isinstance(lull_monitor, VoiceLullMonitor)
        # Reach in to the lull monitor's private hand-off: we only want
        # to verify the subscribe wiring, not re-simulate Deepgram.
        merged = TranscriptionResult(
            text="hello there",
            is_final=True,
            start=0.0,
            end=1.0,
        )

        async def _fire() -> None:
            await lull_monitor._on_utterance_complete(42, merged)

        asyncio.run(_fire())

        monitor_spy.assert_called_once()
        kwargs = monitor_spy.call_args.kwargs
        assert kwargs["channel_id"] == 9000
        assert kwargs["text"] == "hello there"
        assert kwargs["is_mention"] is False
        # Voice already debounced silence — monitor must not start its
        # own (text) lull timer for voice channels.
        assert kwargs["is_lull_endpoint"] is True

    def test_subscribe_my_voice_registers_voice_response_handler(
        self, tmp_path: Path
    ) -> None:
        """A per-channel voice response handler is stashed on familiar.extras."""
        familiar = _make_familiar(tmp_path)
        familiar.transcriber = MagicMock(name="transcriber")
        ctx = _make_voice_ctx(channel_id=9000)

        mock_pipeline = MagicMock()
        mock_pipeline.tagged_audio_queue = MagicMock(name="audio_queue")

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline,
        ):
            asyncio.run(subscribe_my_voice(ctx, familiar))

        handlers = cast("dict[int, Any]", familiar.extras["voice_response_handlers"])
        assert 9000 in handlers
        assert callable(handlers[9000])

    def test_unsubscribe_voice_drops_response_handler_and_monitor_state(
        self, tmp_path: Path
    ) -> None:
        """Cleanup removes the dispatch entry and clears monitor state."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        # Pretend we had wired everything up previously.
        familiar.extras["voice_response_handlers"] = {9000: AsyncMock()}
        clear_spy = MagicMock()
        familiar.monitor.clear_channel = clear_spy  # ty: ignore[invalid-assignment]
        ctx = _make_voice_ctx(already_connected=True)

        asyncio.run(unsubscribe_voice(ctx, familiar))

        handlers = cast("dict[int, Any]", familiar.extras["voice_response_handlers"])
        assert 9000 not in handlers
        clear_spy.assert_called_once_with(9000)

    def test_on_respond_dispatches_to_voice_handler(self, tmp_path: Path) -> None:
        """create_bot's on_respond hands voice channels to the voice handler.

        When a voice-channel handler is registered in
        ``familiar.extras["voice_response_handlers"]``, the
        ``on_respond`` callback routes there instead of attempting the
        text-channel path.
        """
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        voice_handler = AsyncMock()
        familiar.extras["voice_response_handlers"] = {9000: voice_handler}

        bot = create_bot(familiar)
        del bot  # we only need the side-effect: monitor.on_respond installed

        buffer = [BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)]

        async def _invoke() -> None:
            await familiar.monitor.on_respond(
                9000, buffer, ResponseTrigger.direct_address
            )

        asyncio.run(_invoke())

        voice_handler.assert_called_once()
        args, _ = voice_handler.call_args
        assert args[0] == 9000
        assert args[1] == buffer
        assert args[2] is ResponseTrigger.direct_address

    def test_on_respond_text_channel_bypasses_voice_handler_lookup(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: text channels are unaffected by voice wiring.

        A text channel id that is NOT in ``voice_response_handlers``
        must still reach ``_run_text_response`` exactly as before.
        Covers the case where a single familiar has both a voice
        subscription (populates the handlers dict) and a text
        subscription.
        """
        familiar = _make_familiar(tmp_path, reply="Hello.")
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        # A voice handler is registered for the voice channel only.
        familiar.extras["voice_response_handlers"] = {9000: AsyncMock()}

        text_channel = _make_channel(12345)

        with patch(
            "familiar_connect.bot._run_text_response", new_callable=AsyncMock
        ) as text_spy:
            bot = create_bot(familiar)
            # bot.get_channel returns our fake text channel for this id.
            bot.get_channel = MagicMock(  # ty: ignore[invalid-assignment]
                return_value=text_channel,
            )

            buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

            async def _invoke() -> None:
                await familiar.monitor.on_respond(
                    12345, buffer, ResponseTrigger.direct_address
                )

            asyncio.run(_invoke())

        text_spy.assert_called_once()
        kwargs = text_spy.call_args.kwargs
        assert kwargs["channel_id"] == 12345
        assert kwargs["channel"] is text_channel
        # The voice handler was NOT invoked.
        voice_handlers = cast(
            "dict[int, Any]", familiar.extras["voice_response_handlers"]
        )
        voice_handlers[9000].assert_not_called()

    def test_run_voice_response_persists_all_buffered_turns(
        self, tmp_path: Path
    ) -> None:
        """Every utterance in the buffer is persisted, then the reply.

        Parallels TestOnRespond.test_multiple_buffered_messages_all_persisted
        for voice.
        """
        familiar = _make_familiar(tmp_path, reply="Sure thing.")
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)
        buffer = [
            BufferedMessage(speaker="Alice", text="hey", timestamp=0.0),
            BufferedMessage(speaker="Bob", text="what's up", timestamp=1.0),
            BufferedMessage(speaker="Alice", text="aria you there", timestamp=2.0),
        ]

        asyncio.run(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="aria you there",
                buffer=buffer,
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=10
        )
        # 3 user turns + 1 assistant reply
        assert len(turns) == 4
        assert [t.role for t in turns] == ["user", "user", "user", "assistant"]
        assert [t.content for t in turns] == [
            "hey",
            "what's up",
            "aria you there",
            "Sure thing.",
        ]


# ---------------------------------------------------------------------------
# Cancellable LLM generation (Step 7 of the voice-interruption plan)
# ---------------------------------------------------------------------------


class TestVoiceGenerationCancellation:
    """The voice LLM call is exposed as a cancellable task on the tracker.

    Step 7 introduces ``tracker.generation_task`` so later steps can
    cancel mid-generation on a long interruption. No caller cancels
    yet; this class pins the plumbing contract.
    """

    def test_generation_task_cleared_after_normal_completion(
        self, tmp_path: Path
    ) -> None:
        """A successful voice turn leaves ``generation_task = None``."""
        familiar = _make_familiar(tmp_path, reply="Sure thing.")
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        asyncio.run(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="hey",
                buffer=[BufferedMessage(speaker="Alice", text="hey", timestamp=0.0)],
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )

        tracker = familiar.tracker_registry.get(999)
        assert tracker.generation_task is None
        assert tracker.state is ResponseState.IDLE

    def test_cancelling_generation_task_mid_flight_exits_cleanly(
        self, tmp_path: Path
    ) -> None:
        """Cancelling ``tracker.generation_task`` aborts the voice turn.

        The LLM ``chat`` coroutine sees ``CancelledError`` (so no wasted
        tokens), ``_run_voice_response`` returns without raising,
        nothing is persisted to history, no audio is played, and the
        tracker returns to ``IDLE``.
        """
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        slow_client = _SlowLLMClient()
        familiar.llm_clients["main_prose"] = slow_client

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        async def _run() -> None:
            task = asyncio.create_task(
                _run_voice_response(
                    channel_id=9000,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hey",
                    buffer=[
                        BufferedMessage(speaker="Alice", text="hey", timestamp=0.0)
                    ],
                    familiar=familiar,
                    vc=vc,
                    trigger=ResponseTrigger.direct_address,
                )
            )
            # Wait until the chat coroutine is parked inside the LLM.
            await slow_client.started.wait()
            tracker = familiar.tracker_registry.get(999)
            assert tracker.generation_task is not None
            assert not tracker.generation_task.done()
            assert tracker.state is ResponseState.GENERATING
            tracker.generation_task.cancel()
            await task  # Must return without raising.

        asyncio.run(_run())

        tracker = familiar.tracker_registry.get(999)
        assert slow_client.was_cancelled
        assert tracker.generation_task is None
        assert tracker.state is ResponseState.IDLE
        # No TTS, no history writes.
        vc.play.assert_not_called()
        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=10
        )
        assert turns == []


# ---------------------------------------------------------------------------
# Step 8 — dispatch_interruption_regen
# ---------------------------------------------------------------------------


class TestDispatchInterruptionRegen:
    """Step 8: long interruption during GENERATING cancels + re-generates.

    The :func:`dispatch_interruption_regen` helper:

    - Cancels the in-flight ``generation_task`` on the tracker.
    - Awaits the cancelled task so :func:`_run_voice_response` can clean up.
    - Calls :func:`_run_voice_response` again with an ``interruption_context``
      note so the new reply can acknowledge what the user said.
    - The cancelled turn writes nothing to history.
    """

    @pytest.mark.asyncio
    async def test_long_interruption_during_generating_cancels_task(
        self, tmp_path: Path
    ) -> None:
        """Generation task receives ``CancelledError`` when dispatch fires."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        slow_client = _SlowLLMClient()
        familiar.llm_clients["main_prose"] = slow_client

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        voice_task = asyncio.create_task(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="original",
                buffer=[
                    BufferedMessage(speaker="Alice", text="original", timestamp=0.0)
                ],
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )
        # Wait until generation is in-flight.
        await slow_client.started.wait()
        tracker = familiar.tracker_registry.get(999)
        assert tracker.state is ResponseState.GENERATING

        # Fire the interruption dispatch with a fast-returning stub so
        # the regen completes.
        familiar.llm_clients["main_prose"] = _StubLLMClient(reply="Regen reply.")
        dispatch_task = asyncio.create_task(
            dispatch_interruption_regen(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                transcript="hey stop",
                familiar=familiar,
                vc=vc,
            )
        )
        await dispatch_task
        await voice_task  # original _run_voice_response must exit cleanly.

        assert slow_client.was_cancelled

    @pytest.mark.asyncio
    async def test_long_interruption_during_generating_regens_with_context(
        self, tmp_path: Path
    ) -> None:
        """New LLM call contains the interruption_context note."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        slow_client = _SlowLLMClient()
        familiar.llm_clients["main_prose"] = slow_client

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        voice_task = asyncio.create_task(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="original",
                buffer=[
                    BufferedMessage(speaker="Alice", text="original", timestamp=0.0)
                ],
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )
        await slow_client.started.wait()

        regen_client = _StubLLMClient(reply="I heard you.")
        familiar.llm_clients["main_prose"] = regen_client

        dispatch_task = asyncio.create_task(
            dispatch_interruption_regen(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                transcript="hey stop",
                familiar=familiar,
                vc=vc,
            )
        )
        await dispatch_task
        await voice_task

        # Regen LLM must have been called.
        assert len(regen_client.calls) == 1
        # The message list sent to the LLM must contain the interruption note.
        all_content = " ".join(m.content for m in regen_client.calls[0] if m.content)
        assert "interrupted while you were forming a response" in all_content
        assert "hey stop" in all_content

    @pytest.mark.asyncio
    async def test_long_interruption_during_generating_no_history_from_cancelled_turn(
        self, tmp_path: Path
    ) -> None:
        """The cancelled generation must not write any history turns."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        slow_client = _SlowLLMClient()
        familiar.llm_clients["main_prose"] = slow_client

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        voice_task = asyncio.create_task(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="original",
                buffer=[
                    BufferedMessage(speaker="Alice", text="original", timestamp=0.0)
                ],
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )
        await slow_client.started.wait()

        # Swap to a fast client for the regen.
        familiar.llm_clients["main_prose"] = _StubLLMClient(reply="Regen reply.")

        dispatch_task = asyncio.create_task(
            dispatch_interruption_regen(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                transcript="hey stop",
                familiar=familiar,
                vc=vc,
            )
        )
        await dispatch_task
        await voice_task

        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=20
        )
        # Cancelled turn must not be in history; only the regen turn should exist.
        contents = [t.content for t in turns]
        assert "I am here." not in contents  # default stub reply never stored
        # Regen user + assistant turn written.
        assert any("hey stop" in c for c in contents)
        assert any("Regen reply." in c for c in contents)


def _make_stub_detector(
    classification: InterruptionClass | None,
    *,
    guild_id: int = 999,
) -> InterruptionDetector:
    """InterruptionDetector with wait_for_lull mocked to return *classification*.

    After the stale-gate fix, wait_for_lull() returns None when the gate
    is already open. Mocking wait_for_lull directly is the only reliable
    way to exercise specific delivery-gate outcomes without a real burst.
    """
    registry = ResponseTrackerRegistry()
    detector = InterruptionDetector(
        tracker_registry=registry,
        guild_id=guild_id,
        min_interruption_s=1.5,
        short_long_boundary_s=4.0,
        lull_timeout_s=2.0,
        base_tolerance=0.30,
    )
    detector.wait_for_lull = AsyncMock(return_value=classification)  # ty: ignore[invalid-assignment]
    return detector


class TestDeliveryGate:
    """Delivery gate: _run_voice_response awaits wait_for_lull() after TTS synthesis."""

    @pytest.mark.asyncio
    async def test_long_burst_during_tts_prevents_play(self, tmp_path: Path) -> None:
        """wait_for_lull → long → vc.play not called."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(
            InterruptionClass.long
        )
        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        vc.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_burst_during_tts_skips_history(self, tmp_path: Path) -> None:
        """wait_for_lull → long → no history turns written."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(
            InterruptionClass.long
        )
        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=20
        )
        assert len(turns) == 0

    @pytest.mark.asyncio
    async def test_short_burst_during_tts_allows_play(self, tmp_path: Path) -> None:
        """wait_for_lull → short → vc.play IS called."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(
            InterruptionClass.short
        )
        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        vc.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_burst_gate_skipped(self, tmp_path: Path) -> None:
        """wait_for_lull → None → play happens normally."""
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(None)
        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        vc.play.assert_called_once()


class TestDeliverToMonitorGuard:
    """_deliver_to_monitor skips if _regen_pending is set or tracker is not IDLE."""

    def test_deliver_to_monitor_skipped_when_regen_pending(
        self, tmp_path: Path
    ) -> None:
        """extras['_regen_pending'] → on_message never called."""
        familiar = _make_familiar(tmp_path)
        familiar.transcriber = MagicMock(name="transcriber")
        ctx = _make_voice_ctx(channel_id=9000)

        monitor_spy = AsyncMock()
        familiar.monitor.on_message = monitor_spy  # ty: ignore[invalid-assignment]

        mock_pipeline = MagicMock()
        mock_pipeline.tagged_audio_queue = MagicMock(name="audio_queue")

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline,
        ):
            asyncio.run(subscribe_my_voice(ctx, familiar))

        lull_monitor = familiar.extras["voice_lull_monitor"]
        assert isinstance(lull_monitor, VoiceLullMonitor)

        merged = TranscriptionResult(
            text="hey stop",
            is_final=True,
            start=0.0,
            end=1.0,
        )

        async def _fire() -> None:
            # Simulate: _on_long_during_generating set the flag synchronously,
            # then _deliver_to_monitor fires in the next task.
            familiar.extras["_regen_pending"] = True
            await lull_monitor._on_utterance_complete(42, merged)

        asyncio.run(_fire())

        monitor_spy.assert_not_called()

    def test_deliver_to_monitor_skipped_when_tracker_not_idle(
        self, tmp_path: Path
    ) -> None:
        """tracker.state = GENERATING → on_message never called."""
        familiar = _make_familiar(tmp_path)
        familiar.transcriber = MagicMock(name="transcriber")
        ctx = _make_voice_ctx(channel_id=9000)

        monitor_spy = AsyncMock()
        familiar.monitor.on_message = monitor_spy  # ty: ignore[invalid-assignment]

        mock_pipeline = MagicMock()
        mock_pipeline.tagged_audio_queue = MagicMock(name="audio_queue")

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline,
        ):
            asyncio.run(subscribe_my_voice(ctx, familiar))

        lull_monitor = familiar.extras["voice_lull_monitor"]
        assert isinstance(lull_monitor, VoiceLullMonitor)
        tracker = familiar.tracker_registry.get(999)
        tracker.transition(ResponseState.GENERATING)

        merged = TranscriptionResult(
            text="hello",
            is_final=True,
            start=0.0,
            end=1.0,
        )

        async def _fire() -> None:
            await lull_monitor._on_utterance_complete(42, merged)

        asyncio.run(_fire())

        monitor_spy.assert_not_called()

    def test_deliver_to_monitor_skipped_when_short_yield_pending(
        self, tmp_path: Path
    ) -> None:
        """tracker.short_yield_pending=True → on_message never called.

        Reproduces the race where the voice endpointing lull fires in the
        same event-loop window as InterruptionDetector finalizing a
        short@SPEAKING yield. Without the flag the lull transitions
        IDLE→GENERATING, which causes _on_short_yield_resume to bail.
        """
        familiar = _make_familiar(tmp_path)
        familiar.transcriber = MagicMock(name="transcriber")
        ctx = _make_voice_ctx(channel_id=9000)

        monitor_spy = AsyncMock()
        familiar.monitor.on_message = monitor_spy  # ty: ignore[invalid-assignment]

        mock_pipeline = MagicMock()
        mock_pipeline.tagged_audio_queue = MagicMock(name="audio_queue")

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline,
        ):
            asyncio.run(subscribe_my_voice(ctx, familiar))

        lull_monitor = familiar.extras["voice_lull_monitor"]
        assert isinstance(lull_monitor, VoiceLullMonitor)
        tracker = familiar.tracker_registry.get(999)
        # Simulate InterruptionDetector setting the flag after short@SPEAKING yield.
        tracker.short_yield_pending = True

        merged = TranscriptionResult(
            text="alright i'm going to try to do a building event",
            is_final=True,
            start=0.0,
            end=1.0,
        )

        async def _fire() -> None:
            await lull_monitor._on_utterance_complete(42, merged)

        asyncio.run(_fire())

        monitor_spy.assert_not_called()


def _fake_long_speaking_interrupt(
    tracker: ResponseTracker,
    *,
    elapsed_ms: float,
    transcript: str,
    starter_name: str,
) -> None:
    """Simulate what InterruptionDetector sets at yield + finalization.

    Sets all Step 12 interrupt fields on *tracker* and pre-sets the
    event so ``await interrupt_event.wait()`` returns immediately.
    Call this from a ``vc.play`` side_effect so the interrupt state is
    visible to ``_run_voice_response`` as soon as playback starts.
    """
    evt = asyncio.Event()
    evt.set()
    tracker.interruption_elapsed_ms = elapsed_ms
    tracker.interrupt_event = evt
    tracker.interrupt_classification = InterruptionClass.long
    tracker.interrupt_transcript = transcript
    tracker.interrupt_starter_name = starter_name


class TestLongSpeakingYield:
    """Step 12: long-interruption yield path in ``_run_voice_response``.

    Uses a TTS mock with deterministic word timestamps so ``delivered_text``
    is predictable, and a ``vc.play`` side-effect that injects the
    interrupt state immediately so the test runs without real async delays.
    """

    _WORDS: typing.ClassVar[list[WordTimestamp]] = [
        WordTimestamp("hello", 0.0, 300.0),
        WordTimestamp("world", 400.0, 700.0),
        WordTimestamp("goodbye", 800.0, 1100.0),
    ]

    def _make_tts_familiar(
        self, tmp_path: Path, *, reply: str = "I am here."
    ) -> Familiar:
        familiar = _make_familiar(tmp_path, reply=reply)
        tts_mock = MagicMock()
        tts_mock.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00" * 4, timestamps=list(self._WORDS))
        )
        familiar.tts_client = tts_mock  # type: ignore[assignment]
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        return familiar

    def test_long_speaking_yield_history_records_delivered_only(
        self, tmp_path: Path
    ) -> None:
        """History assistant turn = delivered portion only, not full reply.

        elapsed_ms=350 falls between "hello"@300 and "world"@400,
        so only "hello" was delivered before the interrupt.
        """
        familiar = self._make_tts_familiar(tmp_path, reply="hello world goodbye")
        tracker = familiar.tracker_registry.get(999)

        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)

        def _inject_interrupt(audio: object) -> None:  # noqa: ARG001
            _fake_long_speaking_interrupt(
                tracker,
                elapsed_ms=350.0,  # between "hello"@300 and "world"@400
                transcript="tell me more",
                starter_name="Bob",
            )

        vc.play = MagicMock(side_effect=_inject_interrupt)

        buffer = [BufferedMessage(speaker="Alice", text="hello there", timestamp=0.0)]
        asyncio.run(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="hello there",
                buffer=buffer,
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=20
        )
        assistant_turns = [t for t in turns if t.role == "assistant"]
        # First assistant turn = delivered portion only
        assert assistant_turns[0].content == "hello"
        # Second assistant turn = regen reply (from stub LLM)
        assert len(assistant_turns) == 2

    def test_long_speaking_yield_regen_contains_partial_and_transcript(
        self, tmp_path: Path
    ) -> None:
        """Regen LLM call receives system message with delivered text + transcript."""
        familiar = self._make_tts_familiar(tmp_path, reply="hello world goodbye")
        tracker = familiar.tracker_registry.get(999)

        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)

        def _inject_interrupt(audio: object) -> None:  # noqa: ARG001
            _fake_long_speaking_interrupt(
                tracker,
                elapsed_ms=350.0,
                transcript="tell me more",
                starter_name="Bob",
            )

        vc.play = MagicMock(side_effect=_inject_interrupt)

        buffer = [BufferedMessage(speaker="Alice", text="hello there", timestamp=0.0)]
        asyncio.run(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="hello there",
                buffer=buffer,
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )

        main_prose = cast("_StubLLMClient", familiar.llm_clients["main_prose"])
        # Two LLM calls: original + regen
        assert len(main_prose.calls) == 2
        regen_messages = main_prose.calls[1]
        system_msgs = [m for m in regen_messages if m.role == "system"]
        interruption_notes = [
            m
            for m in system_msgs
            if "interrupted" in m.content and "hello" in m.content
        ]
        assert len(interruption_notes) == 1, "regen must have interruption system note"
        note = interruption_notes[0].content
        assert "hello" in note  # delivered portion
        assert "tell me more" in note  # interrupter's transcript
        assert "Bob" in note  # interrupter's name


class TestShortGeneratingFlush:
    """Step 9: _run_voice_response flushes pending_interrupter_turns.

    Short@GENERATING dispatch stashes (name, transcript) pairs on the
    tracker. _run_voice_response must write them to history AFTER the
    original buffer user turns and BEFORE the assistant reply, then
    clear the list.
    """

    @pytest.mark.asyncio
    async def test_chronology_buffer_interrupter_assistant(
        self, tmp_path: Path
    ) -> None:
        """History order: buffer user turn → interrupter turn → assistant."""
        familiar = _make_familiar(tmp_path, reply="my reply")
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(
            InterruptionClass.short
        )
        tracker = familiar.tracker_registry.get(999)
        tracker.pending_interrupter_turns = [("Bob", "wait what")]

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=20
        )
        # Expected: Alice "hello" (buffer), Bob "wait what" (interrupter),
        # assistant reply "my reply".
        assert [(t.role, t.speaker, t.content) for t in turns] == [
            ("user", "Alice", "hello"),
            ("user", "Bob", "wait what"),
            ("assistant", None, "my reply"),
        ]

    @pytest.mark.asyncio
    async def test_pending_cleared_after_flush(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path, reply="my reply")
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(
            InterruptionClass.short
        )
        tracker = familiar.tracker_registry.get(999)
        tracker.pending_interrupter_turns = [("Bob", "hi"), ("Carol", "hey")]

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        assert tracker.pending_interrupter_turns == []

    @pytest.mark.asyncio
    async def test_pending_cleared_on_long_discard(self, tmp_path: Path) -> None:
        """wait_for_lull → long → pending turns cleared, not written."""
        familiar = _make_familiar(tmp_path, reply="my reply")
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        familiar.tts_client = MagicMock()  # type: ignore[assignment]
        familiar.tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"\x00\x00", timestamps=[])
        )
        familiar.extras["interruption_detector"] = _make_stub_detector(
            InterruptionClass.long
        )
        tracker = familiar.tracker_registry.get(999)
        tracker.pending_interrupter_turns = [("Bob", "wait what")]

        vc = MagicMock(spec=discord.VoiceClient)
        vc.play = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        await _run_voice_response(
            channel_id=9000,
            guild_id=999,
            speaker="Alice",
            utterance="hello",
            buffer=[BufferedMessage(speaker="Alice", text="hello", timestamp=0.0)],
            familiar=familiar,
            vc=vc,
            trigger=ResponseTrigger.direct_address,
        )

        assert tracker.pending_interrupter_turns == []
        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=20
        )
        # Long discard skips buffer and interrupter writes alike.
        assert all("wait what" not in t.content for t in turns)


def _fake_short_speaking_interrupt(
    tracker: ResponseTracker,
) -> None:
    """Simulate a short@SPEAKING yield at finalization.

    Sets the interrupt_event (pre-set) and classification=short so that
    ``_run_voice_response`` wakes from ``interrupt_event.wait()`` and
    takes the short path (write full reply, no regen).
    Call from ``vc.play`` side_effect so the state is visible as soon
    as playback starts.
    """
    evt = asyncio.Event()
    evt.set()
    tracker.interrupt_event = evt
    tracker.interrupt_classification = InterruptionClass.short


class TestShortSpeakingYieldHistory:
    """Step 11: short@SPEAKING yield writes the FULL assistant reply to history.

    Contrast with long@SPEAKING (step 12) which writes only the delivered
    portion. For short yields the familiar is resuming the remainder via
    _on_short_yield_resume, so the history entry should represent the
    complete intended response (not just what came out before the stop).
    """

    def test_short_speaking_yield_writes_full_reply_to_history(
        self, tmp_path: Path
    ) -> None:
        familiar = _make_familiar(tmp_path, reply="hello world goodbye")
        tts_mock = MagicMock()
        tts_mock.synthesize = AsyncMock(
            return_value=TTSResult(
                audio=b"\x00" * 4,
                timestamps=[
                    WordTimestamp("hello", 0.0, 300.0),
                    WordTimestamp("world", 400.0, 700.0),
                    WordTimestamp("goodbye", 800.0, 1100.0),
                ],
            )
        )
        familiar.tts_client = tts_mock  # type: ignore[assignment]
        familiar.subscriptions.add(
            channel_id=9000, kind=SubscriptionKind.voice, guild_id=999
        )
        tracker = familiar.tracker_registry.get(999)

        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)

        def _inject_interrupt(audio: object) -> None:  # noqa: ARG001
            _fake_short_speaking_interrupt(tracker)

        vc.play = MagicMock(side_effect=_inject_interrupt)

        buffer = [BufferedMessage(speaker="Alice", text="go ahead", timestamp=0.0)]
        asyncio.run(
            _run_voice_response(
                channel_id=9000,
                guild_id=999,
                speaker="Alice",
                utterance="go ahead",
                buffer=buffer,
                familiar=familiar,
                vc=vc,
                trigger=ResponseTrigger.direct_address,
            )
        )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id, channel_id=9000, limit=20
        )
        assistant_turns = [t for t in turns if t.role == "assistant"]
        # Exactly one assistant turn — the full reply, not a partial.
        assert len(assistant_turns) == 1
        assert assistant_turns[0].content == "hello world goodbye"


class TestShortYieldResumeRace:
    """_on_short_yield_resume must wait for SPEAKING→IDLE before proceeding.

    Reproduces the race where the resume task is scheduled via
    asyncio.create_task before interrupt_event.set() unblocks
    _run_voice_response. Without the wait loop the guard
    ``if t.state is not ResponseState.IDLE`` bails while still SPEAKING,
    so TTS is never called for the remaining words.
    """

    def test_resume_waits_for_idle_when_still_speaking(self, tmp_path: Path) -> None:
        """Resume callback synthesizes remaining words even if called while SPEAKING."""
        familiar = _make_familiar(tmp_path)
        # Transcriber must be set so subscribe_my_voice creates the
        # interruption_detector (the detector is only built when a
        # transcriber is present).
        familiar.transcriber = MagicMock(name="transcriber")
        tts_mock = MagicMock()
        resume_tts_result = TTSResult(
            audio=b"\x00" * 4,
            timestamps=[
                WordTimestamp("world", 400.0, 700.0),
                WordTimestamp("goodbye", 800.0, 1100.0),
            ],
        )
        # First call: subscribe_my_voice greeting ("Hello!").
        # Subsequent call: the resume synthesis under test.
        greeting_result = TTSResult(audio=b"\x00" * 4, timestamps=[])
        tts_mock.synthesize = AsyncMock(
            side_effect=[greeting_result, resume_tts_result]
        )
        familiar.tts_client = tts_mock  # type: ignore[assignment]

        ctx = _make_voice_ctx(channel_id=9000, guild_id=999)
        vc = ctx.author.voice.channel.connect.return_value

        with patch(
            "familiar_connect.bot.start_pipeline",
            new_callable=AsyncMock,
            return_value=MagicMock(tagged_audio_queue=MagicMock()),
        ):
            asyncio.run(subscribe_my_voice(ctx, familiar))

        detector = familiar.extras["interruption_detector"]
        detector = cast("InterruptionDetector", detector)
        resume_cb = detector._on_short_yield_resume
        assert resume_cb is not None
        tracker = familiar.tracker_registry.get(999)

        remaining = [
            WordTimestamp("world", 400.0, 700.0),
            WordTimestamp("goodbye", 800.0, 1100.0),
        ]

        async def _run() -> None:
            # Simulate the race: tracker is SPEAKING when resume task starts.
            tracker.state = ResponseState.SPEAKING
            # Schedule a task that transitions to IDLE after a short yield —
            # mirrors what _run_voice_response does after interrupt_event fires.

            async def _settle() -> None:
                await asyncio.sleep(0.05)
                tracker.state = ResponseState.IDLE

            asyncio.create_task(_settle())  # noqa: RUF006
            await resume_cb(remaining)

        asyncio.run(_run())

        # Resume must have called synthesize with the remaining words.
        # (First call is the subscribe_my_voice greeting "Hello!".)
        calls = [c.args[0] for c in tts_mock.synthesize.call_args_list]
        assert "world goodbye" in calls, f"resume synthesize not called; calls={calls}"
        assert vc.play.call_count >= 1


# ---------------------------------------------------------------------------
# Typing-simulation chunked delivery path
# ---------------------------------------------------------------------------


class TestTypingSimulationDelivery:
    """``_run_text_response`` under ``typing_simulation.enabled=True``.

    Patches ``asyncio.sleep`` to avoid real-time delays.
    """

    def _setup_channel_rp(
        self, tmp_path: Path, *, reply: str
    ) -> tuple[
        Familiar,
        MagicMock,
    ]:
        """Build familiar + subscribed text-rp channel + fake channel."""
        familiar = _make_familiar(tmp_path, reply=reply)
        familiar.subscriptions.add(
            channel_id=12345, kind=SubscriptionKind.text, guild_id=999
        )
        familiar.channel_configs.set_mode(
            channel_id=12345, mode=ChannelMode.text_conversation_rp
        )
        channel = _make_channel(12345)
        return familiar, channel

    def test_single_paragraph_sent_as_single_message(self, tmp_path: Path) -> None:
        familiar, channel = self._setup_channel_rp(tmp_path, reply="Hello Alice.")
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with patch("familiar_connect.bot.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        # single-paragraph → one send call with full text
        assert channel.send.call_count == 1
        channel.send.assert_called_once_with("Hello Alice.")

    def test_multi_paragraph_sent_as_separate_messages(
        self,
        tmp_path: Path,
    ) -> None:
        reply = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        familiar, channel = self._setup_channel_rp(tmp_path, reply=reply)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with patch("familiar_connect.bot.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        assert channel.send.call_count == 3
        sent_texts = [call.args[0] for call in channel.send.call_args_list]
        assert sent_texts == [
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph.",
        ]

    def test_typing_indicator_entered_per_chunk(self, tmp_path: Path) -> None:
        reply = "Para one.\n\nPara two."
        familiar, channel = self._setup_channel_rp(tmp_path, reply=reply)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with patch("familiar_connect.bot.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        # 1 entry for the outer pipeline-wrapping typing() (still there)
        # + 2 for per-chunk typing() contexts (one per paragraph)
        assert channel.typing.call_count >= 3

    def test_full_reply_persisted_when_delivery_completes(
        self,
        tmp_path: Path,
    ) -> None:
        reply = "First.\n\nSecond."
        familiar, channel = self._setup_channel_rp(tmp_path, reply=reply)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with patch("familiar_connect.bot.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        turns = familiar.history_store.recent(
            familiar_id=familiar.id,
            channel_id=12345,
            limit=10,
        )
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "hi"
        assert turns[1].role == "assistant"
        # chunks rejoined with blank-line separator
        assert turns[1].content == "First.\n\nSecond."

    def test_tracker_cleared_after_successful_delivery(
        self,
        tmp_path: Path,
    ) -> None:
        familiar, channel = self._setup_channel_rp(tmp_path, reply="Hello.")
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        with patch("familiar_connect.bot.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )

        tracker = familiar.text_delivery_registry.get(12345)
        assert tracker.is_active() is False
        assert tracker.sent_chunks == []


class TestTypingSimulationCancellation:
    """Mid-flight cancellation yields partial history; no further sends."""

    def test_cancelled_delivery_persists_sent_chunks_only(
        self,
        tmp_path: Path,
    ) -> None:
        reply = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        familiar = _make_familiar(tmp_path, reply=reply)
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        familiar.channel_configs.set_mode(
            channel_id=12345,
            mode=ChannelMode.text_conversation_rp,
        )
        channel = _make_channel(12345)
        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        # Channel.send blocks on the second call so we can cancel mid-way.
        send_gate = asyncio.Event()
        sends_done = 0

        async def fake_send(text: str) -> None:  # noqa: ARG001
            nonlocal sends_done
            sends_done += 1
            if sends_done == 2:
                send_gate.set()
                await asyncio.sleep(10.0)  # mock.patch will shortcut this
                return
            return

        channel.send = AsyncMock(side_effect=fake_send)

        async def runner() -> None:
            # real asyncio.sleep needed so we can interleave
            task = asyncio.create_task(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )
            # wait until first send+second-send-started
            await send_gate.wait()
            # cancel via tracker (simulates on_message hook)
            tracker = familiar.text_delivery_registry.get(12345)
            await tracker.cancel_and_wait()
            await task

        asyncio.run(runner())

        turns = familiar.history_store.recent(
            familiar_id=familiar.id,
            channel_id=12345,
            limit=10,
        )
        # user turn persisted before delivery
        assert turns[0].role == "user"
        assert turns[0].content == "hi"
        # assistant turn contains only the first (completed) chunk
        assistant_turns = [t for t in turns if t.role == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0].content == "First paragraph."

    def test_cancelled_delivery_skips_tts(self, tmp_path: Path) -> None:
        """TTS fan-out should not fire when delivery was cancelled."""
        reply = "One.\n\nTwo.\n\nThree."
        familiar = _make_familiar(tmp_path, reply=reply)
        # install a TTS client so fan-out would fire otherwise
        tts_client = MagicMock()
        tts_client.synthesize = AsyncMock(
            return_value=TTSResult(audio=b"aud", timestamps=[]),
        )
        familiar.tts_client = tts_client
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )
        familiar.subscriptions.add(
            channel_id=9000,
            kind=SubscriptionKind.voice,
            guild_id=999,
        )
        familiar.channel_configs.set_mode(
            channel_id=12345,
            mode=ChannelMode.text_conversation_rp,
        )
        channel = _make_channel(12345)
        # attach a guild w/ voice_client so TTS branch would be entered
        guild = MagicMock()
        guild.id = 999
        vc = MagicMock()
        vc.is_playing = MagicMock(return_value=False)
        vc.play = MagicMock()
        guild.voice_client = vc
        type(channel).guild = PropertyMock(return_value=guild)

        buffer = [BufferedMessage(speaker="Alice", text="hi", timestamp=0.0)]

        send_gate = asyncio.Event()
        sends_done = 0

        async def fake_send(text: str) -> None:  # noqa: ARG001
            nonlocal sends_done
            sends_done += 1
            if sends_done == 1:
                send_gate.set()
                await asyncio.sleep(10.0)

        channel.send = AsyncMock(side_effect=fake_send)

        async def runner() -> None:
            task = asyncio.create_task(
                _run_text_response(
                    channel_id=12345,
                    guild_id=999,
                    speaker="Alice",
                    utterance="hi",
                    buffer=buffer,
                    familiar=familiar,
                    channel=channel,
                )
            )
            await send_gate.wait()
            tracker = familiar.text_delivery_registry.get(12345)
            await tracker.cancel_and_wait()
            await task

        asyncio.run(runner())

        # TTS must not have fired because delivery was cancelled
        tts_client.synthesize.assert_not_called()
        vc.play.assert_not_called()


class TestOnMessageCancellationHook:
    """on_message cancels any in-flight text delivery before routing."""

    def test_on_message_cancels_active_tracker(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )

        # Install a running task in the tracker; on_message should cancel it.
        tracker = familiar.text_delivery_registry.get(12345)
        started = asyncio.Event()
        was_cancelled = False

        async def long_task() -> None:
            nonlocal was_cancelled
            started.set()
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                was_cancelled = True
                raise

        async def runner() -> None:
            nonlocal was_cancelled
            task = asyncio.create_task(long_task())
            tracker.start(task)
            await started.wait()

            # stub monitor so no LLM call happens
            mock = AsyncMock()
            familiar.monitor.on_message = mock  # ty: ignore[invalid-assignment]

            msg = _make_message(channel_id=12345)
            await on_message(msg, familiar)
            assert was_cancelled is True
            # monitor still called after cancel
            mock.assert_called_once()

        asyncio.run(runner())

    def test_on_message_does_not_cancel_idle_tracker(
        self,
        tmp_path: Path,
    ) -> None:
        familiar = _make_familiar(tmp_path)
        familiar.subscriptions.add(
            channel_id=12345,
            kind=SubscriptionKind.text,
            guild_id=999,
        )

        mock = AsyncMock()
        familiar.monitor.on_message = mock  # ty: ignore[invalid-assignment]

        msg = _make_message(channel_id=12345)
        # no active tracker; on_message should flow through without error
        asyncio.run(on_message(msg, familiar))
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------


class TestShowContext:
    def test_sends_public_message(self, tmp_path: Path) -> None:
        """Works without a subscription; short context fits in one followup message."""
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=12345)

        fake_msg = MagicMock()
        fake_msg.author.bot = False
        fake_msg.author.display_name = "Alice"
        fake_msg.content = "hello there"

        async def _history(**_kwargs: object):  # noqa: RUF029
            yield fake_msg

        ctx.channel.history = _history

        asyncio.run(show_context(ctx, familiar))

        ctx.defer.assert_awaited_once_with(ephemeral=False)
        ctx.followup.send.assert_awaited_once()
        text: str = ctx.followup.send.call_args[0][0]
        assert "[0]" in text or "system" in text.lower()

    def test_sends_file_when_large(self, tmp_path: Path) -> None:
        """Oversized context is delivered as a context.md file attachment."""
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=12345)

        fake_msg = MagicMock()
        fake_msg.author.bot = False
        fake_msg.author.display_name = "Alice"
        fake_msg.content = "x" * 5000  # forces oversized output

        async def _history(**_kwargs: object):  # noqa: RUF029
            yield fake_msg

        ctx.channel.history = _history

        asyncio.run(show_context(ctx, familiar))

        ctx.followup.send.assert_awaited_once()
        call_kwargs = ctx.followup.send.call_args[1]
        assert "file" in call_kwargs
        assert call_kwargs["file"].filename == "context.md"
