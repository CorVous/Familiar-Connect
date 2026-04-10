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
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest

from familiar_connect.bot import (
    _run_text_response,
    create_bot,
    on_message,
    set_channel_mode,
    subscribe_my_voice,
    subscribe_text,
    unsubscribe_text,
    unsubscribe_voice,
)
from familiar_connect.chattiness import BufferedMessage
from familiar_connect.config import ChannelMode
from familiar_connect.familiar import Familiar
from familiar_connect.llm import LLMClient, Message
from familiar_connect.subscriptions import SubscriptionKind

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


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
        super().__init__(api_key="test-key")
        self.reply = reply
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(messages)
        return Message(role="assistant", content=self.reply)


def _make_familiar(tmp_path: Path, *, reply: str = "I am here.") -> Familiar:
    root = tmp_path / "aria"
    root.mkdir()
    llm = _StubLLMClient(reply=reply)
    return Familiar.load_from_disk(root, llm_client=llm)


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
) -> MagicMock:
    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.respond = AsyncMock()
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.name = "general"
    type(ctx).channel = PropertyMock(return_value=channel)
    type(ctx).channel_id = PropertyMock(return_value=channel_id)
    if guild_id is not None:
        guild = MagicMock(spec=discord.Guild)
        guild.id = guild_id
        type(ctx).guild_id = PropertyMock(return_value=guild_id)
        type(ctx).guild = PropertyMock(return_value=guild)
    else:
        type(ctx).guild_id = PropertyMock(return_value=None)
        type(ctx).guild = PropertyMock(return_value=None)
    return ctx


def _make_voice_ctx(
    *,
    channel_id: int = 9000,
    guild_id: int = 999,
    already_connected: bool = False,
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

        llm = familiar.llm_client
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
        ctx.respond.assert_called_once()

    def test_subscribe_text_persists_across_reload(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=42, guild_id=999)
        asyncio.run(subscribe_text(ctx, familiar))

        # Reload the Familiar bundle; the subscription should come back.
        reloaded = Familiar.load_from_disk(
            familiar.root,
            llm_client=_StubLLMClient(),
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


# ---------------------------------------------------------------------------
# /channel-* mode commands
# ---------------------------------------------------------------------------


class TestChannelModeCommands:
    def test_set_channel_mode_writes_sidecar(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=77)

        asyncio.run(set_channel_mode(ctx, familiar, ChannelMode.full_rp))

        assert familiar.channel_configs.get(channel_id=77).mode is ChannelMode.full_rp
        ctx.respond.assert_called_once()

    def test_set_channel_mode_persists_across_reload(self, tmp_path: Path) -> None:
        familiar = _make_familiar(tmp_path)
        ctx = _make_text_ctx(channel_id=77)
        asyncio.run(set_channel_mode(ctx, familiar, ChannelMode.imitate_voice))

        reloaded = Familiar.load_from_disk(
            familiar.root,
            llm_client=_StubLLMClient(),
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
