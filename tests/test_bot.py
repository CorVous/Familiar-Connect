"""Tests for Discord bot creation and slash command handlers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import discord
import httpx
import pytest

from familiar_connect.bot import awaken, create_bot, sleep_cmd
from familiar_connect.text_session import clear_session, get_session


@pytest.fixture(autouse=True)
def _fresh_loop_and_session():
    """Fresh asyncio event loop and cleared session registry per test.

    Yields:
        None

    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    clear_session()
    yield
    clear_session()


def _make_ctx(
    *,
    in_voice: bool = True,
    bot_connected: bool = False,
) -> MagicMock:
    """Build a mock ApplicationContext for slash command tests.

    :param in_voice: Whether the invoking user is in a voice channel
    :param bot_connected: Whether the bot already has a voice client
    """
    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.respond = AsyncMock()
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()

    # Author must be spec'd as Member so isinstance(author, discord.Member) passes
    author = MagicMock(spec=discord.Member)
    type(ctx).author = PropertyMock(return_value=author)

    if in_voice:
        voice_state = MagicMock()
        channel = MagicMock(spec=discord.VoiceChannel)
        # connect() returns a mock voice client; give it play() so greeting tests work
        mock_vc = MagicMock(spec=discord.VoiceClient)
        mock_vc.play = MagicMock()
        mock_vc.is_playing = MagicMock(return_value=False)
        channel.connect = AsyncMock(return_value=mock_vc)
        channel.name = "General"
        voice_state.channel = channel
        type(author).voice = PropertyMock(return_value=voice_state)
    else:
        type(author).voice = PropertyMock(return_value=None)

    if bot_connected:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.disconnect = AsyncMock()
        type(ctx).voice_client = PropertyMock(return_value=vc)
    else:
        type(ctx).voice_client = PropertyMock(return_value=None)

    return ctx


def _make_tts_client(pcm: bytes = b"\x00\x01\x02\x03") -> MagicMock:
    """Fake CartesiaTTSClient that returns *pcm* from synthesize()."""
    client = MagicMock()
    client.synthesize = AsyncMock(return_value=pcm)
    return client


# --- Bot factory tests ---


def test_create_bot_returns_discord_bot() -> None:
    """create_bot returns a discord.Bot instance."""
    bot = create_bot()
    assert isinstance(bot, discord.Bot)


def test_create_bot_has_awaken_command() -> None:
    """Bot has a registered slash command named 'awaken'."""
    bot = create_bot()
    names = [cmd.name for cmd in bot.pending_application_commands]
    assert "awaken" in names


def test_create_bot_has_sleep_command() -> None:
    """Bot has a registered slash command named 'sleep'."""
    bot = create_bot()
    names = [cmd.name for cmd in bot.pending_application_commands]
    assert "sleep" in names


# --- /awaken handler tests ---


def test_awaken_user_not_in_voice_channel_binds_text_channel() -> None:
    """When the user is not in a voice channel, /awaken binds to the text channel."""
    clear_session()
    ctx = _make_ctx(in_voice=False)

    asyncio.run(awaken(ctx))

    # Should succeed and create a text session
    ctx.respond.assert_called_once()
    assert get_session() is not None
    clear_session()


def test_awaken_already_connected() -> None:
    """When the bot is already in a voice channel, refuse with an ephemeral message."""
    ctx = _make_ctx(in_voice=True, bot_connected=True)

    asyncio.run(awaken(ctx))

    ctx.respond.assert_called_once()
    assert ctx.respond.call_args.kwargs.get("ephemeral") is True
    ctx.author.voice.channel.connect.assert_not_called()


def test_awaken_joins_channel() -> None:
    """When user is in voice and bot is not connected, join the channel."""
    ctx = _make_ctx(in_voice=True, bot_connected=False)

    asyncio.run(awaken(ctx))

    ctx.author.voice.channel.connect.assert_called_once()
    ctx.defer.assert_called_once()
    ctx.followup.send.assert_called_once()


# --- /sleep handler tests ---


def test_sleep_not_connected() -> None:
    """When the bot is not in a voice channel, respond with an error."""
    ctx = _make_ctx(in_voice=True, bot_connected=False)

    asyncio.run(sleep_cmd(ctx))

    ctx.respond.assert_called_once()
    call_kwargs = ctx.respond.call_args
    assert call_kwargs.kwargs.get("ephemeral") is True


def test_sleep_disconnects() -> None:
    """When the bot is connected, disconnect and confirm."""
    ctx = _make_ctx(in_voice=True, bot_connected=True)

    asyncio.run(sleep_cmd(ctx))

    ctx.voice_client.disconnect.assert_called_once()
    ctx.respond.assert_called_once()


# --- Opening greeting via TTS ---


class TestAwakenVoiceGreeting:
    def test_greeting_synthesized_after_joining(self) -> None:
        """After joining a voice channel, TTS synthesizes a greeting."""
        ctx = _make_ctx(in_voice=True)
        tts = _make_tts_client()

        asyncio.run(awaken(ctx, tts_client=tts))

        tts.synthesize.assert_awaited_once()

    def test_greeting_text_is_hello(self) -> None:
        """The greeting text passed to TTS is 'Hello!'."""
        ctx = _make_ctx(in_voice=True)
        tts = _make_tts_client()

        asyncio.run(awaken(ctx, tts_client=tts))

        tts.synthesize.assert_awaited_once_with("Hello!")

    def test_greeting_played_in_voice_channel(self) -> None:
        """The greeting audio is played on the voice client returned by connect()."""
        ctx = _make_ctx(in_voice=True)
        tts = _make_tts_client(pcm=bytes(8))

        asyncio.run(awaken(ctx, tts_client=tts))

        joined_vc = ctx.author.voice.channel.connect.return_value
        joined_vc.play.assert_called_once()

    def test_greeting_skipped_when_no_tts_client(self) -> None:
        """No TTS client means no greeting — bot still joins silently."""
        ctx = _make_ctx(in_voice=True)

        asyncio.run(awaken(ctx))  # tts_client defaults to None

        joined_vc = ctx.author.voice.channel.connect.return_value
        joined_vc.play.assert_not_called()

    def test_greeting_error_does_not_prevent_followup(self) -> None:
        """A TTS failure during the greeting doesn't block the join confirmation."""
        ctx = _make_ctx(in_voice=True)
        tts = _make_tts_client()
        tts.synthesize = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "https://api.cartesia.ai/tts/bytes"),
                response=httpx.Response(500),
            )
        )

        asyncio.run(awaken(ctx, tts_client=tts))

        ctx.followup.send.assert_called_once()
