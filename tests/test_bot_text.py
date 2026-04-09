"""Tests for text-channel session support in the bot."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import discord
import httpx
import pytest

from familiar_connect.bot import awaken, create_bot, on_message, sleep_cmd
from familiar_connect.llm import Message
from familiar_connect.text_session import (
    TextSession,
    clear_session,
    get_session,
    set_session,
)
from familiar_connect.voice_pipeline import clear_pipeline


@pytest.fixture(autouse=True)
def _fresh_loop_and_session():
    """Fresh asyncio event loop and cleared session/pipeline per test.

    Yields:
        None

    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    clear_session()
    clear_pipeline()
    yield
    clear_session()
    clear_pipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_ctx(*, bot_voice_connected: bool = False) -> MagicMock:
    """ApplicationContext whose channel is a TextChannel (no voice state)."""
    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.respond = AsyncMock()
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()

    # Author is a Member but NOT in a voice channel
    author = MagicMock(spec=discord.Member)
    type(ctx).author = PropertyMock(return_value=author)
    type(author).voice = PropertyMock(return_value=None)

    # Channel is a TextChannel
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = 12345
    type(ctx).channel = PropertyMock(return_value=channel)
    type(ctx).channel_id = PropertyMock(return_value=12345)

    if bot_voice_connected:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.disconnect = AsyncMock()
        type(ctx).voice_client = PropertyMock(return_value=vc)
    else:
        type(ctx).voice_client = PropertyMock(return_value=None)

    return ctx


def _make_message(
    content: str = "hello",
    *,
    author_bot: bool = False,
    channel_id: int = 12345,
    guild_voice_client: MagicMock | None = None,
    in_guild: bool = True,
) -> MagicMock:
    """Fake discord.Message for on_message tests.

    :param guild_voice_client: If provided, the message's guild will have this
        as its voice_client (simulating the bot being in a voice channel).
    :param in_guild: If False, msg.guild is None (simulates a DM).
    """
    msg = MagicMock(spec=discord.Message)
    msg.content = content
    msg.channel = MagicMock(spec=discord.TextChannel)
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    # typing() must be an async context manager
    typing_cm = MagicMock()
    typing_cm.__aenter__ = AsyncMock(return_value=None)
    typing_cm.__aexit__ = AsyncMock(return_value=False)
    msg.channel.typing = MagicMock(return_value=typing_cm)

    author = MagicMock(spec=discord.Member)
    author.bot = author_bot
    author.display_name = "Alice"
    msg.author = author

    if in_guild:
        guild = MagicMock(spec=discord.Guild)
        guild.voice_client = guild_voice_client
        msg.guild = guild
    else:
        msg.guild = None

    return msg


def _make_tts_client(pcm: bytes = b"\x00\x01\x02\x03") -> MagicMock:
    """Fake CartesiaTTSClient that returns *pcm* bytes from synthesize()."""
    client = MagicMock()
    client.synthesize = AsyncMock(return_value=pcm)
    return client


def _make_voice_client(*, is_playing: bool = False) -> MagicMock:
    """Fake discord.VoiceClient."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing = MagicMock(return_value=is_playing)
    vc.play = MagicMock()
    return vc


def _make_llm_client(reply: str = "I am here.") -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value=Message(role="assistant", content=reply))
    return client


# ---------------------------------------------------------------------------
# /awaken in a text channel
# ---------------------------------------------------------------------------


class TestAwakenTextChannel:
    def test_awaken_text_channel_creates_session(self) -> None:
        ctx = _make_text_ctx()
        asyncio.run(awaken(ctx, system_prompt="You are Aria."))
        session = get_session()
        assert session is not None
        assert session.channel_id == 12345

    def test_awaken_text_channel_responds_confirmation(self) -> None:
        ctx = _make_text_ctx()
        asyncio.run(awaken(ctx, system_prompt="You are Aria."))
        ctx.respond.assert_called_once()

    def test_awaken_refused_when_text_session_already_active(self) -> None:
        set_session(TextSession(channel_id=999, system_prompt=""))
        ctx = _make_text_ctx()
        asyncio.run(awaken(ctx, system_prompt="You are Aria."))
        ctx.respond.assert_called_once()
        call_kwargs = ctx.respond.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True
        # Session unchanged
        session = get_session()
        assert session is not None
        assert session.channel_id == 999

    def test_awaken_refused_when_voice_already_connected(self) -> None:
        ctx = _make_text_ctx(bot_voice_connected=True)
        asyncio.run(awaken(ctx, system_prompt="You are Aria."))
        ctx.respond.assert_called_once()
        call_kwargs = ctx.respond.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True


# ---------------------------------------------------------------------------
# /sleep with a text session
# ---------------------------------------------------------------------------


class TestSleepTextSession:
    def test_sleep_clears_text_session(self) -> None:
        set_session(TextSession(channel_id=12345, system_prompt=""))
        ctx = _make_text_ctx()
        asyncio.run(sleep_cmd(ctx))
        assert get_session() is None
        ctx.respond.assert_called_once()

    def test_sleep_responds_when_no_session_at_all(self) -> None:
        ctx = _make_text_ctx()
        asyncio.run(sleep_cmd(ctx))
        ctx.respond.assert_called_once()
        call_kwargs = ctx.respond.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True


# ---------------------------------------------------------------------------
# on_message handler
# ---------------------------------------------------------------------------


class TestOnMessage:
    def test_bot_messages_ignored(self) -> None:
        set_session(TextSession(channel_id=12345, system_prompt="sys"))
        llm = _make_llm_client()
        msg = _make_message(author_bot=True)
        asyncio.run(on_message(msg, llm))
        llm.chat.assert_not_called()

    def test_wrong_channel_ignored(self) -> None:
        set_session(TextSession(channel_id=12345, system_prompt="sys"))
        llm = _make_llm_client()
        msg = _make_message(channel_id=99999)
        asyncio.run(on_message(msg, llm))
        llm.chat.assert_not_called()

    def test_no_session_ignored(self) -> None:
        llm = _make_llm_client()
        msg = _make_message()
        asyncio.run(on_message(msg, llm))
        llm.chat.assert_not_called()

    def test_message_appended_to_history(self) -> None:
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client()
        msg = _make_message(content="Hello there.")
        asyncio.run(on_message(msg, llm))
        assert any(m.content == "Hello there." for m in session.history)

    def test_llm_called_with_system_and_history(self) -> None:
        session = TextSession(channel_id=12345, system_prompt="Be helpful.")
        set_session(session)
        llm = _make_llm_client()
        msg = _make_message(content="What is 2+2?")
        asyncio.run(on_message(msg, llm))

        llm.chat.assert_called_once()
        messages = llm.chat.call_args[0][0]
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful."
        assert any(m.content == "What is 2+2?" for m in messages)

    def test_reply_posted_to_channel(self) -> None:
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client(reply="Four!")
        msg = _make_message()
        asyncio.run(on_message(msg, llm))
        msg.channel.send.assert_called_once_with("Four!")

    def test_reply_appended_to_history(self) -> None:
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client(reply="I am here.")
        msg = _make_message(content="Hello.")
        asyncio.run(on_message(msg, llm))
        assert any(
            m.role == "assistant" and m.content == "I am here." for m in session.history
        )

    def test_speaker_name_set_on_user_message(self) -> None:
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client()
        msg = _make_message(content="Hi.")
        asyncio.run(on_message(msg, llm))
        user_msg = next(m for m in session.history if m.role == "user")
        assert user_msg.name == "Alice"


# ---------------------------------------------------------------------------
# create_bot accepts LLM client + system prompt
# ---------------------------------------------------------------------------


class TestCreateBotWithLLM:
    def test_create_bot_accepts_llm_and_prompt(self) -> None:
        llm = _make_llm_client()
        bot = create_bot(llm_client=llm, system_prompt="You are Aria.")
        assert isinstance(bot, discord.Bot)

    def test_create_bot_accepts_tts_client(self) -> None:
        """create_bot accepts an optional tts_client parameter."""
        llm = _make_llm_client()
        tts = _make_tts_client()
        bot = create_bot(llm_client=llm, system_prompt="You are Aria.", tts_client=tts)
        assert isinstance(bot, discord.Bot)


# ---------------------------------------------------------------------------
# on_message with TTS integration
# ---------------------------------------------------------------------------


class TestOnMessageWithTTS:
    def test_synthesize_called_when_in_voice_channel(self) -> None:
        """TTS is invoked with the LLM reply when a voice client is present."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client(reply="Hello!")
        tts = _make_tts_client()
        vc = _make_voice_client()
        msg = _make_message(guild_voice_client=vc)

        asyncio.run(on_message(msg, llm, tts_client=tts))

        tts.synthesize.assert_awaited_once_with("Hello!")

    def test_voice_client_plays_audio_after_synthesis(self) -> None:
        """voice_client.play() is called once after TTS synthesis."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client(reply="Hello!")
        vc = _make_voice_client()
        tts = _make_tts_client(pcm=bytes(8))  # 4 mono samples
        msg = _make_message(guild_voice_client=vc)

        asyncio.run(on_message(msg, llm, tts_client=tts))

        vc.play.assert_called_once()

    def test_tts_not_called_when_no_voice_client(self) -> None:
        """TTS is skipped when the guild has no voice client."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client()
        tts = _make_tts_client()
        msg = _make_message(guild_voice_client=None)  # no voice client

        asyncio.run(on_message(msg, llm, tts_client=tts))

        tts.synthesize.assert_not_awaited()

    def test_tts_not_called_when_no_tts_client(self) -> None:
        """TTS is skipped when tts_client=None (default)."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client()
        vc = _make_voice_client()
        msg = _make_message(guild_voice_client=vc)

        asyncio.run(on_message(msg, llm))  # no tts_client

        vc.play.assert_not_called()

    def test_tts_skipped_when_already_playing(self) -> None:
        """TTS is skipped if the voice client is already playing audio."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client()
        tts = _make_tts_client()
        vc = _make_voice_client(is_playing=True)
        msg = _make_message(guild_voice_client=vc)

        asyncio.run(on_message(msg, llm, tts_client=tts))

        tts.synthesize.assert_not_awaited()

    def test_dm_message_skips_tts(self) -> None:
        """TTS is skipped for DMs (message.guild is None)."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client()
        tts = _make_tts_client()
        msg = _make_message(in_guild=False)

        asyncio.run(on_message(msg, llm, tts_client=tts))

        tts.synthesize.assert_not_awaited()

    def test_tts_error_does_not_break_text_reply(self) -> None:
        """A TTS failure is swallowed; the text reply is still sent."""
        session = TextSession(channel_id=12345, system_prompt="sys")
        set_session(session)
        llm = _make_llm_client(reply="Still here.")
        tts = _make_tts_client()
        tts.synthesize = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "https://api.cartesia.ai/tts/bytes"),
                response=httpx.Response(500),
            )
        )
        vc = _make_voice_client()
        msg = _make_message(guild_voice_client=vc)

        asyncio.run(on_message(msg, llm, tts_client=tts))

        msg.channel.send.assert_called_once_with("Still here.")
