"""Tests for text-channel session support in the bot."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import discord
import pytest

from familiar_connect.bot import awaken, create_bot, on_message, sleep_cmd
from familiar_connect.llm import Message
from familiar_connect.text_session import (
    TextSession,
    clear_session,
    get_session,
    set_session,
)


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
) -> MagicMock:
    """Fake discord.Message for on_message tests."""
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

    return msg


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
