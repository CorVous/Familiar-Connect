"""Tests for Discord bot creation and slash command handlers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import httpx
import pytest

from familiar_connect.bot import awaken, create_bot, sleep_cmd
from familiar_connect.llm import Message
from familiar_connect.text_session import clear_session, get_session
from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice_pipeline import clear_pipeline, set_pipeline


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
        mock_vc.start_recording = MagicMock()
        mock_vc.stop_recording = MagicMock()
        mock_vc.recording = False
        channel.connect = AsyncMock(return_value=mock_vc)
        channel.name = "General"
        # Channel members for user_names mapping
        member1 = MagicMock(spec=discord.Member)
        member1.id = 111
        member1.display_name = "Alice"
        member2 = MagicMock(spec=discord.Member)
        member2.id = 222
        member2.display_name = "Bob"
        channel.members = [member1, member2]
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


# --- Voice transcription pipeline integration ---


class TestAwakenVoiceTranscription:
    def test_starts_recording_when_transcriber_available(self) -> None:
        """When a transcriber is provided, awaken starts recording on the vc."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()

        with patch("familiar_connect.bot.start_pipeline", new_callable=AsyncMock):
            asyncio.run(awaken(ctx, transcriber=transcriber))

        joined_vc = ctx.author.voice.channel.connect.return_value
        joined_vc.start_recording.assert_called_once()

    def test_starts_pipeline_when_transcriber_available(self) -> None:
        """When a transcriber is provided, awaken starts the voice pipeline."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()

        with patch(
            "familiar_connect.bot.start_pipeline", new_callable=AsyncMock
        ) as mock_start:
            asyncio.run(awaken(ctx, transcriber=transcriber))

        mock_start.assert_awaited_once()

    def test_passes_user_names_from_channel_members(self) -> None:
        """start_pipeline receives a user_names dict built from channel members."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()

        with patch(
            "familiar_connect.bot.start_pipeline", new_callable=AsyncMock
        ) as mock_start:
            asyncio.run(awaken(ctx, transcriber=transcriber))

        _, kwargs = mock_start.call_args
        user_names = kwargs.get("user_names") or mock_start.call_args[0][1]
        assert user_names == {111: "Alice", 222: "Bob"}

    def test_skips_recording_when_no_transcriber(self) -> None:
        """Without a transcriber, awaken does NOT start recording."""
        ctx = _make_ctx(in_voice=True)

        asyncio.run(awaken(ctx))

        joined_vc = ctx.author.voice.channel.connect.return_value
        joined_vc.start_recording.assert_not_called()


# --- /sleep with pipeline teardown ---


class TestSleepPipelineTeardown:
    def test_sleep_stops_recording_and_pipeline(self) -> None:
        """When a pipeline is active, /sleep stops recording and pipeline."""
        ctx = _make_ctx(in_voice=True, bot_connected=True)
        # Simulate active recording
        ctx.voice_client.recording = True

        # Put a mock pipeline in the registry
        mock_pipeline = MagicMock()
        set_pipeline(mock_pipeline)

        with patch(
            "familiar_connect.bot.stop_pipeline", new_callable=AsyncMock
        ) as mock_stop:
            asyncio.run(sleep_cmd(ctx))

        ctx.voice_client.stop_recording.assert_called_once()
        mock_stop.assert_awaited_once()

    def test_sleep_works_without_active_pipeline(self) -> None:
        """When no pipeline is active, /sleep still disconnects normally."""
        ctx = _make_ctx(in_voice=True, bot_connected=True)

        with patch(
            "familiar_connect.bot.stop_pipeline", new_callable=AsyncMock
        ) as mock_stop:
            asyncio.run(sleep_cmd(ctx))

        ctx.voice_client.disconnect.assert_called_once()
        mock_stop.assert_not_awaited()


# --- Voice response handler wiring ---


class TestAwakenVoiceResponseHandler:
    def test_passes_response_handler_to_pipeline(self) -> None:
        """start_pipeline gets a response_handler when llm + transcriber set."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()
        llm_client = MagicMock()

        with patch(
            "familiar_connect.bot.start_pipeline", new_callable=AsyncMock
        ) as mock_start:
            asyncio.run(
                awaken(
                    ctx,
                    transcriber=transcriber,
                    llm_client=llm_client,
                    system_prompt="You are a familiar.",
                )
            )

        _, kwargs = mock_start.call_args
        assert kwargs.get("response_handler") is not None
        assert callable(kwargs["response_handler"])

    def test_no_response_handler_without_llm_client(self) -> None:
        """Without an llm_client, no response_handler is passed to start_pipeline."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()

        with patch(
            "familiar_connect.bot.start_pipeline", new_callable=AsyncMock
        ) as mock_start:
            asyncio.run(awaken(ctx, transcriber=transcriber))

        _, kwargs = mock_start.call_args
        assert kwargs.get("response_handler") is None

    def test_response_handler_calls_llm_and_plays_tts(self) -> None:
        """The response handler calls LLM chat, synthesizes TTS, and plays audio."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()
        llm_client = MagicMock()
        llm_reply = Message(role="assistant", content="Greetings!")
        llm_client.chat = AsyncMock(return_value=llm_reply)
        tts_client = _make_tts_client(pcm=bytes(8))

        captured_handler = None

        async def _capture_start(  # noqa: RUF029
            *args: object,  # noqa: ARG001
            **kwargs: object,
        ) -> MagicMock:
            nonlocal captured_handler
            captured_handler = kwargs.get("response_handler")
            mock_pipeline = MagicMock()
            mock_pipeline.tagged_audio_queue = asyncio.Queue()
            return mock_pipeline

        with patch("familiar_connect.bot.start_pipeline", side_effect=_capture_start):
            asyncio.run(
                awaken(
                    ctx,
                    transcriber=transcriber,
                    llm_client=llm_client,
                    tts_client=tts_client,
                    system_prompt="You are a familiar.",
                )
            )

        assert captured_handler is not None

        # Now invoke the handler with a transcription result
        result = TranscriptionResult(
            text="Hello familiar", is_final=True, start=0.0, end=1.0
        )
        asyncio.run(captured_handler(111, result))

        llm_client.chat.assert_awaited_once()
        tts_client.synthesize.assert_awaited()
        # TTS called once for greeting + once for response
        joined_vc = ctx.author.voice.channel.connect.return_value
        assert joined_vc.play.call_count >= 2  # greeting + response

    def test_response_handler_works_without_tts(self) -> None:
        """The response handler works even without a TTS client — just calls LLM."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()
        llm_client = MagicMock()
        llm_reply = Message(role="assistant", content="I hear you.")
        llm_client.chat = AsyncMock(return_value=llm_reply)

        captured_handler = None

        async def _capture_start(  # noqa: RUF029
            *args: object,  # noqa: ARG001
            **kwargs: object,
        ) -> MagicMock:
            nonlocal captured_handler
            captured_handler = kwargs.get("response_handler")
            mock_pipeline = MagicMock()
            mock_pipeline.tagged_audio_queue = asyncio.Queue()
            return mock_pipeline

        with patch("familiar_connect.bot.start_pipeline", side_effect=_capture_start):
            asyncio.run(
                awaken(
                    ctx,
                    transcriber=transcriber,
                    llm_client=llm_client,
                    system_prompt="You are a familiar.",
                )
            )

        assert captured_handler is not None

        result = TranscriptionResult(text="Hello", is_final=True, start=0.0, end=1.0)
        asyncio.run(captured_handler(111, result))

        llm_client.chat.assert_awaited_once()

    def test_response_handler_waits_for_audio_to_finish(self) -> None:
        """Handler polls is_playing() and waits before calling play()."""
        ctx = _make_ctx(in_voice=True)
        transcriber = MagicMock()
        llm_client = MagicMock()
        llm_client.chat = AsyncMock(
            return_value=Message(role="assistant", content="Reply")
        )
        tts_client = _make_tts_client(pcm=bytes(8))

        captured_handler = None

        async def _capture_start(  # noqa: RUF029
            *args: object,  # noqa: ARG001
            **kwargs: object,
        ) -> MagicMock:
            nonlocal captured_handler
            captured_handler = kwargs.get("response_handler")
            mock_pipeline = MagicMock()
            mock_pipeline.tagged_audio_queue = asyncio.Queue()
            return mock_pipeline

        with patch("familiar_connect.bot.start_pipeline", side_effect=_capture_start):
            asyncio.run(
                awaken(
                    ctx,
                    transcriber=transcriber,
                    llm_client=llm_client,
                    tts_client=tts_client,
                    system_prompt="Test",
                )
            )

        assert captured_handler is not None

        joined_vc = ctx.author.voice.channel.connect.return_value
        # Simulate: audio is playing on first check, finishes on second
        joined_vc.is_playing = MagicMock(side_effect=[True, False])

        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=1.0)
        asyncio.run(captured_handler(111, r1))

        # is_playing must have been polled (proves the handler waits)
        assert joined_vc.is_playing.call_count >= 2
