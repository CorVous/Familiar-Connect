"""Conversation monitor — gates whether and when the familiar speaks.

Implements the ConversationMonitor described in
``future-features/conversation-flow.md``. The monitor is the single
entry point ``bot.py`` calls instead of going straight to the
context pipeline.

Three triggers, one evaluation path:

1. **Direct address** — name/alias/@mention detected → immediate side-model
   evaluation. State always resets after a direct-address evaluation,
   regardless of YES/NO.
2. **Interjection check** — message counter reaches the current threshold
   → side-model evaluated. On YES: respond + reset. On NO: advance the
   step-down curve (next threshold decreases by 3, floor 3).
3. **Lull** — no new message for ``lull_timeout`` seconds → side-model
   evaluated. On YES: respond + reset. On NO: wait for next trigger.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.config import Interjection
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BufferedMessage:
    """A single message accumulated in a channel buffer.

    :param speaker: Sanitised display name of the author.
    :param text: Raw message content.
    :param timestamp: ``time.monotonic()`` value at receipt.
    """

    speaker: str
    text: str
    timestamp: float


@dataclass
class ChannelBuffer:
    """Per-channel state owned by :class:`ConversationMonitor`.

    :param buffer: Messages since the bot last responded (or since
        the channel was first seen).
    :param message_counter: Total messages since last response.
        Drives the interjection check schedule.
    :param check_count: How many interjection checks have fired since
        the last response. Drives the step-down curve.
    :param next_interjection_at: Absolute ``message_counter`` value at
        which the next interjection check fires. Set to the tier's
        starting interval on construction/reset.
    :param lock: Per-channel asyncio lock preventing simultaneous
        evaluations.
    :param lull_timer_handle: Handle to the pending lull callback,
        cancelled and reset on every new message.
    """

    buffer: list[BufferedMessage] = field(default_factory=list)
    message_counter: int = 0
    check_count: int = 0
    next_interjection_at: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lull_timer_handle: asyncio.TimerHandle | None = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def is_direct_address(
    text: str,
    familiar_name: str,
    aliases: list[str],
    *,
    is_mention: bool,
) -> bool:
    """Return True if *text* contains a direct reference to the familiar.

    :param text: Message content to scan.
    :param familiar_name: The familiar's primary name (its folder id).
    :param aliases: Additional names to match.
    :param is_mention: Whether the Discord message already includes an
        @mention of the bot user. When True, this function returns True
        immediately without scanning the text.
    """
    if is_mention:
        return True
    for name in [familiar_name, *aliases]:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            return True
    return False


def _interjection_interval(tier: Interjection, check_count: int) -> int:
    """Return the message interval for the next interjection check.

    The interval shrinks by 3 after each declined check, flooring at 3.

    :param tier: The familiar's :class:`Interjection` setting.
    :param check_count: How many interjection checks have already fired
        since the last response.
    """
    return max(3, tier.starting_interval - check_count * 3)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_EVALUATION_SYSTEM_PROMPT = """\
You are a conversation monitor. Your ONLY job is to decide whether \
{familiar_name} should respond. Reply with EXACTLY one word: YES or NO. \
Do not reply in character. Do not explain your reasoning. \
Do not include punctuation. One word only."""

_LULL_PROMPT = """\
Character: {familiar_name}

{character_card}

Conversational personality: {chattiness}

Recent conversation summary:
{conversation_summary}

Recent messages:
{buffer}

There has been a pause in the conversation. Should {familiar_name} respond?"""

_DIRECT_ADDRESS_PROMPT = """\
Character: {familiar_name}

{character_card}

Conversational personality: {chattiness}

Recent conversation summary:
{conversation_summary}

Recent messages:
{buffer}

{familiar_name} was directly addressed. Should {familiar_name} respond?"""

_INTERJECTION_PROMPT = """\
Character: {familiar_name}

{character_card}

Conversational personality: {chattiness}

Recent conversation summary:
{conversation_summary}

Recent messages:
{buffer}

{message_count} messages have been said without {familiar_name} speaking. \
Should {familiar_name} interject?"""


def _format_buffer(buffer: list[BufferedMessage]) -> str:
    return "\n".join(f"{m.speaker}: {m.text}" for m in buffer)


# ---------------------------------------------------------------------------
# ConversationMonitor
# ---------------------------------------------------------------------------


class ConversationMonitor:
    """Gates whether and when the familiar speaks.

    Owned by :class:`Familiar` and called by ``bot.py`` for every
    message on a subscribed channel. Internally manages per-channel
    :class:`ChannelBuffer` state, evaluates the three triggers, and
    invokes the ``on_respond`` callback when the interjection slot
    says YES.

    :param familiar_name: Primary name (folder id) of the familiar.
    :param aliases: Additional names that trigger direct-address detection.
    :param chattiness: Free-text personality trait injected into the
        evaluation prompt.
    :param interjection: Tier controlling the interjection check schedule.
    :param lull_timeout: Seconds of silence before the lull evaluation fires.
    :param llm_client: :class:`LLMClient` for the
        ``interjection_decision`` slot, used for YES/NO evaluation.
    :param character_card: Pre-loaded text from the familiar's
        ``memory/self/`` files, injected into every evaluation prompt.
    :param on_respond: Async callback invoked when the LLM says YES.
        Receives ``(channel_id, buffer_snapshot)``.
    """

    def __init__(
        self,
        familiar_name: str,
        aliases: list[str],
        chattiness: str,
        interjection: Interjection,
        lull_timeout: float,
        llm_client: LLMClient,
        character_card: str,
        on_respond: Callable[[int, list[BufferedMessage]], Awaitable[None]],
    ) -> None:
        self._familiar_name = familiar_name
        self._aliases = aliases
        self._chattiness = chattiness
        self._interjection = interjection
        self._lull_timeout = lull_timeout
        self._llm_client = llm_client
        self._character_card = character_card
        self.on_respond = on_respond
        self._buffers: dict[int, ChannelBuffer] = {}
        self._lull_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_message(
        self,
        channel_id: int,
        speaker: str,
        text: str,
        *,
        is_mention: bool,
    ) -> None:
        """Process an incoming message on a subscribed channel.

        Called by ``bot.py:on_message`` for every non-bot message on a
        subscribed text channel.

        :param channel_id: Discord channel id.
        :param speaker: Sanitised display name of the author.
        :param text: Raw message content.
        :param is_mention: Whether the Discord message @mentions the bot.
        """
        buf = self._get_or_create_buffer(channel_id)

        # 1. Cancel the existing lull timer (we'll restart it below)
        self._cancel_lull_timer(buf)

        # 2. Append to buffer and increment counter
        buf.buffer.append(
            BufferedMessage(speaker=speaker, text=text, timestamp=time.monotonic())
        )
        buf.message_counter += 1

        # 3. Direct address → evaluate immediately; reset state regardless of result
        if is_direct_address(
            text, self._familiar_name, self._aliases, is_mention=is_mention
        ):
            async with buf.lock:
                yes = await self._evaluate(
                    channel_id,
                    buf,
                    trigger_context=(
                        "You were directly addressed in the conversation."
                    ),
                )
                if yes:
                    await self._fire_respond(channel_id, buf)
                else:
                    # Still reset on direct address, per spec
                    self._reset_buffer(buf)
            return

        # 4. Interjection check
        if buf.message_counter >= buf.next_interjection_at:
            async with buf.lock:
                trigger = (
                    f"{buf.message_counter} messages have been said"
                    " without you speaking."
                )
                yes = await self._evaluate(channel_id, buf, trigger_context=trigger)
                if yes:
                    await self._fire_respond(channel_id, buf)
                    return
                # NO → advance step-down curve
                buf.check_count += 1
                buf.next_interjection_at += _interjection_interval(
                    self._interjection, buf.check_count
                )

        # 5. Start new lull timer
        self._start_lull_timer(channel_id, buf)

    def clear_channel(self, channel_id: int) -> None:
        """Remove all state for *channel_id*.

        Called by ``bot.py`` when a channel is unsubscribed. Cancels any
        pending lull timer so it doesn't fire after the unsubscribe.
        """
        buf = self._buffers.pop(channel_id, None)
        if buf is not None:
            self._cancel_lull_timer(buf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_buffer(self, channel_id: int) -> ChannelBuffer:
        if channel_id not in self._buffers:
            buf = ChannelBuffer(
                next_interjection_at=self._interjection.starting_interval,
            )
            self._buffers[channel_id] = buf
        return self._buffers[channel_id]

    def _cancel_lull_timer(self, buf: ChannelBuffer) -> None:
        if buf.lull_timer_handle is not None:
            buf.lull_timer_handle.cancel()
            buf.lull_timer_handle = None

    def _start_lull_timer(self, channel_id: int, buf: ChannelBuffer) -> None:
        loop = asyncio.get_event_loop()
        buf.lull_timer_handle = loop.call_later(
            self._lull_timeout,
            self._schedule_lull_evaluation,
            channel_id,
        )

    def _schedule_lull_evaluation(self, channel_id: int) -> None:
        """Sync callback from call_later — schedules the async evaluation."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run_lull_evaluation(channel_id))
        # Keep a strong reference so the task isn't garbage-collected before
        # it runs. The done-callback removes it from the set automatically.
        self._lull_tasks.add(task)
        task.add_done_callback(self._lull_tasks.discard)

    async def _run_lull_evaluation(self, channel_id: int) -> None:
        buf = self._buffers.get(channel_id)
        if buf is None or not buf.buffer:
            return
        async with buf.lock:
            yes = await self._evaluate(channel_id, buf, trigger_context=None)
            if yes:
                await self._fire_respond(channel_id, buf)

    async def _evaluate(
        self,
        channel_id: int,
        buf: ChannelBuffer,
        trigger_context: str | None,
    ) -> bool:
        """Call the interjection LLM and return True if it responds YES."""
        if trigger_context is None:
            trigger_label = "lull"
        elif trigger_context.startswith("You were directly"):
            trigger_label = "direct_address"
        else:
            trigger_label = "interjection"

        _logger.debug(
            "evaluate channel=%s trigger=%s msgs=%d buffer_len=%d",
            channel_id,
            trigger_label,
            buf.message_counter,
            len(buf.buffer),
        )

        buffer_text = _format_buffer(buf.buffer)
        fmt = {
            "familiar_name": self._familiar_name,
            "character_card": self._character_card,
            "chattiness": self._chattiness,
            "conversation_summary": "",
            "buffer": buffer_text,
        }
        if trigger_label == "lull":
            user_prompt = _LULL_PROMPT.format(**fmt)
        elif trigger_label == "direct_address":
            user_prompt = _DIRECT_ADDRESS_PROMPT.format(**fmt)
        else:
            user_prompt = _INTERJECTION_PROMPT.format(
                **fmt,
                message_count=buf.message_counter,
            )

        system_prompt = _EVALUATION_SYSTEM_PROMPT.format(
            familiar_name=self._familiar_name,
        )

        try:
            reply = await self._llm_client.chat(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
            )
        except Exception:
            _logger.exception(
                "Interjection LLM evaluation failed for channel %s",
                channel_id,
            )
            return False

        response = reply.content
        # Check the first token for an affirmative. The system prompt
        # asks for exactly one word, but some models add punctuation
        # or a short prefix — stripping to the first token is robust
        # against both.
        stripped = response.strip()
        first_word = stripped.split()[0].upper().rstrip(".,!") if stripped else ""
        decision = first_word == "YES"
        _logger.info(
            "interjection channel=%s trigger=%s decision=%s msgs=%d raw=%r",
            channel_id,
            trigger_label,
            "YES" if decision else "NO",
            buf.message_counter,
            response[:120],
        )
        return decision

    async def _fire_respond(self, channel_id: int, buf: ChannelBuffer) -> None:
        """Invoke on_respond with a snapshot of the buffer, then reset state."""
        snapshot = list(buf.buffer)
        self._reset_buffer(buf)
        await self.on_respond(channel_id, snapshot)

    def _reset_buffer(self, buf: ChannelBuffer) -> None:
        """Reset all per-channel state after a response (or direct address)."""
        self._cancel_lull_timer(buf)
        buf.buffer.clear()
        buf.message_counter = 0
        buf.check_count = 0
        buf.next_interjection_at = self._interjection.starting_interval
