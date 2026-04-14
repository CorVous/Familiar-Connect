"""Conversation monitor — gates whether and when the familiar speaks.

Three triggers, one evaluation path:

1. **Direct address** — name/alias/@mention → immediate evaluation;
   state resets regardless of YES/NO.
2. **Interjection** — counter hits threshold → evaluate; on NO advance
   step-down curve (threshold shrinks by 3, floor 3).
3. **Lull** — no message for ``lull_timeout`` seconds → evaluate.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.config import Interjection
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)


class ResponseTrigger(Enum):
    """Which gate fired: direct_address, interjection, or lull.

    Voice interruption state machine uses ``is_unsolicited`` to bias
    interrupt-tolerance RNG toward pushing through self-started remarks.
    """

    direct_address = "direct_address"
    interjection = "interjection"
    lull = "lull"

    @property
    def is_unsolicited(self) -> bool:
        """True when the familiar chose to speak without being addressed."""
        return self is not ResponseTrigger.direct_address


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BufferedMessage:
    """Single message accumulated in a channel buffer."""

    speaker: str
    text: str
    timestamp: float


@dataclass
class ChannelBuffer:
    """Per-channel state owned by :class:`ConversationMonitor`."""

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
    """Check whether *text* contains a direct address to the familiar.

    Word-boundary-aware, case-insensitive. Returns immediately on
    ``is_mention=True``.
    """
    if is_mention:
        return True
    for name in [familiar_name, *aliases]:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            return True
    return False


def _interjection_interval(tier: Interjection, check_count: int) -> int:
    """Message interval for next interjection check.

    Shrinks by 3 after each declined check, flooring at 3.
    """
    return max(3, tier.starting_interval - check_count * 3)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_EVALUATION_SYSTEM_PROMPT = """\
You are a conversation monitor. Your ONLY job is to decide whether \
{familiar_name} should respond. Reply with exactly two lines:
Line 1: One brief sentence in {familiar_name}'s voice explaining why.
Line 2: YES or NO (one word, no punctuation)"""

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

    Per-channel :class:`ChannelBuffer` state, three triggers,
    ``on_respond(channel_id, snapshot, trigger)`` callback.
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
        on_respond: Callable[
            [int, list[BufferedMessage], ResponseTrigger],
            Awaitable[None],
        ],
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
        is_lull_endpoint: bool = False,
    ) -> None:
        """Process incoming message on subscribed channel.

        When ``is_lull_endpoint=True`` (voice path), caller already
        debounced silence; monitor evaluates lull inline instead of
        starting its own timer.
        """
        buf = self._get_or_create_buffer(channel_id)

        # 1. cancel existing lull timer (restarted below on text path)
        self._cancel_lull_timer(buf)

        # 2. append to buffer and increment counter
        buf.buffer.append(
            BufferedMessage(speaker=speaker, text=text, timestamp=time.monotonic())
        )
        buf.message_counter += 1

        # 3. direct address → immediate evaluation; reset regardless of result
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
                    await self._fire_respond(
                        channel_id, buf, ResponseTrigger.direct_address
                    )
                else:
                    # still reset on direct address, per spec
                    self._reset_buffer(buf)
            return

        # 4. interjection check
        if buf.message_counter >= buf.next_interjection_at:
            async with buf.lock:
                trigger = (
                    f"{buf.message_counter} messages have been said"
                    " without you speaking."
                )
                yes = await self._evaluate(channel_id, buf, trigger_context=trigger)
                if yes:
                    await self._fire_respond(
                        channel_id, buf, ResponseTrigger.interjection
                    )
                    return
                # no → advance step-down curve
                buf.check_count += 1
                buf.next_interjection_at += _interjection_interval(
                    self._interjection, buf.check_count
                )

        # 5. conversational lull
        if is_lull_endpoint:
            # voice path: caller already debounced silence, so this
            # call is itself the lull — evaluate inline without a timer
            async with buf.lock:
                yes = await self._evaluate(channel_id, buf, trigger_context=None)
                if yes:
                    await self._fire_respond(channel_id, buf, ResponseTrigger.lull)
        else:
            # text path: wait ``lull_timeout`` seconds for more messages
            self._start_lull_timer(channel_id, buf)

    def clear_channel(self, channel_id: int) -> None:
        """Remove all state for *channel_id*, cancelling pending lull timer."""
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
        """Sync callback from call_later — schedules async evaluation."""
        _logger.info("conversational lull expired channel=%s", channel_id)
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run_lull_evaluation(channel_id))
        # strong ref prevents GC before completion; done-callback cleans up
        self._lull_tasks.add(task)
        task.add_done_callback(self._lull_tasks.discard)

    async def _run_lull_evaluation(self, channel_id: int) -> None:
        buf = self._buffers.get(channel_id)
        if buf is None or not buf.buffer:
            return
        async with buf.lock:
            yes = await self._evaluate(channel_id, buf, trigger_context=None)
            if yes:
                await self._fire_respond(channel_id, buf, ResponseTrigger.lull)

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
        # first line: in-character reason; last line: YES/NO
        # single-line fallback: treat as decision only (empty reason)
        # robust against models that add punctuation despite system prompt
        stripped = response.strip()
        lines = stripped.splitlines()
        if len(lines) >= 2:
            reason = lines[0].strip()
            last_line = lines[-1]
        else:
            reason = ""
            last_line = lines[0] if lines else ""
        last_word = (
            last_line.split()[-1].upper().rstrip(".,!") if last_line.split() else ""
        )
        decision = last_word == "YES"
        _logger.info(
            "interjection channel=%s trigger=%s decision=%s msgs=%d reason=%r",
            channel_id,
            trigger_label,
            "YES" if decision else "NO",
            buf.message_counter,
            reason,
        )
        return decision

    async def _fire_respond(
        self,
        channel_id: int,
        buf: ChannelBuffer,
        trigger: ResponseTrigger,
    ) -> None:
        """Invoke on_respond with a snapshot of the buffer, then reset state."""
        snapshot = list(buf.buffer)
        self._reset_buffer(buf)
        await self.on_respond(channel_id, snapshot, trigger)

    def _reset_buffer(self, buf: ChannelBuffer) -> None:
        """Reset all per-channel state after a response (or direct address)."""
        self._cancel_lull_timer(buf)
        buf.buffer.clear()
        buf.message_counter = 0
        buf.check_count = 0
        buf.next_interjection_at = self._interjection.starting_interval
