"""Tests for the ConversationMonitor and its supporting components.

Covers familiar_connect.chattiness per the implementation plan in
future-features/conversation-flow.md.

Steps covered:
  2. is_direct_address() — pure word-boundary name/alias matching
  3. BufferedMessage and ChannelBuffer dataclasses
  4. _interjection_interval() step-down curve
  5. ConversationMonitor core — buffer management, counter, triggers
  6. Lull timer — start, reset, expiry
  7. Side-model evaluation wiring — YES/NO parsing
  8. on_respond callback invocation and state reset
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from familiar_connect.chattiness import (
    BufferedMessage,
    ChannelBuffer,
    ConversationMonitor,
    _interjection_interval,
    is_direct_address,
)
from familiar_connect.config import Interjection
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monitor(
    *,
    familiar_name: str = "aria",
    aliases: list[str] | None = None,
    chattiness: str = "Curious",
    interjection: Interjection = Interjection.average,
    lull_timeout: float = 100.0,  # large default so lull never fires unintentionally
    side_model_reply: str = "YES",
    on_respond: Callable[[int, list[BufferedMessage]], Awaitable[None]] | None = None,
) -> tuple[ConversationMonitor, list[tuple[int, list[BufferedMessage]]]]:
    """Build a ConversationMonitor with a stub ``interjection_decision`` client.

    Returns the monitor and a list that records every ``on_respond`` call as
    ``(channel_id, buffer_snapshot)``.
    """
    calls: list[tuple[int, list[BufferedMessage]]] = []

    async def _default_on_respond(  # noqa: RUF029
        channel_id: int, buffer: list[BufferedMessage]
    ) -> None:
        calls.append((channel_id, list(buffer)))

    llm_client = MagicMock()
    llm_client.chat = AsyncMock(
        return_value=Message(role="assistant", content=side_model_reply),
    )

    monitor = ConversationMonitor(
        familiar_name=familiar_name,
        aliases=aliases or [],
        chattiness=chattiness,
        interjection=interjection,
        lull_timeout=lull_timeout,
        llm_client=llm_client,
        character_card="You are Aria.",
        on_respond=on_respond if on_respond is not None else _default_on_respond,
    )
    return monitor, calls


# ---------------------------------------------------------------------------
# Step 2 — is_direct_address
# ---------------------------------------------------------------------------


class TestIsDirectAddress:
    def test_is_mention_always_true(self) -> None:
        assert is_direct_address("hello", "aria", [], is_mention=True)

    def test_familiar_name_match(self) -> None:
        assert is_direct_address(
            "Hey Aria, what do you think?", "aria", [], is_mention=False
        )

    def test_alias_match(self) -> None:
        assert is_direct_address("ari come here", "aria", ["ari"], is_mention=False)

    def test_case_insensitive(self) -> None:
        assert is_direct_address("ARIA how are you?", "aria", [], is_mention=False)

    def test_word_boundary_no_partial_match(self) -> None:
        assert not is_direct_address(
            "malaria is spreading", "aria", [], is_mention=False
        )

    def test_word_boundary_no_partial_alias(self) -> None:
        # alias "ari" should not match inside the word "familiar"
        assert not is_direct_address(
            "familiar is here", "aria", ["ari"], is_mention=False
        )

    def test_no_match_returns_false(self) -> None:
        assert not is_direct_address("hello world", "aria", ["ari"], is_mention=False)

    def test_name_at_end_of_sentence(self) -> None:
        assert is_direct_address("good morning, aria", "aria", [], is_mention=False)

    def test_multiple_aliases_checked(self) -> None:
        assert is_direct_address("hey bob", "aria", ["bob", "boo"], is_mention=False)

    def test_familiar_id_with_no_aliases(self) -> None:
        assert not is_direct_address("hey there", "aria", [], is_mention=False)

    def test_familiar_name_with_punctuation_boundary(self) -> None:
        assert is_direct_address("aria!", "aria", [], is_mention=False)


# ---------------------------------------------------------------------------
# Step 3 — BufferedMessage and ChannelBuffer
# ---------------------------------------------------------------------------


class TestBufferedMessage:
    def test_fields(self) -> None:
        msg = BufferedMessage(speaker="Alice", text="hello", timestamp=1.0)
        assert msg.speaker == "Alice"
        assert msg.text == "hello"
        assert msg.timestamp == 1.0  # noqa: RUF069


class TestChannelBuffer:
    def test_starts_empty(self) -> None:
        buf = ChannelBuffer()
        assert buf.buffer == []
        assert buf.message_counter == 0
        assert buf.check_count == 0
        assert buf.lull_timer_handle is None


# ---------------------------------------------------------------------------
# Step 4 — _interjection_interval step-down curve
# ---------------------------------------------------------------------------


class TestInterjectionInterval:
    def test_average_check_0(self) -> None:
        assert _interjection_interval(Interjection.average, 0) == 9

    def test_average_check_1(self) -> None:
        assert _interjection_interval(Interjection.average, 1) == 6

    def test_average_check_2(self) -> None:
        assert _interjection_interval(Interjection.average, 2) == 3

    def test_average_check_3_floors_at_3(self) -> None:
        assert _interjection_interval(Interjection.average, 3) == 3

    def test_average_check_100_floors_at_3(self) -> None:
        assert _interjection_interval(Interjection.average, 100) == 3

    def test_very_quiet_check_0(self) -> None:
        # Spec: check 1 fires at message 15
        assert _interjection_interval(Interjection.very_quiet, 0) == 15

    def test_very_quiet_check_1(self) -> None:
        assert _interjection_interval(Interjection.very_quiet, 1) == 12

    def test_very_quiet_check_2(self) -> None:
        assert _interjection_interval(Interjection.very_quiet, 2) == 9

    def test_very_quiet_check_3(self) -> None:
        assert _interjection_interval(Interjection.very_quiet, 3) == 6

    def test_very_quiet_check_4(self) -> None:
        assert _interjection_interval(Interjection.very_quiet, 4) == 3

    def test_very_quiet_check_5_floors_at_3(self) -> None:
        assert _interjection_interval(Interjection.very_quiet, 5) == 3

    def test_very_eager_always_3(self) -> None:
        for check in range(5):
            assert _interjection_interval(Interjection.very_eager, check) == 3

    def test_eager_check_0(self) -> None:
        assert _interjection_interval(Interjection.eager, 0) == 6

    def test_eager_check_1_floors(self) -> None:
        assert _interjection_interval(Interjection.eager, 1) == 3


# ---------------------------------------------------------------------------
# Step 5 — ConversationMonitor core
# ---------------------------------------------------------------------------


class TestConversationMonitorDirectAddress:
    def test_direct_address_triggers_evaluation(self) -> None:
        monitor, calls = _make_monitor(side_model_reply="YES")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="hey aria!", is_mention=False
            )
        )
        assert len(calls) == 1
        assert calls[0][0] == 1

    def test_is_mention_triggers_evaluation(self) -> None:
        monitor, calls = _make_monitor(side_model_reply="YES")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="blah", is_mention=True
            )
        )
        assert len(calls) == 1

    def test_direct_address_no_triggers_reset_not_respond(self) -> None:
        """When side model says NO on direct address, on_respond is not called.

        State is still reset (counter, check_count, buffer cleared).
        """
        monitor, calls = _make_monitor(side_model_reply="NO")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        assert len(calls) == 0
        buf = monitor._buffers.get(1)
        # Buffer and counters should be reset after direct address regardless of YES/NO
        assert buf is None or buf.message_counter == 0

    def test_non_address_message_not_evaluated_before_threshold(self) -> None:
        monitor, calls = _make_monitor(
            interjection=Interjection.average, side_model_reply="YES"
        )
        # Send 8 messages (threshold is 9 for average)
        for i in range(8):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        assert len(calls) == 0

    def test_message_counter_increments(self) -> None:
        monitor, _ = _make_monitor(side_model_reply="NO")
        for i in range(5):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        buf = monitor._buffers[1]
        assert buf.message_counter == 5

    def test_buffer_accumulates_messages(self) -> None:
        monitor, _ = _make_monitor(side_model_reply="NO")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="first", is_mention=False
            )
        )
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Bob", text="second", is_mention=False
            )
        )
        buf = monitor._buffers[1]
        assert len(buf.buffer) == 2
        assert buf.buffer[0].speaker == "Alice"
        assert buf.buffer[1].text == "second"


class TestConversationMonitorInterjection:
    def test_interjection_fires_at_threshold(self) -> None:
        monitor, calls = _make_monitor(
            interjection=Interjection.average,  # threshold = 9
            side_model_reply="YES",
        )
        for i in range(9):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        assert len(calls) == 1

    def test_interjection_very_eager_fires_at_3(self) -> None:
        monitor, calls = _make_monitor(
            interjection=Interjection.very_eager, side_model_reply="YES"
        )
        for i in range(3):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        assert len(calls) == 1

    def test_interjection_no_advances_step_down(self) -> None:
        """After a NO, the next check fires 6 messages later (average step-down)."""
        monitor, calls = _make_monitor(
            interjection=Interjection.average,  # starts at 9, then 6
            side_model_reply="NO",
        )
        # First check fires at message 9 → NO
        for i in range(9):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        assert len(calls) == 0
        buf = monitor._buffers[1]
        assert buf.check_count == 1

        # Second check fires 6 messages later (at 15 total) → NO
        for i in range(6):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg2 {i}", is_mention=False
                )
            )
        assert buf.check_count == 2

    def test_state_resets_after_yes(self) -> None:
        monitor, calls = _make_monitor(
            interjection=Interjection.average, side_model_reply="YES"
        )
        for i in range(9):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        assert len(calls) == 1
        buf = monitor._buffers.get(1)
        # After YES, counter and check_count should be reset
        assert buf is None or buf.message_counter == 0
        assert buf is None or buf.check_count == 0

    def test_separate_channels_are_independent(self) -> None:
        monitor, calls = _make_monitor(
            interjection=Interjection.very_eager, side_model_reply="YES"
        )
        # 3 messages on channel 1 → fires
        for i in range(3):
            asyncio.run(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text=f"msg {i}", is_mention=False
                )
            )
        # 2 messages on channel 2 → no fire yet
        for i in range(2):
            asyncio.run(
                monitor.on_message(
                    channel_id=2, speaker="Carol", text=f"msg {i}", is_mention=False
                )
            )
        assert sum(1 for cid, _ in calls if cid == 1) == 1
        assert sum(1 for cid, _ in calls if cid == 2) == 0


# ---------------------------------------------------------------------------
# Step 7 — Side-model YES/NO parsing
# ---------------------------------------------------------------------------


class TestSideModelEvaluation:
    def test_evaluation_sends_system_and_user_messages(self) -> None:
        """The LLM call uses a system message for structured output control."""
        monitor, _ = _make_monitor(side_model_reply="YES")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        llm_client = monitor._llm_client
        messages = llm_client.chat.call_args.args[0]  # ty: ignore[unresolved-attribute]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert "one word" in messages[0].content.lower()
        assert messages[1].role == "user"

    def test_yes_response_calls_on_respond(self) -> None:
        monitor, calls = _make_monitor(side_model_reply="YES")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        assert len(calls) == 1

    def test_no_response_does_not_call_on_respond(self) -> None:
        monitor, calls = _make_monitor(side_model_reply="NO")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        assert len(calls) == 0

    def test_yes_lowercase_accepted(self) -> None:
        monitor, calls = _make_monitor(side_model_reply="yes, I want to respond")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        assert len(calls) == 1

    def test_prose_without_yes_is_rejected(self) -> None:
        """A character-prose reply that doesn't start with YES is treated as NO.

        Regression test for the bug where the interjection LLM
        responded in character (e.g. "The bells on my scarf are
        jingling...") and the monitor incorrectly let it pass when
        "YES" appeared as a substring, or incorrectly rejected it
        even though the model clearly intended to engage.
        """
        monitor, calls = _make_monitor(
            side_model_reply=(
                "The bells on my scarf are jingling quite loudly in response to that!"
            ),
        )
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        assert len(calls) == 0

    def test_yes_with_trailing_punctuation_accepted(self) -> None:
        """``YES.`` or ``YES!`` should still be accepted."""
        monitor, calls = _make_monitor(side_model_reply="YES.")
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="aria?", is_mention=False
            )
        )
        assert len(calls) == 1

    def test_on_respond_receives_buffer_snapshot(self) -> None:
        monitor, calls = _make_monitor(
            interjection=Interjection.very_eager, side_model_reply="YES"
        )
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Alice", text="hello", is_mention=False
            )
        )
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Bob", text="world", is_mention=False
            )
        )
        asyncio.run(
            monitor.on_message(
                channel_id=1, speaker="Carol", text="yes", is_mention=False
            )
        )
        # very_eager threshold = 3 → fires on 3rd message
        assert len(calls) == 1
        _, buf_snapshot = calls[0]
        assert len(buf_snapshot) == 3
        assert buf_snapshot[0].speaker == "Alice"

    def test_lull_endpoint_evaluates_inline_without_starting_timer(self) -> None:
        """Voice path: ``is_lull_endpoint=True`` runs the lull eval inline.

        The voice pipeline already debounced silence via
        ``VoiceLullMonitor``; the conversation monitor must not start
        its own (text) lull timer or wait further. Passing
        ``is_lull_endpoint=True`` should evaluate immediately and skip
        the timer.
        """
        monitor, calls = _make_monitor(
            lull_timeout=100.0,  # would never fire in test if started
            interjection=Interjection.very_quiet,  # counter threshold far away
            side_model_reply="YES",
        )
        asyncio.run(
            monitor.on_message(
                channel_id=99,
                speaker="Alice",
                text="hey there",
                is_mention=False,
                is_lull_endpoint=True,
            )
        )
        # Inline lull eval said YES → on_respond fired exactly once.
        assert len(calls) == 1
        # No pending lull timer left behind.
        buf = monitor._buffers.get(99)
        assert buf is None or buf.lull_timer_handle is None

    def test_lull_endpoint_no_response_does_not_start_timer(self) -> None:
        """``is_lull_endpoint=True`` + NO must not leave a timer armed."""
        monitor, calls = _make_monitor(
            lull_timeout=100.0,
            interjection=Interjection.very_quiet,
            side_model_reply="NO",
        )
        asyncio.run(
            monitor.on_message(
                channel_id=99,
                speaker="Alice",
                text="hey there",
                is_mention=False,
                is_lull_endpoint=True,
            )
        )
        assert len(calls) == 0
        buf = monitor._buffers[99]
        assert buf.lull_timer_handle is None

    def test_yes_decision_logged_at_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log the YES interjection decision at INFO.

        Operators need to see in real time whether the familiar chose
        to respond.
        """
        monitor, _ = _make_monitor(side_model_reply="YES")
        with caplog.at_level("INFO", logger="familiar_connect.chattiness"):
            asyncio.run(
                monitor.on_message(
                    channel_id=42, speaker="Alice", text="aria?", is_mention=False
                )
            )
        matches = [
            r
            for r in caplog.records
            if r.name == "familiar_connect.chattiness"
            and r.levelname == "INFO"
            and "interjection" in r.getMessage()
        ]
        assert len(matches) == 1
        msg = matches[0].getMessage()
        assert "decision=YES" in msg
        assert "channel=42" in msg
        assert "trigger=direct_address" in msg

    def test_no_decision_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Log NO decisions at INFO too.

        Operators need to see the familiar choosing to stay silent, not
        just when it speaks.
        """
        monitor, _ = _make_monitor(side_model_reply="NO")
        with caplog.at_level("INFO", logger="familiar_connect.chattiness"):
            asyncio.run(
                monitor.on_message(
                    channel_id=7, speaker="Alice", text="aria?", is_mention=False
                )
            )
        matches = [
            r
            for r in caplog.records
            if r.name == "familiar_connect.chattiness"
            and r.levelname == "INFO"
            and "interjection" in r.getMessage()
        ]
        assert len(matches) == 1
        assert "decision=NO" in matches[0].getMessage()


# ---------------------------------------------------------------------------
# Step 6 — Lull timer
# ---------------------------------------------------------------------------


class TestLullTimer:
    def test_lull_timer_fires_evaluation(self) -> None:
        """Lull timer fires after lull_timeout seconds with no new messages."""
        monitor, calls = _make_monitor(
            lull_timeout=0.05,  # 50ms for fast test
            interjection=Interjection.very_quiet,  # threshold far away
            side_model_reply="YES",
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text="hello", is_mention=False
                )
            )
            # Wait for lull timer to fire
            loop.run_until_complete(asyncio.sleep(0.15))
        finally:
            loop.close()

        assert len(calls) == 1

    def test_new_message_resets_lull_timer(self) -> None:
        """A second message before lull_timeout cancels the first timer."""
        monitor, calls = _make_monitor(
            lull_timeout=0.1,
            interjection=Interjection.very_quiet,
            side_model_reply="YES",
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # First message starts timer
            loop.run_until_complete(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text="hi", is_mention=False
                )
            )
            # Second message before 100ms cancels and resets timer
            loop.run_until_complete(asyncio.sleep(0.04))
            loop.run_until_complete(
                monitor.on_message(
                    channel_id=1, speaker="Carol", text="hey", is_mention=False
                )
            )
            # Wait past the first timer but not the second
            loop.run_until_complete(asyncio.sleep(0.08))
        finally:
            loop.close()

        # Only the second timer should fire eventually — here we just verify
        # exactly 0 fires so far (second timer hasn't expired yet)
        assert len(calls) == 0

    def test_lull_expiry_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """_schedule_lull_evaluation emits an INFO log when the lull timer fires."""
        monitor, _ = _make_monitor(
            lull_timeout=0.05,
            interjection=Interjection.very_quiet,
            side_model_reply="NO",
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                monitor.on_message(
                    channel_id=1, speaker="Bob", text="hello", is_mention=False
                )
            )
            with caplog.at_level(logging.INFO, logger="familiar_connect.chattiness"):
                loop.run_until_complete(asyncio.sleep(0.15))
        finally:
            loop.close()

        assert any(
            r.levelno == logging.INFO and "conversational lull expired" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# clear_channel
# ---------------------------------------------------------------------------


class TestClearChannel:
    def test_clear_channel_removes_buffer(self) -> None:
        monitor, _ = _make_monitor(side_model_reply="NO")
        asyncio.run(
            monitor.on_message(
                channel_id=5, speaker="Alice", text="hello", is_mention=False
            )
        )
        assert 5 in monitor._buffers
        monitor.clear_channel(5)
        assert 5 not in monitor._buffers

    def test_clear_channel_cancels_lull_timer(self) -> None:
        monitor, calls = _make_monitor(
            lull_timeout=0.05,
            interjection=Interjection.very_quiet,
            side_model_reply="YES",
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                monitor.on_message(
                    channel_id=5, speaker="Alice", text="hello", is_mention=False
                )
            )
            monitor.clear_channel(5)
            # If timer was cancelled, on_respond should NOT be called
            loop.run_until_complete(asyncio.sleep(0.1))
        finally:
            loop.close()

        assert len(calls) == 0

    def test_clear_nonexistent_channel_is_noop(self) -> None:
        monitor, _ = _make_monitor()
        monitor.clear_channel(999)  # should not raise
