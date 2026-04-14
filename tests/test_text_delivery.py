"""Tests for the text-channel chunked-delivery helpers and tracker."""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.config import TypingSimulationConfig
from familiar_connect.text.delivery import (
    TextDeliveryRegistry,
    TextDeliveryTracker,
    compute_typing_delay,
    split_reply_into_chunks,
)


class TestSplitReplyIntoChunks:
    def test_empty_returns_empty(self) -> None:
        assert split_reply_into_chunks("", sentence_split_threshold=400) == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert (
            split_reply_into_chunks("   \n\n  \t  ", sentence_split_threshold=400) == []
        )

    def test_single_paragraph_under_threshold_kept_whole(self) -> None:
        text = "Hello there, friend. How are you today?"
        chunks = split_reply_into_chunks(text, sentence_split_threshold=400)
        assert chunks == [text]

    def test_splits_on_blank_line_between_paragraphs(self) -> None:
        text = "First para.\n\nSecond para.\n\nThird para."
        chunks = split_reply_into_chunks(text, sentence_split_threshold=400)
        assert chunks == ["First para.", "Second para.", "Third para."]

    def test_splits_on_multi_blank_lines(self) -> None:
        text = "A.\n\n\n\nB."
        chunks = split_reply_into_chunks(text, sentence_split_threshold=400)
        assert chunks == ["A.", "B."]

    def test_strips_leading_trailing_whitespace_per_paragraph(self) -> None:
        text = "   para1   \n\n   para2   "
        chunks = split_reply_into_chunks(text, sentence_split_threshold=400)
        assert chunks == ["para1", "para2"]

    def test_oversize_paragraph_splits_by_sentence(self) -> None:
        sentence = "This is one reasonably long sentence."
        para = " ".join([sentence] * 20)  # 20 sentences, ~760 chars
        chunks = split_reply_into_chunks(para, sentence_split_threshold=200)
        # each sentence should be its own chunk
        assert len(chunks) == 20
        assert all(c == sentence for c in chunks)

    def test_sentence_split_handles_question_and_exclamation(self) -> None:
        para = "Are you ok? I hope so! Tell me more."
        chunks = split_reply_into_chunks(para, sentence_split_threshold=10)
        assert chunks == ["Are you ok?", "I hope so!", "Tell me more."]

    def test_sentence_split_handles_ellipsis(self) -> None:
        para = "Well\u2026 I don't know. Maybe later\u2026 yes."
        chunks = split_reply_into_chunks(para, sentence_split_threshold=10)
        # splits after … and after .
        assert "Well\u2026" in chunks
        assert any("I don't know." in c for c in chunks)

    def test_under_threshold_paragraph_not_sentence_split(self) -> None:
        para = "Short one. Another short one. And one more."
        chunks = split_reply_into_chunks(para, sentence_split_threshold=400)
        assert chunks == [para]

    def test_hard_cap_splits_oversize_single_sentence(self) -> None:
        # one huge run-on with no sentence terminators
        text = "word " * 500  # ~2500 chars, no periods
        chunks = split_reply_into_chunks(
            text.strip(),
            sentence_split_threshold=10,
            hard_cap=100,
        )
        assert all(len(c) <= 100 for c in chunks)
        # joined should reconstruct whitespace-flattened content
        assert " ".join(chunks).replace("  ", " ").startswith("word word word")

    def test_mixed_short_and_long_paragraphs(self) -> None:
        short = "Short."
        long_para = " ".join([f"Sentence number {i}." for i in range(10)])
        text = f"{short}\n\n{long_para}\n\n{short}"
        chunks = split_reply_into_chunks(text, sentence_split_threshold=50)
        assert chunks[0] == short
        assert chunks[-1] == short
        # middle paragraph exploded into its 10 sentences
        assert len(chunks) == 12


class TestComputeTypingDelay:
    def _cfg(self, **kw: object) -> TypingSimulationConfig:
        base: dict[str, object] = {
            "enabled": True,
            "chars_per_second": 40.0,
            "min_delay_s": 0.5,
            "max_delay_s": 5.0,
            "inter_line_pause_s": 0.5,
            "sentence_split_threshold": 400,
        }
        base.update(kw)
        return TypingSimulationConfig(**base)  # type: ignore[arg-type]

    def test_proportional_to_length(self) -> None:
        cfg = self._cfg(chars_per_second=10.0, min_delay_s=0.0, max_delay_s=100.0)
        assert compute_typing_delay("a" * 50, cfg) == pytest.approx(5.0)

    def test_clamped_to_min(self) -> None:
        cfg = self._cfg(chars_per_second=100.0, min_delay_s=2.0, max_delay_s=10.0)
        # short chunk would compute to < 2.0 → clamp
        assert compute_typing_delay("hi", cfg) == pytest.approx(2.0)

    def test_clamped_to_max(self) -> None:
        cfg = self._cfg(chars_per_second=1.0, min_delay_s=0.0, max_delay_s=3.0)
        # 100 chars at 1 cps = 100s → clamp to 3.0
        assert compute_typing_delay("a" * 100, cfg) == pytest.approx(3.0)

    def test_zero_cps_returns_max_delay(self) -> None:
        cfg = self._cfg(chars_per_second=0.0, min_delay_s=0.5, max_delay_s=4.0)
        assert compute_typing_delay("anything", cfg) == pytest.approx(4.0)

    def test_negative_cps_returns_max_delay(self) -> None:
        cfg = self._cfg(chars_per_second=-5.0, min_delay_s=0.5, max_delay_s=4.0)
        assert compute_typing_delay("anything", cfg) == pytest.approx(4.0)


class TestTextDeliveryTrackerDefaults:
    def test_new_tracker_has_no_task(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        assert t.task is None

    def test_new_tracker_has_empty_sent(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        assert t.sent_chunks == []

    def test_new_tracker_inactive(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        assert t.is_active() is False


class TestTextDeliveryTrackerLifecycle:
    @pytest.mark.asyncio
    async def test_start_registers_task_and_clears_scratch(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        t.sent_chunks = ["stale"]

        async def noop() -> None:
            await asyncio.sleep(0)

        task = asyncio.create_task(noop())
        t.start(task)

        assert t.task is task
        assert t.sent_chunks == []
        await task  # let it finish to avoid pending-task warnings

    @pytest.mark.asyncio
    async def test_is_active_true_while_running(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        started = asyncio.Event()

        async def run() -> None:
            started.set()
            await asyncio.sleep(1.0)

        task = asyncio.create_task(run())
        t.start(task)
        await started.wait()
        assert t.is_active() is True
        await t.cancel_and_wait()

    @pytest.mark.asyncio
    async def test_is_active_false_after_completion(self) -> None:
        t = TextDeliveryTracker(channel_id=1)

        async def run() -> None:
            await asyncio.sleep(0)

        task = asyncio.create_task(run())
        t.start(task)
        await task
        assert t.is_active() is False

    def test_mark_sent_accumulates(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        t.mark_sent("a")
        t.mark_sent("b")
        assert t.sent_chunks == ["a", "b"]

    @pytest.mark.asyncio
    async def test_cancel_and_wait_returns_sent_snapshot(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        started = asyncio.Event()

        async def run() -> None:
            started.set()
            await asyncio.sleep(1.0)

        task = asyncio.create_task(run())
        t.start(task)
        t.mark_sent("line1")
        t.mark_sent("line2")
        await started.wait()

        sent = await t.cancel_and_wait()
        assert sent == ["line1", "line2"]
        assert t.task is None

    @pytest.mark.asyncio
    async def test_cancel_and_wait_swallows_cancelled_error(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        started = asyncio.Event()

        async def run() -> None:
            started.set()
            await asyncio.sleep(10.0)

        task = asyncio.create_task(run())
        t.start(task)
        await started.wait()

        # should not raise
        await t.cancel_and_wait()

    @pytest.mark.asyncio
    async def test_cancel_and_wait_idle_returns_empty(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        assert await t.cancel_and_wait() == []

    @pytest.mark.asyncio
    async def test_cancel_and_wait_after_natural_completion(self) -> None:
        t = TextDeliveryTracker(channel_id=1)

        async def run() -> None:
            await asyncio.sleep(0)

        task = asyncio.create_task(run())
        t.start(task)
        t.mark_sent("x")
        await task
        # task already done; cancel_and_wait should still return sent snapshot
        sent = await t.cancel_and_wait()
        assert sent == ["x"]

    def test_clear_resets_task_and_scratch(self) -> None:
        t = TextDeliveryTracker(channel_id=1)
        t.sent_chunks = ["a", "b"]
        t.clear()
        assert t.task is None
        assert t.sent_chunks == []


class TestTextDeliveryRegistry:
    def test_get_creates_on_first_access(self) -> None:
        reg = TextDeliveryRegistry()
        tracker = reg.get(channel_id=42)
        assert isinstance(tracker, TextDeliveryTracker)
        assert tracker.channel_id == 42

    def test_get_returns_same_tracker_on_repeat(self) -> None:
        reg = TextDeliveryRegistry()
        t1 = reg.get(channel_id=42)
        t2 = reg.get(channel_id=42)
        assert t1 is t2

    def test_get_different_channels_different_trackers(self) -> None:
        reg = TextDeliveryRegistry()
        t1 = reg.get(channel_id=1)
        t2 = reg.get(channel_id=2)
        assert t1 is not t2

    def test_drop_removes_tracker(self) -> None:
        reg = TextDeliveryRegistry()
        reg.get(channel_id=1)
        reg.drop(1)
        assert 1 not in reg.snapshot()

    def test_drop_unknown_channel_is_noop(self) -> None:
        reg = TextDeliveryRegistry()
        reg.drop(999)  # should not raise

    def test_snapshot_is_shallow_copy(self) -> None:
        reg = TextDeliveryRegistry()
        reg.get(1)
        snap = reg.snapshot()
        snap.clear()
        assert 1 in reg.snapshot()
