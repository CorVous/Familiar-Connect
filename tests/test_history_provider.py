"""Red-first tests for the HistoryProvider.

Step 6 of docs/architecture/context-pipeline.md. Reads turns from the
HistoryStore for the request's ``(familiar_id, channel_id)`` and
emits one recent_history Contribution containing the most recent N
turns *in this channel*, and — when there are enough older turns
*globally* — emits a second history_summary Contribution from the
``history_summary`` slot's :class:`LLMClient`, cached in the store
under ``familiar_id`` so we don't pay for the same prefix twice.

The recent window is partitioned per channel; the rolling summary is
global per familiar.

Covers familiar_connect.context.providers.history.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from familiar_connect.config import ChannelMode
from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.history import (
    HISTORY_RECENT_PRIORITY,
    HISTORY_SUMMARY_PRIORITY,
    HistoryProvider,
)
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CHANNEL = 100
_FAMILIAR = "aria"
_GUILD = 1  # observability only
_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": _FAMILIAR,
        "channel_id": _CHANNEL,
        "guild_id": _GUILD,
        "author": _ALICE,
        "utterance": "hello",
        "modality": Modality.text,
        "budget_tokens": 2048,
        "deadline_s": 10.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)  # type: ignore[arg-type]


def _seed(store: HistoryStore, n: int, *, channel_id: int = _CHANNEL) -> None:
    """Append *n* alternating user/assistant turns to the test channel."""
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        author = _ALICE if role == "user" else None
        store.append_turn(
            channel_id=channel_id,
            familiar_id=_FAMILIAR,
            role=role,
            content=f"turn {i}",
            author=author,
        )


class _StubLLMClient(LLMClient):
    """Scripted :class:`LLMClient` for deterministic provider tests."""

    def __init__(
        self,
        response: str = "stub summary",
        *,
        delay_s: float = 0.0,
        exc: Exception | None = None,
    ) -> None:
        super().__init__(api_key="stub-test-key", model="stub/test-model")
        self._response = response
        self._delay_s = delay_s
        self._exc = exc
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._exc is not None:
            raise self._exc
        return Message(role="assistant", content=self._response)

    async def close(self) -> None:  # pragma: no cover — no resources
        return

    def prompt_at(self, index: int) -> str:
        """Return the user-message text of the chat call at *index*."""
        return self.calls[index][0].content


@pytest.fixture
def store(tmp_path: Path) -> HistoryStore:
    """Return a fresh HistoryStore for each test."""
    return HistoryStore(tmp_path / "history.db")


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id_and_deadline(self, store: HistoryStore) -> None:
        provider = HistoryProvider(store=store, llm_client=_StubLLMClient())
        assert provider.id == "history"
        assert provider.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self, store: HistoryStore) -> None:
        provider = HistoryProvider(store=store, llm_client=_StubLLMClient())
        assert isinstance(provider, ContextProvider)


# ---------------------------------------------------------------------------
# Recent-history slice
# ---------------------------------------------------------------------------


class TestRecentHistorySlice:
    @pytest.mark.asyncio
    async def test_empty_history_yields_nothing(self, store: HistoryStore) -> None:
        provider = HistoryProvider(store=store, llm_client=_StubLLMClient())
        contributions = await provider.contribute(_request())
        assert contributions == []

    @pytest.mark.asyncio
    async def test_under_window_returns_only_recent(self, store: HistoryStore) -> None:
        _seed(store, 3)
        side = _StubLLMClient()
        provider = HistoryProvider(store=store, llm_client=side, window_size=20)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        recent = contributions[0]
        assert isinstance(recent, Contribution)
        assert recent.layer is Layer.recent_history
        assert recent.priority == HISTORY_RECENT_PRIORITY
        assert "turn 0" in recent.text
        assert "turn 2" in recent.text
        # No summariser call when nothing has aged out.
        assert side.calls == []

    @pytest.mark.asyncio
    async def test_recent_text_includes_speakers(self, store: HistoryStore) -> None:
        _seed(store, 2)
        side = _StubLLMClient()
        provider = HistoryProvider(store=store, llm_client=side)
        (recent,) = await provider.contribute(_request())
        # Speaker name appears for user turns; assistant turns are still labelled.
        assert "Alice" in recent.text


# ---------------------------------------------------------------------------
# Summary path
# ---------------------------------------------------------------------------


async def _drain_refresh(provider: HistoryProvider) -> None:
    """Await all currently-pending background refresh tasks on *provider*."""
    await asyncio.gather(
        *provider._pending_refresh.values(),
        return_exceptions=True,
    )


class TestSummaryPath:
    @pytest.mark.asyncio
    async def test_over_window_schedules_summary_in_background(
        self, store: HistoryStore
    ) -> None:
        """Cache miss: first call returns recent only; bg task populates cache."""
        _seed(store, 25)
        side = _StubLLMClient(response="early-conversation summary")
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        # First call: recent only, summariser scheduled.
        first = await provider.contribute(_request())
        assert {c.layer for c in first} == {Layer.recent_history}

        # Drain background refresh.
        await _drain_refresh(provider)
        assert len(side.calls) == 1
        prompt = side.prompt_at(0)
        assert "turn 0" in prompt
        assert "turn 14" in prompt
        assert "turn 15" not in prompt  # in the recent window now

        # Second call: cache now warm, summary surfaces.
        second = await provider.contribute(_request())
        layers = {c.layer for c in second}
        assert layers == {Layer.recent_history, Layer.history_summary}
        summary = next(c for c in second if c.layer is Layer.history_summary)
        assert summary.text == "early-conversation summary"
        assert summary.priority == HISTORY_SUMMARY_PRIORITY

    @pytest.mark.asyncio
    async def test_fresh_cache_is_reused(self, store: HistoryStore) -> None:
        _seed(store, 25)
        side = _StubLLMClient(response="cached summary")
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        # First call schedules refresh.
        await provider.contribute(_request())
        await _drain_refresh(provider)
        assert len(side.calls) == 1

        # Second call hits the cache — no new side-model call.
        contributions = await provider.contribute(_request())
        assert len(side.calls) == 1
        summary = next(c for c in contributions if c.layer is Layer.history_summary)
        assert summary.text == "cached summary"

    @pytest.mark.asyncio
    async def test_stale_cache_is_served_then_refreshed(
        self, store: HistoryStore
    ) -> None:
        """Stale cache surfaces immediately; bg refresh updates for next turn."""
        _seed(store, 25)
        side = _StubLLMClient(response="first")
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        # Prime the cache.
        await provider.contribute(_request())
        await _drain_refresh(provider)
        assert len(side.calls) == 1

        # Add new turns so the watermark advances past the cache.
        for i in range(5):
            store.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content=f"new {i}",
                author=_ALICE,
            )

        # Flip response before triggering the refresh so the bg task
        # picks up the new value.
        side._response = "second"

        # Next call surfaces the stale cache immediately and schedules refresh.
        stale_contribs = await provider.contribute(_request())
        summary = next(c for c in stale_contribs if c.layer is Layer.history_summary)
        assert summary.text == "first"  # stale — acceptable
        await _drain_refresh(provider)
        assert len(side.calls) == 2

        # Subsequent call sees the refreshed summary.
        fresh_contribs = await provider.contribute(_request())
        summary = next(c for c in fresh_contribs if c.layer is Layer.history_summary)
        assert summary.text == "second"


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestSummariserFailures:
    @pytest.mark.asyncio
    async def test_side_model_exception_falls_back_to_recent_only(
        self, store: HistoryStore
    ) -> None:
        _seed(store, 25)
        side = _StubLLMClient(exc=RuntimeError("kaboom"))
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        contributions = await provider.contribute(_request())

        layers = {c.layer for c in contributions}
        assert Layer.recent_history in layers
        assert Layer.history_summary not in layers

    @pytest.mark.asyncio
    async def test_side_model_timeout_falls_back_to_recent_only(
        self, store: HistoryStore
    ) -> None:
        _seed(store, 25)
        # 2-second delay against a 0.05-second internal soft deadline.
        side = _StubLLMClient(delay_s=2.0)
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=10,
            summary_timeout_s=0.05,
        )

        contributions = await provider.contribute(_request())

        layers = {c.layer for c in contributions}
        assert Layer.recent_history in layers
        assert Layer.history_summary not in layers

    @pytest.mark.asyncio
    async def test_stale_cache_surfaces_while_slow_refresh_runs(
        self, store: HistoryStore
    ) -> None:
        """Stale cache surfaces on the critical path while a slow refresh times out."""
        _seed(store, 25)
        # First, populate the cache with a fast side-model.
        fast = _StubLLMClient(response="cached value")
        provider = HistoryProvider(store=store, llm_client=fast, window_size=10)
        await provider.contribute(_request())
        await _drain_refresh(provider)

        # Add more turns and switch to a slow side-model.
        for i in range(5):
            store.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content=f"new {i}",
                author=_ALICE,
            )

        slow = _StubLLMClient(response="never returned", delay_s=2.0)
        provider2 = HistoryProvider(
            store=store,
            llm_client=slow,
            window_size=10,
            summary_timeout_s=0.05,
        )
        contributions = await provider2.contribute(_request())

        # Stale cache surfaces immediately — no wait for the slow refresh.
        summary = next(
            (c for c in contributions if c.layer is Layer.history_summary),
            None,
        )
        assert summary is not None
        assert summary.text == "cached value"
        # Drain the bg refresh (times out, cache unchanged).
        await _drain_refresh(provider2)


# ---------------------------------------------------------------------------
# Mode-scoped recent window
# ---------------------------------------------------------------------------


def _seed_with_mode(
    store: HistoryStore,
    n: int,
    *,
    mode: ChannelMode,
    channel_id: int = _CHANNEL,
) -> None:
    """Append *n* alternating user/assistant turns with an explicit mode."""
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        author = _ALICE if role == "user" else None
        store.append_turn(
            channel_id=channel_id,
            familiar_id=_FAMILIAR,
            role=role,
            content=f"{mode.value} turn {i}",
            author=author,
            mode=mode,
        )


class TestModeFilteredRecentWindow:
    @pytest.mark.asyncio
    async def test_recent_contribution_only_contains_matching_mode(
        self, store: HistoryStore
    ) -> None:
        """When mode is set, only turns from that mode appear in the recent window."""
        _seed_with_mode(store, 3, mode=ChannelMode.full_rp)
        _seed_with_mode(store, 3, mode=ChannelMode.imitate_voice)

        side = _StubLLMClient()
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=20,
            mode=ChannelMode.full_rp,
        )
        contributions = await provider.contribute(_request())
        assert len(contributions) >= 1
        recent = next(c for c in contributions if c.layer is Layer.recent_history)
        assert "full_rp turn 0" in recent.text
        assert "full_rp turn 2" in recent.text
        assert "imitate_voice" not in recent.text

    @pytest.mark.asyncio
    async def test_summary_scoped_to_channel(self, store: HistoryStore) -> None:
        """The rolling summary should scope to the current channel."""
        # 25 turns in the request channel, 5 in another channel.
        _seed_with_mode(store, 25, mode=ChannelMode.full_rp, channel_id=_CHANNEL)
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=999)

        side = _StubLLMClient(response="channel summary")
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=10,
            mode=ChannelMode.full_rp,
        )
        # First call schedules both within-channel summary and
        # cross-context refreshes in the background.
        await provider.contribute(_request())
        await _drain_refresh(provider)
        # Two side-model calls: within-channel summary + cross-context.
        assert len(side.calls) == 2
        prompt = side.prompt_at(0)
        assert "Summarise the following" in prompt

        # Next call picks up the cached within-channel summary.
        contributions = await provider.contribute(_request())
        summary = next(
            (c for c in contributions if c.layer is Layer.history_summary),
            None,
        )
        assert summary is not None
        assert summary.text == "channel summary"


# ---------------------------------------------------------------------------
# Cross-context "meanwhile elsewhere" summaries
# ---------------------------------------------------------------------------


class TestCrossContextContributions:
    @pytest.mark.asyncio
    async def test_cross_context_emitted_for_non_full_rp(
        self, store: HistoryStore
    ) -> None:
        """Non-full_rp modes get cross-context as contributions."""
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=100)
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=200)

        side = _StubLLMClient(response="Meanwhile in the RP scene...")
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=20,
            mode=ChannelMode.text_conversation_rp,
        )
        # First call schedules cross-context refresh — not yet cached.
        await provider.contribute(_request(channel_id=100))
        await _drain_refresh(provider)

        # Next call emits the now-cached cross-context contribution.
        contributions = await provider.contribute(_request(channel_id=100))
        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        assert len(cross) == 1
        assert cross[0].layer is Layer.history_summary
        assert cross[0].priority == 55
        assert "Meanwhile" in cross[0].text

    @pytest.mark.asyncio
    async def test_full_rp_caches_but_does_not_emit_cross_context(
        self, store: HistoryStore
    ) -> None:
        """full_rp caches cross-context but does not emit contributions.

        The renderer inserts them as mid-chat breadcrumbs instead.
        """
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=100)
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=200)

        side = _StubLLMClient(response="Meanwhile in text chat...")
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=20,
            mode=ChannelMode.full_rp,
        )
        contributions = await provider.contribute(_request(channel_id=100))

        # No cross-context contributions emitted for full_rp.
        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        assert cross == []

        # Background refresh populates the cache for the renderer.
        await _drain_refresh(provider)
        cached = store.get_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=200,
        )
        assert cached is not None
        assert "Meanwhile" in cached.summary_text

    @pytest.mark.asyncio
    async def test_cross_context_cache_reuse(self, store: HistoryStore) -> None:
        """A fresh cross-context cache is reused without re-calling the side model."""
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=100)
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=200)

        side = _StubLLMClient(response="cached cross summary")
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=20,
            mode=ChannelMode.text_conversation_rp,
        )
        # First call schedules the bg refresh; drain it so cache warms.
        await provider.contribute(_request(channel_id=100))
        await _drain_refresh(provider)
        call_count_after_first = len(side.calls)

        # Second call should reuse the cache — no new LLM calls for
        # the cross-context summary.
        await provider.contribute(_request(channel_id=100))
        new_calls = side.calls[call_count_after_first:]
        cross_calls = [
            messages
            for messages in new_calls
            if "cross" in messages[0].content.lower()
            or "meanwhile" in messages[0].content.lower()
        ]
        assert cross_calls == []

    @pytest.mark.asyncio
    async def test_cross_context_falls_back_on_side_model_failure(
        self, store: HistoryStore
    ) -> None:
        """When the side model fails, cross-context contributions are skipped."""
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=100)
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=200)

        side = _StubLLMClient(exc=RuntimeError("kaboom"))
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=20,
            mode=ChannelMode.text_conversation_rp,
        )
        contributions = await provider.contribute(_request(channel_id=100))

        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        assert cross == []
        # Drain bg refresh (errors logged, swallowed).
        await _drain_refresh(provider)

    @pytest.mark.asyncio
    async def test_no_cross_context_when_no_other_channels(
        self, store: HistoryStore
    ) -> None:
        """Single-channel familiars produce no cross-context contributions."""
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=100)

        side = _StubLLMClient(response="should not appear")
        provider = HistoryProvider(
            store=store,
            llm_client=side,
            window_size=20,
            mode=ChannelMode.text_conversation_rp,
        )
        contributions = await provider.contribute(_request(channel_id=100))

        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        assert cross == []


# ---------------------------------------------------------------------------
# Background refresh dedupe + error handling
# ---------------------------------------------------------------------------


class TestBackgroundRefresh:
    @pytest.mark.asyncio
    async def test_cache_miss_schedules_bg_task_and_warms_on_next_turn(
        self, store: HistoryStore
    ) -> None:
        """Refresh runs off the critical path; second turn sees the fresh summary."""
        _seed(store, 25)
        side = _StubLLMClient(response="fresh summary")
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        first = await provider.contribute(_request())
        # Cache was empty on the first turn → no summary contribution yet.
        assert {c.layer for c in first} == {Layer.recent_history}
        # One task queued on the instance.
        assert len(provider._pending_refresh) == 1

        await _drain_refresh(provider)
        assert len(side.calls) == 1

        second = await provider.contribute(_request())
        assert {c.layer for c in second} == {
            Layer.recent_history,
            Layer.history_summary,
        }
        summary = next(c for c in second if c.layer is Layer.history_summary)
        assert summary.text == "fresh summary"
        # No new LLM calls on the warmed-cache turn.
        assert len(side.calls) == 1

    @pytest.mark.asyncio
    async def test_concurrent_cache_miss_calls_dedupe_to_one_task(
        self, store: HistoryStore
    ) -> None:
        """Two concurrent contribute() calls schedule at most one refresh."""
        _seed(store, 25)
        side = _StubLLMClient(response="one", delay_s=0.05)
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        await asyncio.gather(
            provider.contribute(_request()),
            provider.contribute(_request()),
        )
        # Exactly one bg task held by the provider (the second was deduped).
        assert len(provider._pending_refresh) == 1
        await _drain_refresh(provider)
        # Side model called once.
        assert len(side.calls) == 1

    @pytest.mark.asyncio
    async def test_bg_refresh_error_leaves_cache_untouched(
        self,
        store: HistoryStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Errors in the bg task are logged and swallowed; cache not written."""
        _seed(store, 25)
        side = _StubLLMClient(exc=RuntimeError("kaboom"))
        provider = HistoryProvider(store=store, llm_client=side, window_size=10)

        with caplog.at_level("WARNING"):
            await provider.contribute(_request())
            await _drain_refresh(provider)

        cached = store.get_summary(familiar_id=_FAMILIAR, channel_id=_CHANNEL)
        assert cached is None
        assert any("background refresh" in r.message.lower() for r in caplog.records)
