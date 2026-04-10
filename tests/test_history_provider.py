"""Red-first tests for the HistoryProvider.

Step 6 of future-features/context-management.md. Reads turns from the
HistoryStore for the request's ``(familiar_id, channel_id)`` and
emits one recent_history Contribution containing the most recent N
turns *in this channel*, and — when there are enough older turns
*globally* — emits a second history_summary Contribution from a
cheap SideModel, cached in the store under ``familiar_id`` so we
don't pay for the same prefix twice.

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

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CHANNEL = 100
_FAMILIAR = "aria"
_GUILD = 1  # observability only


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": _FAMILIAR,
        "channel_id": _CHANNEL,
        "guild_id": _GUILD,
        "speaker": "Alice",
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
        speaker = "Alice" if role == "user" else None
        store.append_turn(
            channel_id=channel_id,
            familiar_id=_FAMILIAR,
            role=role,
            content=f"turn {i}",
            speaker=speaker,
        )


class _StubSideModel:
    """Scripted SideModel for deterministic provider tests."""

    id = "stub"

    def __init__(
        self,
        response: str = "stub summary",
        *,
        delay_s: float = 0.0,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._delay_s = delay_s
        self._exc = exc
        self.calls: list[str] = []

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,  # noqa: ARG002
    ) -> str:
        self.calls.append(prompt)
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._exc is not None:
            raise self._exc
        return self._response


@pytest.fixture
def store(tmp_path: Path) -> HistoryStore:
    """Return a fresh HistoryStore for each test."""
    return HistoryStore(tmp_path / "history.db")


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id_and_deadline(self, store: HistoryStore) -> None:
        provider = HistoryProvider(store=store, side_model=_StubSideModel())
        assert provider.id == "history"
        assert provider.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self, store: HistoryStore) -> None:
        provider = HistoryProvider(store=store, side_model=_StubSideModel())
        assert isinstance(provider, ContextProvider)


# ---------------------------------------------------------------------------
# Recent-history slice
# ---------------------------------------------------------------------------


class TestRecentHistorySlice:
    @pytest.mark.asyncio
    async def test_empty_history_yields_nothing(self, store: HistoryStore) -> None:
        provider = HistoryProvider(store=store, side_model=_StubSideModel())
        contributions = await provider.contribute(_request())
        assert contributions == []

    @pytest.mark.asyncio
    async def test_under_window_returns_only_recent(self, store: HistoryStore) -> None:
        _seed(store, 3)
        side = _StubSideModel()
        provider = HistoryProvider(store=store, side_model=side, window_size=20)

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
        side = _StubSideModel()
        provider = HistoryProvider(store=store, side_model=side)
        (recent,) = await provider.contribute(_request())
        # Speaker name appears for user turns; assistant turns are still labelled.
        assert "Alice" in recent.text


# ---------------------------------------------------------------------------
# Summary path
# ---------------------------------------------------------------------------


class TestSummaryPath:
    @pytest.mark.asyncio
    async def test_over_window_calls_side_model_and_emits_summary(
        self, store: HistoryStore
    ) -> None:
        _seed(store, 25)
        side = _StubSideModel(response="early-conversation summary")
        provider = HistoryProvider(store=store, side_model=side, window_size=10)

        contributions = await provider.contribute(_request())

        layers = {c.layer for c in contributions}
        assert layers == {Layer.recent_history, Layer.history_summary}
        assert len(side.calls) == 1
        # The summariser was given the *older* turns, not the recent window.
        prompt = side.calls[0]
        assert "turn 0" in prompt
        assert "turn 14" in prompt
        assert "turn 15" not in prompt  # in the recent window now

        summary = next(c for c in contributions if c.layer is Layer.history_summary)
        assert summary.text == "early-conversation summary"
        assert summary.priority == HISTORY_SUMMARY_PRIORITY

    @pytest.mark.asyncio
    async def test_fresh_cache_is_reused(self, store: HistoryStore) -> None:
        _seed(store, 25)
        side = _StubSideModel(response="cached summary")
        provider = HistoryProvider(store=store, side_model=side, window_size=10)

        # First call writes the cache.
        await provider.contribute(_request())
        assert len(side.calls) == 1

        # Second call must not invoke the side-model again.
        contributions = await provider.contribute(_request())
        assert len(side.calls) == 1
        summary = next(c for c in contributions if c.layer is Layer.history_summary)
        assert summary.text == "cached summary"

    @pytest.mark.asyncio
    async def test_stale_cache_is_regenerated(self, store: HistoryStore) -> None:
        _seed(store, 25)
        side = _StubSideModel(response="first")
        provider = HistoryProvider(store=store, side_model=side, window_size=10)

        await provider.contribute(_request())  # caches "first"
        assert len(side.calls) == 1

        # Add more turns so the global watermark advances past what the
        # cache covers.
        seed_more = 5
        for i in range(seed_more):
            store.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content=f"new {i}",
                speaker="Alice",
            )

        side._response = "second"
        contributions = await provider.contribute(_request())
        assert len(side.calls) == 2
        summary = next(c for c in contributions if c.layer is Layer.history_summary)
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
        side = _StubSideModel(exc=RuntimeError("kaboom"))
        provider = HistoryProvider(store=store, side_model=side, window_size=10)

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
        side = _StubSideModel(delay_s=2.0)
        provider = HistoryProvider(
            store=store,
            side_model=side,
            window_size=10,
            summary_timeout_s=0.05,
        )

        contributions = await provider.contribute(_request())

        layers = {c.layer for c in contributions}
        assert Layer.recent_history in layers
        assert Layer.history_summary not in layers

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_stale_cached_summary(
        self, store: HistoryStore
    ) -> None:
        """A timed-out new run surfaces the previously-cached summary.

        Better stale than nothing — once we've paid for a summary,
        the rolling window keeps it around even when the next refresh
        misses its deadline.
        """
        _seed(store, 25)
        # First, populate the cache with a fast side-model.
        fast = _StubSideModel(response="cached value")
        provider = HistoryProvider(store=store, side_model=fast, window_size=10)
        await provider.contribute(_request())

        # Add more turns and switch to a slow side-model.
        for i in range(5):
            store.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content=f"new {i}",
                speaker="Alice",
            )

        slow = _StubSideModel(response="never returned", delay_s=2.0)
        provider2 = HistoryProvider(
            store=store,
            side_model=slow,
            window_size=10,
            summary_timeout_s=0.05,
        )
        contributions = await provider2.contribute(_request())

        summary = next(
            (c for c in contributions if c.layer is Layer.history_summary),
            None,
        )
        assert summary is not None
        assert summary.text == "cached value"


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
        speaker = "Alice" if role == "user" else None
        store.append_turn(
            channel_id=channel_id,
            familiar_id=_FAMILIAR,
            role=role,
            content=f"{mode.value} turn {i}",
            speaker=speaker,
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

        side = _StubSideModel()
        provider = HistoryProvider(
            store=store,
            side_model=side,
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

        side = _StubSideModel(response="channel summary")
        provider = HistoryProvider(
            store=store,
            side_model=side,
            window_size=10,
            mode=ChannelMode.full_rp,
        )
        contributions = await provider.contribute(_request())
        summary = next(
            (c for c in contributions if c.layer is Layer.history_summary),
            None,
        )
        assert summary is not None
        # The side model was called twice: once for the within-channel
        # summary and once for the cross-context summary of channel 999.
        assert len(side.calls) == 2
        # The first call is the within-channel summary — verify it
        # doesn't contain the other channel's turns.
        prompt = side.calls[0]
        assert "Summarise the following" in prompt
        # The summary should come from the within-channel older turns only.
        assert summary.text == "channel summary"


# ---------------------------------------------------------------------------
# Cross-context "meanwhile elsewhere" summaries
# ---------------------------------------------------------------------------


class TestCrossContextContributions:
    @pytest.mark.asyncio
    async def test_cross_context_emitted_per_other_channel(
        self, store: HistoryStore
    ) -> None:
        """When other channels have activity, cross-context summaries appear."""
        # Current channel (100) has full_rp turns.
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=100)
        # Other channel (200) has text_conversation_rp turns.
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=200)

        side = _StubSideModel(response="Meanwhile in text chat...")
        provider = HistoryProvider(
            store=store,
            side_model=side,
            window_size=20,
            mode=ChannelMode.full_rp,
        )
        contributions = await provider.contribute(_request(channel_id=100))

        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        assert len(cross) == 1
        assert cross[0].layer is Layer.history_summary
        assert cross[0].priority == 55
        assert "Meanwhile" in cross[0].text

    @pytest.mark.asyncio
    async def test_cross_context_cache_reuse(self, store: HistoryStore) -> None:
        """A fresh cross-context cache is reused without re-calling the side model."""
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=100)
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=200)

        side = _StubSideModel(response="cached cross summary")
        provider = HistoryProvider(
            store=store,
            side_model=side,
            window_size=20,
            mode=ChannelMode.full_rp,
        )
        # First call builds the cache.
        await provider.contribute(_request(channel_id=100))
        call_count_after_first = len(side.calls)

        # Second call should reuse the cache — no new side-model calls
        # for the cross-context summary.
        await provider.contribute(_request(channel_id=100))
        # The side model may be called for the within-channel summary,
        # but the cross-context summary call should not repeat.
        cross_calls = [
            c
            for c in side.calls[call_count_after_first:]
            if "cross" in c.lower() or "meanwhile" in c.lower()
        ]
        # No new cross-context calls expected — the cache is fresh.
        assert cross_calls == []

    @pytest.mark.asyncio
    async def test_cross_context_falls_back_on_side_model_failure(
        self, store: HistoryStore
    ) -> None:
        """When the side model fails, cross-context contributions are skipped."""
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=100)
        _seed_with_mode(store, 5, mode=ChannelMode.text_conversation_rp, channel_id=200)

        side = _StubSideModel(exc=RuntimeError("kaboom"))
        provider = HistoryProvider(
            store=store,
            side_model=side,
            window_size=20,
            mode=ChannelMode.full_rp,
        )
        contributions = await provider.contribute(_request(channel_id=100))

        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        # The failure is gracefully handled — no cross contributions.
        assert cross == []

    @pytest.mark.asyncio
    async def test_no_cross_context_when_no_other_channels(
        self, store: HistoryStore
    ) -> None:
        """Single-channel familiars produce no cross-context contributions."""
        _seed_with_mode(store, 5, mode=ChannelMode.full_rp, channel_id=100)

        side = _StubSideModel(response="should not appear")
        provider = HistoryProvider(
            store=store,
            side_model=side,
            window_size=20,
            mode=ChannelMode.full_rp,
        )
        contributions = await provider.contribute(_request(channel_id=100))

        cross = [
            c for c in contributions if c.source.startswith("history:cross_channel:")
        ]
        assert cross == []
