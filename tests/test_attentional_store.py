"""Tests for attentional stream additions to :class:`HistoryStore`.

Covers:
- arrived_at / consumed_at columns on turns
- stage_turn / promote_staged_turns / count_staged / staged_channels
- recent_cross_channel
- FocusPointers get/set
- digest watermark get/set
- Migration of existing rows
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from familiar_connect.history.store import (
    FocusPointers,
    HistoryStore,
    HistoryTurn,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _store() -> HistoryStore:
    return HistoryStore(":memory:")


def _append(
    store: HistoryStore,
    *,
    familiar_id: str = "fam",
    channel_id: int = 1,
    content: str = "hello",
    consumed: bool = True,
    arrived_at: datetime | None = None,
) -> HistoryTurn:
    return store.append_turn(
        familiar_id=familiar_id,
        channel_id=channel_id,
        role="user",
        content=content,
        consumed=consumed,
        arrived_at=arrived_at,
    )


# ---------------------------------------------------------------------------
# arrived_at / consumed_at on append_turn
# ---------------------------------------------------------------------------


class TestAppendTurnNewFields:
    def test_consumed_turn_has_consumed_at(self) -> None:
        store = _store()
        turn = _append(store, consumed=True)
        assert turn.consumed_at is not None

    def test_staged_turn_consumed_at_is_none(self) -> None:
        store = _store()
        turn = _append(store, consumed=False)
        assert turn.consumed_at is None

    def test_arrived_at_populated_by_default(self) -> None:
        store = _store()
        turn = _append(store)
        assert turn.arrived_at is not None

    def test_explicit_arrived_at_round_trips(self) -> None:
        store = _store()
        ts = datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC)
        turn = _append(store, arrived_at=ts)
        assert turn.arrived_at is not None
        # isoformat round-trip; just compare as ISO strings to avoid tz edge cases
        assert turn.arrived_at.isoformat() == ts.isoformat()

    def test_consumed_at_equals_arrived_at_when_consumed(self) -> None:
        store = _store()
        ts = datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)
        turn = _append(store, arrived_at=ts, consumed=True)
        assert turn.arrived_at is not None
        assert turn.consumed_at is not None
        assert turn.arrived_at.isoformat() == turn.consumed_at.isoformat()

    def test_legacy_default_consumed_true(self) -> None:
        """Existing callers omitting consumed= still get consumed turns."""
        store = _store()
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="legacy",
        )
        assert turn.consumed_at is not None


# ---------------------------------------------------------------------------
# stage_turn convenience wrapper
# ---------------------------------------------------------------------------


class TestStageTurn:
    def test_stage_turn_returns_unconsummed(self) -> None:
        store = _store()
        turn = store.stage_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="staged",
        )
        assert turn.consumed_at is None

    def test_stage_turn_arrived_at_set(self) -> None:
        store = _store()
        turn = store.stage_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="staged",
        )
        assert turn.arrived_at is not None


# ---------------------------------------------------------------------------
# count_staged
# ---------------------------------------------------------------------------


class TestCountStaged:
    def test_zero_when_empty(self) -> None:
        store = _store()
        assert store.count_staged(familiar_id="fam", channel_id=1) == 0

    def test_counts_only_staged(self) -> None:
        store = _store()
        _append(store, consumed=True)
        _append(store, consumed=False)
        _append(store, consumed=False)
        assert store.count_staged(familiar_id="fam", channel_id=1) == 2

    def test_channel_scoped(self) -> None:
        store = _store()
        _append(store, channel_id=1, consumed=False)
        _append(store, channel_id=2, consumed=False)
        assert store.count_staged(familiar_id="fam", channel_id=1) == 1
        assert store.count_staged(familiar_id="fam", channel_id=2) == 1


# ---------------------------------------------------------------------------
# promote_staged_turns
# ---------------------------------------------------------------------------


class TestPromoteStagedTurns:
    def test_promotes_all_staged_in_channel(self) -> None:
        store = _store()
        _append(store, channel_id=1, consumed=False)
        _append(store, channel_id=1, consumed=False)
        count = store.promote_staged_turns(familiar_id="fam", channel_id=1)
        assert count == 2
        assert store.count_staged(familiar_id="fam", channel_id=1) == 0

    def test_does_not_promote_already_consumed(self) -> None:
        store = _store()
        _append(store, consumed=True)
        count = store.promote_staged_turns(familiar_id="fam", channel_id=1)
        assert count == 0

    def test_channel_scoped_promotion(self) -> None:
        store = _store()
        _append(store, channel_id=1, consumed=False)
        _append(store, channel_id=2, consumed=False)
        store.promote_staged_turns(familiar_id="fam", channel_id=1)
        assert store.count_staged(familiar_id="fam", channel_id=1) == 0
        assert store.count_staged(familiar_id="fam", channel_id=2) == 1

    def test_promoted_turns_consumed_at_set(self) -> None:
        store = _store()
        turn = _append(store, channel_id=1, consumed=False)
        store.promote_staged_turns(familiar_id="fam", channel_id=1)
        # re-fetch via recent()
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        matching = [t for t in turns if t.id == turn.id]
        assert len(matching) == 1
        assert matching[0].consumed_at is not None


# ---------------------------------------------------------------------------
# staged_channels
# ---------------------------------------------------------------------------


class TestStagedChannels:
    def test_empty_when_none_staged(self) -> None:
        store = _store()
        _append(store, channel_id=1, consumed=True)
        assert store.staged_channels(familiar_id="fam") == {}

    def test_maps_channel_to_count(self) -> None:
        store = _store()
        _append(store, channel_id=10, consumed=False)
        _append(store, channel_id=10, consumed=False)
        _append(store, channel_id=20, consumed=False)
        result = store.staged_channels(familiar_id="fam")
        assert result == {10: 2, 20: 1}

    def test_familiar_scoped(self) -> None:
        store = _store()
        _append(store, familiar_id="fam", channel_id=1, consumed=False)
        _append(store, familiar_id="other", channel_id=1, consumed=False)
        assert store.staged_channels(familiar_id="fam") == {1: 1}


# ---------------------------------------------------------------------------
# recent_cross_channel
# ---------------------------------------------------------------------------


class TestRecentCrossChannel:
    def test_empty_when_no_consumed(self) -> None:
        store = _store()
        _append(store, consumed=False)
        result = store.recent_cross_channel(familiar_id="fam", limit=10)
        assert result == []

    def test_returns_consumed_turns_oldest_first(self) -> None:
        store = _store()
        base = datetime(2025, 1, 1, tzinfo=UTC)
        t1 = _append(store, channel_id=1, arrived_at=base, consumed=True)
        t2 = _append(
            store, channel_id=2, arrived_at=base + timedelta(seconds=1), consumed=True
        )
        t3 = _append(
            store, channel_id=1, arrived_at=base + timedelta(seconds=2), consumed=True
        )
        result = store.recent_cross_channel(familiar_id="fam", limit=10)
        assert [t.id for t in result] == [t1.id, t2.id, t3.id]

    def test_limit_returns_last_n(self) -> None:
        store = _store()
        base = datetime(2025, 1, 1, tzinfo=UTC)
        turns = [
            _append(
                store,
                channel_id=i,
                arrived_at=base + timedelta(seconds=i),
                consumed=True,
                content=f"msg{i}",
            )
            for i in range(5)
        ]
        result = store.recent_cross_channel(familiar_id="fam", limit=3)
        assert len(result) == 3
        # last 3 in time order
        assert [t.id for t in result] == [t.id for t in turns[-3:]]

    def test_excludes_staged(self) -> None:
        store = _store()
        _append(store, consumed=True, content="consumed")
        _append(store, consumed=False, content="staged")
        result = store.recent_cross_channel(familiar_id="fam", limit=10)
        assert len(result) == 1
        assert result[0].content == "consumed"

    def test_familiar_scoped(self) -> None:
        store = _store()
        _append(store, familiar_id="fam", consumed=True)
        _append(store, familiar_id="other", consumed=True)
        result = store.recent_cross_channel(familiar_id="fam", limit=10)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# FocusPointers
# ---------------------------------------------------------------------------


class TestFocusPointers:
    def test_returns_none_when_not_set(self) -> None:
        store = _store()
        assert store.get_focus_pointers("fam") is None

    def test_roundtrip(self) -> None:
        store = _store()
        store.set_focus_pointers("fam", text_channel_id=100, voice_channel_id=200)
        fp = store.get_focus_pointers("fam")
        assert fp is not None
        assert fp.text_channel_id == 100
        assert fp.voice_channel_id == 200

    def test_updated_at_populated(self) -> None:
        store = _store()
        before = datetime.now(UTC)
        store.set_focus_pointers("fam", text_channel_id=1, voice_channel_id=None)
        fp = store.get_focus_pointers("fam")
        assert fp is not None
        assert fp.updated_at >= before

    def test_upsert_replaces(self) -> None:
        store = _store()
        store.set_focus_pointers("fam", text_channel_id=1, voice_channel_id=2)
        store.set_focus_pointers("fam", text_channel_id=3, voice_channel_id=None)
        fp = store.get_focus_pointers("fam")
        assert fp is not None
        assert fp.text_channel_id == 3
        assert fp.voice_channel_id is None

    def test_null_channels_allowed(self) -> None:
        store = _store()
        store.set_focus_pointers("fam", text_channel_id=None, voice_channel_id=None)
        fp = store.get_focus_pointers("fam")
        assert fp is not None
        assert fp.text_channel_id is None
        assert fp.voice_channel_id is None

    def test_familiar_scoped(self) -> None:
        store = _store()
        store.set_focus_pointers("fam", text_channel_id=1, voice_channel_id=None)
        assert store.get_focus_pointers("other") is None


# ---------------------------------------------------------------------------
# Digest watermark
# ---------------------------------------------------------------------------


class TestDigestWatermark:
    def test_returns_none_when_not_set(self) -> None:
        store = _store()
        assert store.get_digest_watermark("fam") is None

    def test_roundtrip(self) -> None:
        store = _store()
        ts = datetime(2025, 3, 15, 9, 0, 0, tzinfo=UTC)
        store.set_digest_watermark("fam", ts)
        result = store.get_digest_watermark("fam")
        assert result is not None
        assert result.isoformat() == ts.isoformat()

    def test_upsert_replaces(self) -> None:
        store = _store()
        ts1 = datetime(2025, 1, 1, tzinfo=UTC)
        ts2 = datetime(2025, 6, 1, tzinfo=UTC)
        store.set_digest_watermark("fam", ts1)
        store.set_digest_watermark("fam", ts2)
        result = store.get_digest_watermark("fam")
        assert result is not None
        assert result.isoformat() == ts2.isoformat()

    def test_familiar_scoped(self) -> None:
        store = _store()
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        store.set_digest_watermark("fam", ts)
        assert store.get_digest_watermark("other") is None


# ---------------------------------------------------------------------------
# Migration: existing rows get arrived_at/consumed_at populated
# ---------------------------------------------------------------------------


class TestMigration:
    def test_migration_fills_arrived_at_from_timestamp(self) -> None:
        """arrived_at must be non-NULL after migration (copied from timestamp)."""
        store = _store()
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="legacy",
        )
        # arrived_at should be populated (migration ran in __init__)
        assert turn.arrived_at is not None

    def test_migration_marks_existing_as_consumed(self) -> None:
        """Existing rows start consumed (consumed_at = timestamp)."""
        store = _store()
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="legacy",
        )
        assert turn.consumed_at is not None


# ---------------------------------------------------------------------------
# _row_to_turn fallback for missing columns (legacy row simulation)
# ---------------------------------------------------------------------------


class TestRowToTurnFallback:
    def test_history_turn_arrived_at_defaults_to_none(self) -> None:
        """HistoryTurn.arrived_at and consumed_at have None defaults."""
        from familiar_connect.history.store import HistoryTurn

        t = HistoryTurn(
            id=1,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            role="user",
            author=None,
            content="x",
        )
        assert t.arrived_at is None
        assert t.consumed_at is None


# ---------------------------------------------------------------------------
# AsyncHistoryStore wrappers exist
# ---------------------------------------------------------------------------


class TestAsyncStoreWrappers:
    @pytest.mark.asyncio
    async def test_stage_turn_async(self) -> None:
        from familiar_connect.history.async_store import AsyncHistoryStore

        store = AsyncHistoryStore(HistoryStore(":memory:"))
        turn = await store.stage_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="staged",
        )
        assert turn.consumed_at is None
        store.close()

    @pytest.mark.asyncio
    async def test_promote_staged_turns_async(self) -> None:
        from familiar_connect.history.async_store import AsyncHistoryStore

        store = AsyncHistoryStore(HistoryStore(":memory:"))
        await store.stage_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="staged",
        )
        count = await store.promote_staged_turns(familiar_id="fam", channel_id=1)
        assert count == 1
        store.close()

    @pytest.mark.asyncio
    async def test_get_set_focus_pointers_async(self) -> None:
        from familiar_connect.history.async_store import AsyncHistoryStore

        store = AsyncHistoryStore(HistoryStore(":memory:"))
        await store.set_focus_pointers("fam", text_channel_id=5, voice_channel_id=6)
        fp = await store.get_focus_pointers("fam")
        assert fp is not None
        assert fp.text_channel_id == 5
        store.close()

    @pytest.mark.asyncio
    async def test_get_set_digest_watermark_async(self) -> None:
        from familiar_connect.history.async_store import AsyncHistoryStore

        ts = datetime(2025, 4, 1, tzinfo=UTC)
        store = AsyncHistoryStore(HistoryStore(":memory:"))
        await store.set_digest_watermark("fam", ts)
        result = await store.get_digest_watermark("fam")
        assert result is not None
        assert result.isoformat() == ts.isoformat()
        store.close()

    @pytest.mark.asyncio
    async def test_recent_cross_channel_async(self) -> None:
        from familiar_connect.history.async_store import AsyncHistoryStore

        inner = HistoryStore(":memory:")
        store = AsyncHistoryStore(inner)
        inner.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="msg",
            consumed=True,
        )
        result = await store.recent_cross_channel(familiar_id="fam", limit=10)
        assert len(result) == 1
        store.close()
