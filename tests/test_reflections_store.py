"""Tests for the reflections table (M3) on :class:`HistoryStore`.

Covers append + read + watermark queries and the stale-citation
helper used by :class:`ReflectionLayer`.
"""

from __future__ import annotations

from familiar_connect.history.store import HistoryStore, Reflection


def _seed_facts(store: HistoryStore, count: int = 3) -> list[int]:
    """Insert ``count`` placeholder facts; return their ids."""
    out: list[int] = []
    for i in range(count):
        f = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text=f"fact {i}",
            source_turn_ids=[i + 1],
        )
        out.append(f.id)
    return out


class TestReflectionsAppend:
    def test_append_returns_reflection_with_provenance(self) -> None:
        store = HistoryStore(":memory:")
        r = store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="The crew keeps circling back to homesickness.",
            cited_turn_ids=[1, 2, 3],
            cited_fact_ids=[5, 7],
            last_turn_id=10,
            last_fact_id=8,
        )
        assert isinstance(r, Reflection)
        assert r.id > 0
        assert r.cited_turn_ids == (1, 2, 3)
        assert r.cited_fact_ids == (5, 7)
        assert r.last_turn_id == 10
        assert r.last_fact_id == 8
        assert r.channel_id == 1
        assert r.familiar_id == "fam"

    def test_channel_agnostic_reflection_stores_null_channel(self) -> None:
        store = HistoryStore(":memory:")
        r = store.append_reflection(
            familiar_id="fam",
            channel_id=None,
            text="A pattern across channels.",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        assert r.channel_id is None


class TestRecentReflections:
    def test_recent_returns_newest_first(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(3):
            store.append_reflection(
                familiar_id="fam",
                channel_id=1,
                text=f"reflection {i}",
                cited_turn_ids=[i],
                cited_fact_ids=[],
                last_turn_id=i,
                last_fact_id=0,
            )
        recents = store.recent_reflections(familiar_id="fam", channel_id=1, limit=10)
        assert [r.text for r in recents] == [
            "reflection 2",
            "reflection 1",
            "reflection 0",
        ]

    def test_recent_scopes_by_familiar(self) -> None:
        store = HistoryStore(":memory:")
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="mine",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        store.append_reflection(
            familiar_id="other",
            channel_id=1,
            text="not mine",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        recents = store.recent_reflections(familiar_id="fam", channel_id=1, limit=10)
        assert [r.text for r in recents] == ["mine"]

    def test_channel_filter_includes_channel_agnostic_rows(self) -> None:
        store = HistoryStore(":memory:")
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="channel 1 reflection",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        store.append_reflection(
            familiar_id="fam",
            channel_id=None,
            text="global reflection",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        store.append_reflection(
            familiar_id="fam",
            channel_id=2,
            text="channel 2 reflection",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        ch1 = store.recent_reflections(familiar_id="fam", channel_id=1, limit=10)
        texts = {r.text for r in ch1}
        assert "channel 1 reflection" in texts
        assert "global reflection" in texts
        assert "channel 2 reflection" not in texts

    def test_unscoped_returns_all_channels(self) -> None:
        store = HistoryStore(":memory:")
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="ch1",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        store.append_reflection(
            familiar_id="fam",
            channel_id=2,
            text="ch2",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        recents = store.recent_reflections(familiar_id="fam", limit=10)
        assert len(recents) == 2

    def test_limit_caps_result_count(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(5):
            store.append_reflection(
                familiar_id="fam",
                channel_id=1,
                text=f"r{i}",
                cited_turn_ids=[i],
                cited_fact_ids=[],
                last_turn_id=i,
                last_fact_id=0,
            )
        recents = store.recent_reflections(familiar_id="fam", channel_id=1, limit=2)
        assert len(recents) == 2

    def test_limit_zero_returns_empty(self) -> None:
        store = HistoryStore(":memory:")
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="r",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=1,
            last_fact_id=0,
        )
        assert store.recent_reflections(familiar_id="fam", channel_id=1, limit=0) == []


class TestLatestReflectionWatermarks:
    def test_zeros_when_none_exist(self) -> None:
        store = HistoryStore(":memory:")
        assert store.latest_reflection_watermarks(familiar_id="fam") == (0, 0)

    def test_returns_newest_rows_watermarks(self) -> None:
        store = HistoryStore(":memory:")
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="older",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=10,
            last_fact_id=3,
        )
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text="newer",
            cited_turn_ids=[1],
            cited_fact_ids=[],
            last_turn_id=42,
            last_fact_id=11,
        )
        assert store.latest_reflection_watermarks(familiar_id="fam") == (42, 11)


class TestSupersededFactIds:
    def test_empty_input_returns_empty(self) -> None:
        store = HistoryStore(":memory:")
        assert store.superseded_fact_ids(familiar_id="fam", fact_ids=[]) == set()

    def test_returns_only_superseded_subset(self) -> None:
        store = HistoryStore(":memory:")
        ids = _seed_facts(store, count=3)
        # Supersede the second one: insert a replacement, then mark the
        # old row superseded by the new id.
        replacement = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="replacement fact",
            source_turn_ids=[ids[1]],
        )
        store.supersede_fact(familiar_id="fam", old_id=ids[1], new_id=replacement.id)
        result = store.superseded_fact_ids(familiar_id="fam", fact_ids=ids)
        assert result == {ids[1]}

    def test_ignores_other_familiars(self) -> None:
        store = HistoryStore(":memory:")
        # Seed a superseded fact under another familiar.
        f1 = store.append_fact(
            familiar_id="other",
            channel_id=1,
            text="other-fam fact",
            source_turn_ids=[1],
        )
        f2 = store.append_fact(
            familiar_id="other",
            channel_id=1,
            text="replacement",
            source_turn_ids=[1],
        )
        store.supersede_fact(familiar_id="other", old_id=f1.id, new_id=f2.id)
        # Query under "fam" — the superseded row is not visible.
        result = store.superseded_fact_ids(familiar_id="fam", fact_ids=[f1.id, f2.id])
        assert result == set()
