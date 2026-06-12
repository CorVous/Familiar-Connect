"""Tests for facts table + FTS over facts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from familiar_connect.history.store import Fact, FactSubject, HistoryStore


def _store_with_turns_and_facts() -> HistoryStore:
    store = HistoryStore(":memory:")
    for i in range(5):
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content=f"turn text {i}",
            author=None,
        )
    store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria likes strawberries.",
        source_turn_ids=[1, 2],
    )
    store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Boris works night shifts on Tuesdays.",
        source_turn_ids=[3, 4],
    )
    return store


class TestFactStore:
    def test_append_returns_fact_with_provenance(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A fact.",
            source_turn_ids=[7, 9, 11],
        )
        assert isinstance(fact, Fact)
        assert fact.source_turn_ids == (7, 9, 11)
        assert fact.text == "A fact."
        assert fact.channel_id == 1
        assert fact.id > 0

    def test_recent_facts_ordered_newest_first(self) -> None:
        store = _store_with_turns_and_facts()
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert len(recents) == 2
        assert recents[0].text.startswith("Boris")
        assert recents[1].text.startswith("Aria")

    def test_search_facts_finds_by_content(self) -> None:
        store = _store_with_turns_and_facts()
        found = store.search_facts(familiar_id="fam", query="strawberry", limit=5)
        # English stemmer maps "strawberry"/"strawberries" to a shared stem.
        assert len(found) == 1
        assert "strawberries" in found[0].text

    def test_search_respects_familiar(self) -> None:
        store = _store_with_turns_and_facts()
        store.append_fact(
            familiar_id="other",
            channel_id=1,
            text="Other familiar knows strawberries too.",
            source_turn_ids=[1],
        )
        found = store.search_facts(familiar_id="fam", query="strawberry", limit=10)
        # Only the "fam" fact returned
        assert len(found) == 1
        assert found[0].familiar_id == "fam"

    def test_latest_fact_id(self) -> None:
        store = _store_with_turns_and_facts()
        assert store.latest_fact_id(familiar_id="fam") == 2
        assert store.latest_fact_id(familiar_id="nobody") == 0

    def test_empty_query_returns_nothing(self) -> None:
        store = _store_with_turns_and_facts()
        assert store.search_facts(familiar_id="fam", query="", limit=10) == []

    def test_source_turn_ids_roundtrip(self) -> None:
        store = _store_with_turns_and_facts()
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].source_turn_ids == (3, 4)
        assert recents[1].source_turn_ids == (1, 2)


class TestFactSubjects:
    """Subject-key metadata: best-effort link from fact text to canonical_keys.

    The extractor's identification of *who* a fact is about is a soft
    annotation, not authoritative — mic-sharing, relayed quotes, and
    plain ambiguity all break a 1:1 mapping. Storage must preserve
    whatever the extractor emits (zero, one, or many subjects per
    fact) and surface it back unchanged.
    """

    def test_append_with_subjects_roundtrip(self) -> None:
        store = HistoryStore(":memory:")
        subjects = (
            FactSubject(canonical_key="discord:123", display_at_write="Cass"),
            FactSubject(canonical_key="discord:456", display_at_write="Aria"),
        )
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass and Aria are roommates.",
            source_turn_ids=[1],
            subjects=subjects,
        )
        assert fact.subjects == subjects
        # And again on re-read.
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].subjects == subjects

    def test_subjects_default_empty(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A fact with no identified subjects.",
            source_turn_ids=[1],
        )
        assert fact.subjects == ()

    def test_search_facts_returns_subjects(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(
                FactSubject(canonical_key="discord:123", display_at_write="Cass"),
            ),
        )
        found = store.search_facts(familiar_id="fam", query="pho", limit=5)
        assert len(found) == 1
        assert found[0].subjects == (
            FactSubject(canonical_key="discord:123", display_at_write="Cass"),
        )


class TestFactSupersession:
    """Version chains: facts that go stale are superseded, not overwritten.

    A new fact replaces the old; the old keeps its row with
    ``superseded_at`` set and ``superseded_by`` pointing at the new
    row. Reads default to current (non-superseded), but the history
    is preserved for audit and contradiction inspection.
    """

    def test_new_facts_are_current_by_default(self) -> None:
        store = _store_with_turns_and_facts()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        for f in facts:
            assert f.superseded_at is None
            assert f.superseded_by is None

    def test_supersede_marks_old_and_links_new(self) -> None:
        store = _store_with_turns_and_facts()
        # Replace the strawberry fact with an updated one.
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        # Reload via recent (include superseded so we can inspect the old row)
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        old = next(f for f in all_facts if f.id == 1)
        assert old.superseded_at is not None
        assert old.superseded_by == new.id

    def test_recent_facts_excludes_superseded_by_default(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        current = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in current}
        assert "Aria likes strawberries." not in texts
        assert "Aria is allergic to strawberries now." in texts

    def test_recent_facts_includes_superseded_when_asked(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        texts = {f.text for f in all_facts}
        assert "Aria likes strawberries." in texts
        assert "Aria is allergic to strawberries now." in texts

    def test_search_facts_excludes_superseded(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        found = store.search_facts(familiar_id="fam", query="strawberry", limit=10)
        texts = {f.text for f in found}
        assert "Aria likes strawberries." not in texts
        assert "Aria is allergic to strawberries now." in texts

    def test_supersede_idempotent_raises_on_already_superseded(self) -> None:
        """Supersession is one-shot — re-superseding signals a bug upstream."""
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="updated.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        with pytest.raises(ValueError, match="already superseded"):
            store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)

    def test_supersede_unknown_id_raises(self) -> None:
        store = _store_with_turns_and_facts()
        with pytest.raises(ValueError, match="unknown fact"):
            store.supersede_fact(familiar_id="fam", old_id=999, new_id=1)


class TestFactBiTemporal:
    """Bi-temporal validity: ``valid_from`` / ``valid_to`` (M1 — world-time).

    ``superseded_at`` captures *when we recorded the change* (system-
    time). ``valid_from`` / ``valid_to`` capture *when the fact applied
    in the world* (world-time). Default reads stay "current truth";
    ``as_of=...`` unlocks audit queries against world-time.
    """

    def test_append_defaults_valid_from_to_now_when_no_source_turns(self) -> None:
        store = HistoryStore(":memory:")
        before = datetime.now(tz=UTC)
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A fact.",
            source_turn_ids=[],
        )
        after = datetime.now(tz=UTC)
        assert fact.valid_from is not None
        assert before <= fact.valid_from <= after
        assert fact.valid_to is None

    def test_append_with_explicit_valid_from(self) -> None:
        store = HistoryStore(":memory:")
        when = datetime(2025, 6, 1, tzinfo=UTC)
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria moved to Berlin.",
            source_turn_ids=[1],
            valid_from=when,
        )
        assert fact.valid_from == when
        assert fact.valid_to is None
        # And on re-read.
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].valid_from == when

    def test_append_with_explicit_valid_to(self) -> None:
        store = HistoryStore(":memory:")
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme in 2025.",
            source_turn_ids=[1],
            valid_from=start,
            valid_to=end,
        )
        assert fact.valid_from == start
        assert fact.valid_to == end

    def test_recent_facts_default_excludes_expired(self) -> None:
        """A fact whose ``valid_to`` is in the past is not "current truth"."""
        store = HistoryStore(":memory:")
        past = datetime.now(tz=UTC) - timedelta(days=30)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            valid_from=past - timedelta(days=365),
            valid_to=past,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            valid_from=past,
        )
        current = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in current}
        assert texts == {"Aria works at Globex."}

    def test_as_of_returns_world_time_slice(self) -> None:
        store = HistoryStore(":memory:")
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        t3 = datetime(2025, 9, 1, tzinfo=UTC)
        # Fact applied Jan-Jun.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            valid_from=t1,
            valid_to=t2,
        )
        # Fact applies from Jun onward.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            valid_from=t2,
        )
        # As of March: only Acme.
        march = datetime(2025, 3, 1, tzinfo=UTC)
        slice_march = store.recent_facts(familiar_id="fam", limit=10, as_of=march)
        assert {f.text for f in slice_march} == {"Aria worked at Acme."}
        # As of September: only Globex.
        slice_sept = store.recent_facts(familiar_id="fam", limit=10, as_of=t3)
        assert {f.text for f in slice_sept} == {"Aria works at Globex."}

    def test_as_of_includes_superseded_for_audit(self) -> None:
        """``as_of`` is an audit query — superseded rows participate."""
        store = HistoryStore(":memory:")
        old_time = datetime(2025, 1, 1, tzinfo=UTC)
        old = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[1],
            valid_from=old_time,
        )
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries.",
            source_turn_ids=[2],
            valid_from=datetime(2025, 6, 1, tzinfo=UTC),
        )
        store.supersede_fact(familiar_id="fam", old_id=old.id, new_id=new.id)
        # As of February the world believed Aria liked strawberries.
        feb = datetime(2025, 2, 1, tzinfo=UTC)
        slice_feb = store.recent_facts(familiar_id="fam", limit=10, as_of=feb)
        texts = {f.text for f in slice_feb}
        assert "Aria likes strawberries." in texts

    def test_search_facts_supports_as_of(self) -> None:
        store = HistoryStore(":memory:")
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            valid_from=t1,
            valid_to=t2,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            valid_from=t2,
        )
        march = datetime(2025, 3, 1, tzinfo=UTC)
        found = store.search_facts(
            familiar_id="fam", query="Aria", limit=10, as_of=march
        )
        assert {f.text for f in found} == {"Aria worked at Acme."}

    def test_facts_for_subject_supports_as_of(self) -> None:
        store = HistoryStore(":memory:")
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        subject = (FactSubject(canonical_key="discord:1", display_at_write="Aria"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            subjects=subject,
            valid_from=t1,
            valid_to=t2,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            subjects=subject,
            valid_from=t2,
        )
        march = datetime(2025, 3, 1, tzinfo=UTC)
        slice_march = store.facts_for_subject(
            familiar_id="fam", canonical_key="discord:1", as_of=march
        )
        assert {f.text for f in slice_march} == {"Aria worked at Acme."}


class TestFactImportance:
    """Importance score (M2): 1-10 hint for retrieval ranking.

    Persisted alongside each fact; consumed by ``RagContextLayer``.
    Optional — legacy rows and extractions that omit it read back as
    ``None`` and are treated as the neutral midpoint at rank time.
    """

    def test_append_with_importance_roundtrip(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to peanuts.",
            source_turn_ids=[1],
            importance=9,
        )
        assert fact.importance == 9
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].importance == 9

    def test_importance_defaults_none(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Casual aside.",
            source_turn_ids=[1],
        )
        assert fact.importance is None

    def test_importance_clamped_to_range(self) -> None:
        """Out-of-range values clamp to [1, 10] so the LLM can't poison ranks."""
        store = HistoryStore(":memory:")
        low = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="too low.",
            source_turn_ids=[1],
            importance=0,
        )
        high = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="too high.",
            source_turn_ids=[2],
            importance=42,
        )
        assert low.importance == 1
        assert high.importance == 10

    def test_search_facts_scored_returns_bm25(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[1],
            importance=8,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Boris likes strawberries too.",
            source_turn_ids=[2],
            importance=2,
        )
        scored = store.search_facts_scored(
            familiar_id="fam", query="strawberry", limit=5
        )
        assert len(scored) == 2
        # bm25 is a real number; both rows score (tantivy returns positive
        # numbers, higher = better — exposed verbatim so callers can rerank).
        for fact, score in scored:
            assert isinstance(fact, Fact)
            assert isinstance(score, float)
        # Order matches default search_facts (BM25 desc = best first).
        assert scored[0][1] <= scored[1][1]


class TestFactDedup:
    """Conservative near-duplicate suppression at insert.

    Alias/nickname restatements ("Cor is called X", "Cor goes by X")
    piled up — one current fact per turn batch. Guard at insert:
    normalized-text exact-match scoped to same subject-key set +
    same familiar skips the redundant row. No supersede, no mutate.
    """

    def test_normalized_duplicate_skips_insert(self) -> None:
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:1", display_at_write="Cor"),)
        first = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Postbirb Prime is called Cor.",
            source_turn_ids=[1],
            subjects=subj,
        )
        # case + whitespace + surrounding-quote/punct variants normalize equal.
        again = store.append_fact(
            familiar_id="fam",
            channel_id=2,
            text='  "postbirb   prime is called cor"  ',
            source_turn_ids=[2],
            subjects=subj,
        )
        assert store.all_fact_ids(familiar_id="fam") == {first.id}
        assert again.id == first.id

    def test_differing_text_inserts(self) -> None:
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:1", display_at_write="Cor"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor likes tea.",
            source_turn_ids=[1],
            subjects=subj,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor likes coffee.",
            source_turn_ids=[2],
            subjects=subj,
        )
        assert len(store.all_fact_ids(familiar_id="fam")) == 2

    def test_same_text_differing_subjects_inserts(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="likes tea.",
            source_turn_ids=[1],
            subjects=(FactSubject(canonical_key="discord:1", display_at_write="Cor"),),
        )
        # same normalized text, different subject — distinct fact.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="likes tea.",
            source_turn_ids=[2],
            subjects=(FactSubject(canonical_key="discord:2", display_at_write="Aria"),),
        )
        assert len(store.all_fact_ids(familiar_id="fam")) == 2

    def test_null_subject_not_dup_of_keyed_subject(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="likes tea.",
            source_turn_ids=[1],
            subjects=(FactSubject(canonical_key="discord:1", display_at_write="Cor"),),
        )
        # subjectless fact with same text is NOT a dup of the keyed one.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="likes tea.",
            source_turn_ids=[2],
        )
        assert len(store.all_fact_ids(familiar_id="fam")) == 2

    def test_dup_scoped_to_familiar(self) -> None:
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:1", display_at_write="Cor"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor likes tea.",
            source_turn_ids=[1],
            subjects=subj,
        )
        store.append_fact(
            familiar_id="other",
            channel_id=1,
            text="Cor likes tea.",
            source_turn_ids=[2],
            subjects=subj,
        )
        assert len(store.all_fact_ids(familiar_id="fam")) == 1
        assert len(store.all_fact_ids(familiar_id="other")) == 1

    def test_superseded_match_does_not_block_insert(self) -> None:
        """Only CURRENT facts dedup — a superseded twin must not block."""
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:1", display_at_write="Cor"),)
        old = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor likes tea.",
            source_turn_ids=[1],
            subjects=subj,
        )
        replacement = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor prefers coffee.",
            source_turn_ids=[2],
            subjects=subj,
        )
        store.supersede_fact(familiar_id="fam", old_id=old.id, new_id=replacement.id)
        # re-stating the retired fact inserts a fresh current row.
        again = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor likes tea.",
            source_turn_ids=[3],
            subjects=subj,
        )
        assert again.id not in {old.id, replacement.id}

    def test_valid_to_bypasses_dedup(self) -> None:
        """A fact carrying ``valid_to`` may close/bound an existing one.

        Such inserts must never be swallowed as dups.
        """
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:1", display_at_write="Cor"),)
        first = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor lives in Berlin.",
            source_turn_ids=[1],
            subjects=subj,
        )
        bounding = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cor lives in Berlin.",
            source_turn_ids=[2],
            subjects=subj,
            valid_to=datetime(2025, 6, 1, tzinfo=UTC),
        )
        assert bounding.id != first.id
        assert len(store.all_fact_ids(familiar_id="fam")) == 2

    def test_quote_wrapped_variant_dedups(self) -> None:
        """Internal quotes normalize away so ``called 'Cor'`` == ``called Cor``."""
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:1", display_at_write="Cor"),)
        first = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Postbirb Prime is called Cor.",
            source_turn_ids=[1],
            subjects=subj,
        )
        again = store.append_fact(
            familiar_id="fam",
            channel_id=2,
            text="Postbirb Prime is called 'Cor'.",
            source_turn_ids=[2],
            subjects=subj,
        )
        assert again.id == first.id
        assert store.all_fact_ids(familiar_id="fam") == {first.id}
