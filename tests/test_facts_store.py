"""Tests for facts table + FTS over facts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from familiar_connect.history.store import (
    Fact,
    FactDraft,
    FactSubject,
    HistoryStore,
    _canonical_keys_from_subjects_json,
)


class TestCanonicalKeysFromSubjectsJson:
    """Shared parser for ``subjects_json`` → canonical str keys."""

    def test_none_and_empty(self) -> None:
        assert _canonical_keys_from_subjects_json(None) == frozenset()
        assert _canonical_keys_from_subjects_json("") == frozenset()

    def test_unparseable_or_non_list(self) -> None:
        assert _canonical_keys_from_subjects_json("{not json") == frozenset()
        assert _canonical_keys_from_subjects_json('{"a": 1}') == frozenset()

    def test_extracts_str_keys_skips_malformed(self) -> None:
        blob = (
            '[{"canonical_key": "discord:1", "display_at_write": "Cor"},'
            ' {"canonical_key": 5},'  # non-str key skipped
            ' "not a dict",'  # non-dict skipped
            ' {"display_at_write": "no key"},'  # missing key skipped
            ' {"canonical_key": "discord:2", "display_at_write": "Aria"}]'
        )
        assert _canonical_keys_from_subjects_json(blob) == frozenset({
            "discord:1",
            "discord:2",
        })


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
        store.supersede(familiar_id="fam", obsolete_facts=[1], new_fact=new.id)
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
        store.supersede(familiar_id="fam", obsolete_facts=[1], new_fact=new.id)
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
        store.supersede(familiar_id="fam", obsolete_facts=[1], new_fact=new.id)
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
        store.supersede(familiar_id="fam", obsolete_facts=[1], new_fact=new.id)
        found = store.search_facts(familiar_id="fam", query="strawberry", limit=10)
        texts = {f.text for f in found}
        assert "Aria likes strawberries." not in texts
        assert "Aria is allergic to strawberries now." in texts


class TestSupersedeInvalidatesDossier:
    """Superseding a fact must drop affected subjects' baked dossiers.

    PeopleDossierWorker compounds prior dossier text + new facts; it
    never un-bakes a superseded fact. Deleting the row forces the
    worker's prior=None clean rebuild.
    """

    def _store_with_subject_fact(self) -> tuple[HistoryStore, int, int]:
        """Store + (old_id, new_id) for subject ``discord:A``."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=None,
        )
        subjects = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)
        old = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria loves hiking.",
            source_turn_ids=[1],
            subjects=subjects,
        )
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria hates hiking now.",
            source_turn_ids=[1],
            subjects=subjects,
        )
        return store, old.id, new.id

    def test_supersede_deletes_subject_dossier(self) -> None:
        store, old_id, new_id = self._store_with_subject_fact()
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:A",
            last_fact_id=old_id,
            dossier_text="Aria loves hiking.",
        )
        store.supersede(familiar_id="fam", obsolete_facts=[old_id], new_fact=new_id)
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:A")
            is None
        )

    def test_supersede_null_subject_leaves_dossiers(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=None,
        )
        old = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="The weather is nice.",
            source_turn_ids=[1],
        )
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="The weather turned grim.",
            source_turn_ids=[1],
        )
        # Unrelated dossier must survive a subjectless supersede.
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:A",
            last_fact_id=1,
            dossier_text="Aria loves hiking.",
        )
        store.supersede(familiar_id="fam", obsolete_facts=[old.id], new_fact=new.id)
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:A")
            is not None
        )

    def test_supersede_spares_other_subjects_dossier(self) -> None:
        store, old_id, new_id = self._store_with_subject_fact()
        # Subject B's dossier is unrelated to A's superseded fact.
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:A",
            last_fact_id=old_id,
            dossier_text="Aria loves hiking.",
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:B",
            last_fact_id=old_id,
            dossier_text="Boris works nights.",
        )
        store.supersede(familiar_id="fam", obsolete_facts=[old_id], new_fact=new_id)
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:A")
            is None
        )
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:B")
            is not None
        )

    def test_supersede_spares_other_familiars_dossier(self) -> None:
        store, old_id, new_id = self._store_with_subject_fact()
        # Same canonical key, different familiar — must not be touched.
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:A",
            last_fact_id=old_id,
            dossier_text="Aria loves hiking.",
        )
        store.put_people_dossier(
            familiar_id="other",
            canonical_key="discord:A",
            last_fact_id=old_id,
            dossier_text="Aria, per other familiar.",
        )
        store.supersede(familiar_id="fam", obsolete_facts=[old_id], new_fact=new_id)
        assert (
            store.get_people_dossier(familiar_id="other", canonical_key="discord:A")
            is not None
        )


class TestRetireForm:
    """Retire form (``supersede(new_fact=None)``) read + dossier effects.

    Metadata + skip-stale semantics live in ``TestSupersedeUnified``;
    here we pin the read-path exclusion and dossier invalidation a
    no-replacement retire triggers.
    """

    def _store_with_fact(self) -> tuple[HistoryStore, int]:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam", channel_id=1, role="user", content="hi", author=None
        )
        f = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Noise that should not have been minted.",
            source_turn_ids=[1],
        )
        return store, f.id

    def test_retired_fact_excluded_from_current(self) -> None:
        store, fid = self._store_with_fact()
        store.supersede(familiar_id="fam", obsolete_facts=[fid], new_fact=None)
        assert store.recent_facts(familiar_id="fam", limit=10) == []

    def test_retire_invalidates_subject_dossier(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam", channel_id=1, role="user", content="hi", author=None
        )
        subjects = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)
        f = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria once claimed a thing — bit, not fact.",
            source_turn_ids=[1],
            subjects=subjects,
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:A",
            last_fact_id=f.id,
            dossier_text="Aria claimed a thing.",
        )
        store.supersede(familiar_id="fam", obsolete_facts=[f.id], new_fact=None)
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:A")
            is None
        )


class TestSupersedeUnified:
    """Unified ``supersede`` — one mutation surface over retire + merge.

    Subsumes retire (``new_fact=None``) and the apply.py
    decomposition of merge (``new_fact`` a draft to mint, or an
    already-minted fact/id). The store owns merge lineage +
    provenance: obsolete rows point at the replacement (many->one),
    and the minted fact's ``source_turn_ids`` is the union of the
    obsolete rows' provenance — the caller never supplies it.
    """

    def _store(self) -> HistoryStore:
        store = HistoryStore(":memory:")
        for i in range(6):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"turn {i}",
                author=None,
            )
        return store

    def test_retire_form_marks_superseded_without_replacement(self) -> None:
        store = self._store()
        f = store.append_fact(
            familiar_id="fam", channel_id=1, text="junk.", source_turn_ids=[1]
        )
        store.supersede(familiar_id="fam", obsolete_facts=[f.id], new_fact=None)
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        row = next(r for r in all_facts if r.id == f.id)
        assert row.superseded_at is not None
        assert row.superseded_by is None

    def test_merge_points_all_obsolete_at_minted_many_to_one(self) -> None:
        store = self._store()
        a = store.append_fact(
            familiar_id="fam", channel_id=1, text="Aria likes A.", source_turn_ids=[1]
        )
        b = store.append_fact(
            familiar_id="fam", channel_id=1, text="Aria likes B.", source_turn_ids=[2]
        )
        c = store.append_fact(
            familiar_id="fam", channel_id=1, text="Aria likes C.", source_turn_ids=[3]
        )
        result = store.supersede(
            familiar_id="fam",
            obsolete_facts=[a.id, b.id, c.id],
            new_fact=FactDraft(
                channel_id=1, text="Aria likes A, B, and C.", subjects=()
            ),
        )
        minted = result.minted
        assert minted is not None
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        by_id = {r.id: r for r in all_facts}
        assert by_id[a.id].superseded_by == minted.id
        assert by_id[b.id].superseded_by == minted.id
        assert by_id[c.id].superseded_by == minted.id
        # minted row is current + FTS-searchable
        current = store.recent_facts(familiar_id="fam", limit=10)
        assert minted.id in {r.id for r in current}
        found = store.search_facts(familiar_id="fam", query="Aria", limit=10)
        assert minted.id in {r.id for r in found}

    def test_store_owns_ancestry_resolution(self) -> None:
        store = self._store()
        a = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact A.", source_turn_ids=[1]
        )
        b = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact B.", source_turn_ids=[2]
        )
        c = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact C.", source_turn_ids=[3]
        )
        result = store.supersede(
            familiar_id="fam",
            obsolete_facts=[a.id, b.id, c.id],
            new_fact=FactDraft(channel_id=1, text="merged.", subjects=()),
        )
        assert result.minted is not None
        ancestors = store.ancestors_of(familiar_id="fam", fact_id=result.minted.id)
        assert {f.id for f in ancestors} == {a.id, b.id, c.id}

    def test_store_owns_provenance_union(self) -> None:
        store = self._store()
        a = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact A.", source_turn_ids=[1, 2]
        )
        b = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact B.", source_turn_ids=[2, 3]
        )
        c = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact C.", source_turn_ids=[4]
        )
        result = store.supersede(
            familiar_id="fam",
            obsolete_facts=[a.id, b.id, c.id],
            # caller supplies NO source_turn_ids — store computes the union
            new_fact=FactDraft(channel_id=1, text="merged.", subjects=()),
        )
        assert result.minted is not None
        assert set(result.minted.source_turn_ids) == {1, 2, 3, 4}

    def test_existing_fact_form_mints_no_new_row(self) -> None:
        store = self._store()
        x = store.append_fact(
            familiar_id="fam", channel_id=1, text="old.", source_turn_ids=[1]
        )
        existing = store.append_fact(
            familiar_id="fam", channel_id=1, text="replacement.", source_turn_ids=[2]
        )
        before = store.latest_fact_id(familiar_id="fam")
        result = store.supersede(
            familiar_id="fam", obsolete_facts=[x.id], new_fact=existing
        )
        assert store.latest_fact_id(familiar_id="fam") == before  # no new row
        assert result.minted is None
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        row = next(r for r in all_facts if r.id == x.id)
        assert row.superseded_by == existing.id

    def test_merge_declined_atomically_when_any_obsolete_stale(self) -> None:
        store = self._store()
        a = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact A.", source_turn_ids=[1]
        )
        b = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact B.", source_turn_ids=[2]
        )
        before = store.latest_fact_id(familiar_id="fam")
        # a concurrent writer already retired `a` before our merge runs
        store.supersede(familiar_id="fam", obsolete_facts=[a.id], new_fact=None)
        result = store.supersede(
            familiar_id="fam",
            obsolete_facts=[a.id, b.id],
            new_fact=FactDraft(channel_id=1, text="merged.", subjects=()),
        )
        # merge declined whole: no phantom minted, nothing newly superseded
        assert result.minted is None
        assert result.superseded == ()
        assert store.latest_fact_id(familiar_id="fam") == before  # no new row
        # the still-current `b` is untouched — not superseded by a phantom
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        by_id = {r.id: r for r in all_facts}
        assert by_id[b.id].superseded_at is None
        assert by_id[b.id].superseded_by is None
        # the stale `a` is recorded as the reason the merge was declined
        assert a.id in {sid for sid, _ in result.skipped}

    def test_merge_empty_obsolete_is_noop(self) -> None:
        store = self._store()
        before = store.latest_fact_id(familiar_id="fam")
        result = store.supersede(
            familiar_id="fam",
            obsolete_facts=[],
            new_fact=FactDraft(channel_id=1, text="orphan merge.", subjects=()),
        )
        assert result.minted is None
        assert result.superseded == ()
        assert store.latest_fact_id(familiar_id="fam") == before  # no new row

    def test_retire_form_skips_stale_processes_rest(self) -> None:
        store = self._store()
        a = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact A.", source_turn_ids=[1]
        )
        b = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact B.", source_turn_ids=[2]
        )
        # a concurrent writer already retired `a` before our retire runs
        store.supersede(familiar_id="fam", obsolete_facts=[a.id], new_fact=None)
        # retire form (new_fact=None) is per-id skip-and-record, NOT atomic
        result = store.supersede(
            familiar_id="fam", obsolete_facts=[a.id, b.id], new_fact=None
        )
        # b is retired; a is skipped (already gone), not fatal
        assert b.id in result.superseded
        assert a.id not in result.superseded
        assert a.id in {sid for sid, _ in result.skipped}
        assert result.minted is None
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        by_id = {r.id: r for r in all_facts}
        assert by_id[b.id].superseded_at is not None
        assert by_id[b.id].superseded_by is None

    def test_existing_id_form_skips_stale_processes_rest(self) -> None:
        store = self._store()
        a = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact A.", source_turn_ids=[1]
        )
        b = store.append_fact(
            familiar_id="fam", channel_id=1, text="fact B.", source_turn_ids=[2]
        )
        existing = store.append_fact(
            familiar_id="fam", channel_id=1, text="replacement.", source_turn_ids=[3]
        )
        # a concurrent writer already retired `a` before our repoint runs
        store.supersede(familiar_id="fam", obsolete_facts=[a.id], new_fact=None)
        before = store.latest_fact_id(familiar_id="fam")
        # existing-id form is per-id skip-and-record, NOT atomic, mints nothing
        result = store.supersede(
            familiar_id="fam", obsolete_facts=[a.id, b.id], new_fact=existing
        )
        assert store.latest_fact_id(familiar_id="fam") == before  # no new row
        assert result.minted is None
        assert b.id in result.superseded
        assert a.id not in result.superseded
        assert a.id in {sid for sid, _ in result.skipped}
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        by_id = {r.id: r for r in all_facts}
        assert by_id[b.id].superseded_by == existing.id

    def test_invalidates_dossier_per_obsolete_subject(self) -> None:
        store = self._store()
        subj_a = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)
        subj_b = (FactSubject(canonical_key="discord:B", display_at_write="Boris"),)
        a = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria thing.",
            source_turn_ids=[1],
            subjects=subj_a,
        )
        b = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Boris thing.",
            source_turn_ids=[2],
            subjects=subj_b,
        )
        for key in ("discord:A", "discord:B"):
            store.put_people_dossier(
                familiar_id="fam",
                canonical_key=key,
                last_fact_id=a.id,
                dossier_text="baked.",
            )
        store.supersede(
            familiar_id="fam",
            obsolete_facts=[a.id, b.id],
            new_fact=FactDraft(channel_id=1, text="merged.", subjects=()),
        )
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:A")
            is None
        )
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:B")
            is None
        )


class TestFactsByIds:
    """Exact fetch by id set — used by sleep-apply to snapshot sources."""

    def test_fetches_requested_including_superseded(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam", channel_id=1, text="replacement", source_turn_ids=[5]
        )
        store.supersede(familiar_id="fam", obsolete_facts=[1], new_fact=new.id)
        got = store.facts_by_ids(familiar_id="fam", ids=[1, 2])
        assert {f.id for f in got} == {1, 2}
        # superseded row still returned (snapshot must see it)
        assert next(f for f in got if f.id == 1).superseded_at is not None

    def test_empty_and_unknown(self) -> None:
        store = _store_with_turns_and_facts()
        assert store.facts_by_ids(familiar_id="fam", ids=[]) == []
        assert store.facts_by_ids(familiar_id="fam", ids=[999]) == []

    def test_scoped_to_familiar(self) -> None:
        store = _store_with_turns_and_facts()
        store.append_fact(
            familiar_id="other", channel_id=1, text="x", source_turn_ids=[1]
        )
        got = store.facts_by_ids(familiar_id="other", ids=[1, 2, 3])
        assert all(f.familiar_id == "other" for f in got)


class TestSleepWatermark:
    """Two-axis watermark: hygiene owns fact id, dream owns turn id.

    ``advance_sleep_watermark`` is partial-update — each pass touches
    only its own axis so neither can clobber the other's progress.
    """

    def test_get_unset_returns_none(self) -> None:
        store = HistoryStore(":memory:")
        assert store.get_sleep_watermark(familiar_id="fam") is None

    def test_advance_inserts_with_zero_for_omitted_axis(self) -> None:
        store = HistoryStore(":memory:")
        store.advance_sleep_watermark(familiar_id="fam", last_fact_id=42)
        wm = store.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        assert (wm.last_fact_id, wm.last_turn_id) == (42, 0)

    def test_advance_updates_only_named_axis(self) -> None:
        store = HistoryStore(":memory:")
        store.advance_sleep_watermark(familiar_id="fam", last_fact_id=42)
        # dream advances turn axis — fact axis must be untouched
        store.advance_sleep_watermark(familiar_id="fam", last_turn_id=99)
        wm = store.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        assert (wm.last_fact_id, wm.last_turn_id) == (42, 99)

    def test_advance_both_axes(self) -> None:
        store = HistoryStore(":memory:")
        store.advance_sleep_watermark(
            familiar_id="fam", last_fact_id=10, last_turn_id=20
        )
        wm = store.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        assert (wm.last_fact_id, wm.last_turn_id) == (10, 20)

    def test_advance_noop_when_both_none(self) -> None:
        store = HistoryStore(":memory:")
        store.advance_sleep_watermark(familiar_id="fam")
        assert store.get_sleep_watermark(familiar_id="fam") is None


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
        store.supersede(familiar_id="fam", obsolete_facts=[old.id], new_fact=new.id)
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
        store.supersede(
            familiar_id="fam", obsolete_facts=[old.id], new_fact=replacement.id
        )
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
