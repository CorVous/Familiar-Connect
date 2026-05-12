"""Tests for the ``people_dossiers`` table + store API.

Mirrors the ``summaries`` table shape: one row per
``(familiar_id, canonical_key)``, with a ``last_fact_id`` watermark.
The dossier is a compounded summary of facts whose ``subjects_json``
mentions the canonical key. Read path is a cheap SQLite lookup; write
path is owned by ``PeopleDossierWorker``.
"""

from __future__ import annotations

from familiar_connect.history.store import (
    FactSubject,
    HistoryStore,
    PeopleDossierEntry,
)


class TestPeopleDossierCRUD:
    def test_get_returns_none_when_absent(self) -> None:
        store = HistoryStore(":memory:")
        assert (
            store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
            is None
        )

    def test_put_then_get_roundtrips(self) -> None:
        store = HistoryStore(":memory:")
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=7,
            dossier_text="Cass likes pho. Lives in Toronto.",
        )
        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert isinstance(entry, PeopleDossierEntry)
        assert entry.last_fact_id == 7
        assert "pho" in entry.dossier_text

    def test_put_overwrites_existing(self) -> None:
        store = HistoryStore(":memory:")
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=3,
            dossier_text="v1",
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=10,
            dossier_text="v2",
        )
        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert entry is not None
        assert entry.last_fact_id == 10
        assert entry.dossier_text == "v2"

    def test_separate_familiars_dont_collide(self) -> None:
        store = HistoryStore(":memory:")
        store.put_people_dossier(
            familiar_id="famA",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="A's view",
        )
        store.put_people_dossier(
            familiar_id="famB",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="B's view",
        )
        a = store.get_people_dossier(familiar_id="famA", canonical_key="discord:1")
        b = store.get_people_dossier(familiar_id="famB", canonical_key="discord:1")
        assert a is not None
        assert b is not None
        assert a.dossier_text == "A's view"
        assert b.dossier_text == "B's view"


class TestSubjectsWithFacts:
    """The worker's "what changed" query.

    Returns ``{canonical_key: max(facts.id)}`` across facts whose
    ``subjects_json`` lists that key. Worker compares against each
    dossier's ``last_fact_id`` watermark to decide refresh-or-skip.
    """

    def test_empty_when_no_facts(self) -> None:
        store = HistoryStore(":memory:")
        assert store.subjects_with_facts(familiar_id="fam") == {}

    def test_skips_facts_without_subjects(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A subject-less fact.",
            source_turn_ids=[1],
        )
        assert store.subjects_with_facts(familiar_id="fam") == {}

    def test_returns_max_fact_id_per_subject(self) -> None:
        store = HistoryStore(":memory:")
        cass = FactSubject(canonical_key="discord:1", display_at_write="Cass")
        aria = FactSubject(canonical_key="discord:2", display_at_write="Aria")
        store.append_fact(  # id 1
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(cass,),
        )
        store.append_fact(  # id 2
            familiar_id="fam",
            channel_id=1,
            text="Aria likes ramen.",
            source_turn_ids=[2],
            subjects=(aria,),
        )
        store.append_fact(  # id 3
            familiar_id="fam",
            channel_id=1,
            text="Cass and Aria are roommates.",
            source_turn_ids=[3],
            subjects=(cass, aria),
        )
        latest = store.subjects_with_facts(familiar_id="fam")
        assert latest == {"discord:1": 3, "discord:2": 3}

    def test_scoped_to_familiar(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="famA",
            channel_id=1,
            text="x",
            source_turn_ids=[1],
            subjects=(FactSubject(canonical_key="discord:1", display_at_write="C"),),
        )
        store.append_fact(
            familiar_id="famB",
            channel_id=1,
            text="y",
            source_turn_ids=[1],
            subjects=(FactSubject(canonical_key="discord:2", display_at_write="A"),),
        )
        assert store.subjects_with_facts(familiar_id="famA") == {"discord:1": 1}
        assert store.subjects_with_facts(familiar_id="famB") == {"discord:2": 2}


class TestFactsForSubject:
    """Worker pulls these to compound a dossier from new facts.

    Returns facts whose ``subjects_json`` mentions the canonical key,
    ASC by id, optionally filtered by ``min_id_exclusive`` so the
    worker only re-summarises new evidence.
    """

    def test_returns_only_facts_mentioning_subject(self) -> None:
        store = HistoryStore(":memory:")
        cass = FactSubject(canonical_key="discord:1", display_at_write="Cass")
        aria = FactSubject(canonical_key="discord:2", display_at_write="Aria")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass fact.",
            source_turn_ids=[1],
            subjects=(cass,),
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria fact.",
            source_turn_ids=[2],
            subjects=(aria,),
        )
        out = store.facts_for_subject(familiar_id="fam", canonical_key="discord:1")
        assert [f.text for f in out] == ["Cass fact."]

    def test_filters_by_min_id_exclusive(self) -> None:
        store = HistoryStore(":memory:")
        cass = FactSubject(canonical_key="discord:1", display_at_write="Cass")
        for i in range(3):
            store.append_fact(
                familiar_id="fam",
                channel_id=1,
                text=f"Cass fact {i}.",
                source_turn_ids=[i],
                subjects=(cass,),
            )
        # min_id_exclusive=1 ⇒ only id 2, 3 returned.
        out = store.facts_for_subject(
            familiar_id="fam", canonical_key="discord:1", min_id_exclusive=1
        )
        assert [f.id for f in out] == [2, 3]

    def test_excludes_superseded_by_default(self) -> None:
        store = HistoryStore(":memory:")
        cass = FactSubject(canonical_key="discord:1", display_at_write="Cass")
        old = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass was a baker.",
            source_turn_ids=[1],
            subjects=(cass,),
        )
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass is a chef.",
            source_turn_ids=[2],
            subjects=(cass,),
        )
        store.supersede_fact(familiar_id="fam", old_id=old.id, new_id=new.id)
        out = store.facts_for_subject(familiar_id="fam", canonical_key="discord:1")
        assert [f.text for f in out] == ["Cass is a chef."]

    def test_returns_facts_where_subject_is_one_of_many(self) -> None:
        store = HistoryStore(":memory:")
        cass = FactSubject(canonical_key="discord:1", display_at_write="Cass")
        aria = FactSubject(canonical_key="discord:2", display_at_write="Aria")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass and Aria are roommates.",
            source_turn_ids=[1],
            subjects=(cass, aria),
        )
        out = store.facts_for_subject(familiar_id="fam", canonical_key="discord:2")
        assert len(out) == 1
        assert "roommates" in out[0].text
