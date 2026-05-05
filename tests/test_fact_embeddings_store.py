"""Tests for the fact_embeddings side-index (M6).

Covers persistence, the ``unembedded_facts`` watermark query, and the
``model``-keyed dimension that lets a model swap accumulate beside
prior runs without destroying audit history.
"""

from __future__ import annotations

import math

from familiar_connect.history.store import HistoryStore


def _store_with_facts(n: int = 3) -> HistoryStore:
    store = HistoryStore(":memory:")
    for i in range(n):
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text=f"fact number {i}",
            source_turn_ids=[i],
        )
    return store


class TestFactEmbeddingPersistence:
    def test_set_then_get_round_trips_within_float32_precision(self) -> None:
        store = _store_with_facts(1)
        vec = [0.1, -0.2, 0.3, 0.4]
        store.set_fact_embedding(fact_id=1, model="hash-v1", vector=vec)
        got = store.get_fact_embeddings(fact_ids=[1], model="hash-v1")
        assert set(got) == {1}
        for a, b in zip(got[1], vec, strict=True):
            assert math.isclose(a, b, abs_tol=1e-6)

    def test_set_rejects_empty_vector(self) -> None:
        store = _store_with_facts(1)
        try:
            store.set_fact_embedding(fact_id=1, model="hash-v1", vector=[])
        except ValueError:
            return
        msg = "expected ValueError on empty vector"
        raise AssertionError(msg)

    def test_upsert_overwrites_same_pk(self) -> None:
        store = _store_with_facts(1)
        store.set_fact_embedding(fact_id=1, model="hash-v1", vector=[1.0, 2.0])
        store.set_fact_embedding(fact_id=1, model="hash-v1", vector=[3.0, 4.0, 5.0])
        got = store.get_fact_embeddings(fact_ids=[1], model="hash-v1")
        assert len(got[1]) == 3
        assert math.isclose(got[1][2], 5.0, abs_tol=1e-6)

    def test_different_models_keep_separate_rows(self) -> None:
        store = _store_with_facts(1)
        store.set_fact_embedding(fact_id=1, model="hash-v1", vector=[0.1, 0.2])
        store.set_fact_embedding(fact_id=1, model="bge-small", vector=[0.5, 0.6, 0.7])
        a = store.get_fact_embeddings(fact_ids=[1], model="hash-v1")
        b = store.get_fact_embeddings(fact_ids=[1], model="bge-small")
        assert len(a[1]) == 2
        assert len(b[1]) == 3

    def test_get_filters_by_model(self) -> None:
        store = _store_with_facts(1)
        store.set_fact_embedding(fact_id=1, model="model-a", vector=[1.0])
        assert store.get_fact_embeddings(fact_ids=[1], model="model-b") == {}

    def test_get_returns_only_requested_ids(self) -> None:
        store = _store_with_facts(3)
        for i in range(1, 4):
            store.set_fact_embedding(fact_id=i, model="m", vector=[float(i)])
        got = store.get_fact_embeddings(fact_ids=[1, 3], model="m")
        assert set(got) == {1, 3}

    def test_get_with_empty_ids_returns_empty_dict(self) -> None:
        store = _store_with_facts(0)
        assert store.get_fact_embeddings(fact_ids=[], model="m") == {}


class TestUnembeddedFacts:
    def test_returns_all_facts_when_no_embeddings_yet(self) -> None:
        store = _store_with_facts(3)
        pending = store.unembedded_facts(familiar_id="fam", model="m", limit=10)
        assert [f.id for f in pending] == [1, 2, 3]

    def test_skips_facts_already_embedded_for_model(self) -> None:
        store = _store_with_facts(3)
        store.set_fact_embedding(fact_id=2, model="m", vector=[0.1])
        pending = store.unembedded_facts(familiar_id="fam", model="m", limit=10)
        assert [f.id for f in pending] == [1, 3]

    def test_does_not_skip_when_only_a_different_model_embedded(self) -> None:
        store = _store_with_facts(2)
        store.set_fact_embedding(fact_id=1, model="other", vector=[0.1])
        pending = store.unembedded_facts(familiar_id="fam", model="m", limit=10)
        assert [f.id for f in pending] == [1, 2]

    def test_excludes_superseded_facts(self) -> None:
        store = _store_with_facts(2)
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=2)
        pending = store.unembedded_facts(familiar_id="fam", model="m", limit=10)
        assert [f.id for f in pending] == [2]

    def test_orders_by_id_ascending_for_deterministic_resume(self) -> None:
        store = _store_with_facts(5)
        pending = store.unembedded_facts(familiar_id="fam", model="m", limit=10)
        assert [f.id for f in pending] == sorted(f.id for f in pending)

    def test_respects_limit(self) -> None:
        store = _store_with_facts(5)
        pending = store.unembedded_facts(familiar_id="fam", model="m", limit=2)
        assert len(pending) == 2

    def test_zero_limit_returns_empty(self) -> None:
        store = _store_with_facts(2)
        assert store.unembedded_facts(familiar_id="fam", model="m", limit=0) == []


class TestLatestEmbeddedFactId:
    def test_returns_zero_when_no_embeddings(self) -> None:
        store = _store_with_facts(3)
        assert store.latest_embedded_fact_id(familiar_id="fam", model="m") == 0

    def test_returns_max_id_for_model(self) -> None:
        store = _store_with_facts(3)
        store.set_fact_embedding(fact_id=1, model="m", vector=[0.1])
        store.set_fact_embedding(fact_id=3, model="m", vector=[0.3])
        assert store.latest_embedded_fact_id(familiar_id="fam", model="m") == 3

    def test_isolates_models(self) -> None:
        store = _store_with_facts(3)
        store.set_fact_embedding(fact_id=2, model="a", vector=[0.1])
        store.set_fact_embedding(fact_id=3, model="b", vector=[0.1])
        assert store.latest_embedded_fact_id(familiar_id="fam", model="a") == 2
        assert store.latest_embedded_fact_id(familiar_id="fam", model="b") == 3
