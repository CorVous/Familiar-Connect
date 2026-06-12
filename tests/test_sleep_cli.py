"""Sleep CLI orchestrator — dry-run (default) vs --apply."""

from __future__ import annotations

import json

import pytest

import familiar_connect.sleep.passes as passes_mod
from familiar_connect.commands.sleep import execute_dream, execute_sleep
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import is_self_key
from tests.conftest import FakeLLMClient

ARIA = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)


def _store() -> HistoryStore:
    store = HistoryStore(":memory:")
    store.append_turn(
        familiar_id="fam", channel_id=1, role="user", content="hi", author=None
    )
    store.append_fact(
        familiar_id="fam", channel_id=1, text="noise", source_turn_ids=[1]
    )
    store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria likes tea.",
        source_turn_ids=[1],
        subjects=ARIA,
    )
    return store


def _reply() -> str:
    return json.dumps({
        "retire": [{"fact_ids": [1], "reason": "noise"}],
        "rewrite": [],
    })


class TestExecuteSleep:
    @pytest.mark.asyncio
    async def test_dry_run_writes_audit_without_mutating(self, tmp_path) -> None:  # noqa: ANN001
        raw = _store()
        store = AsyncHistoryStore(raw)
        plan, audit_path = await execute_sleep(
            store=store,
            llm=FakeLLMClient(replies=[_reply()]),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            audit_dir=tmp_path,
            apply=False,
        )
        # live facts untouched
        assert {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)} == {
            "noise",
            "Aria likes tea.",
        }
        # no watermark written on dry-run
        assert raw.get_sleep_watermark(familiar_id="fam") is None
        # audit on disk, applied=False
        audit = json.loads(audit_path.read_text())
        assert audit["applied"] is False
        assert len(audit["retire"]) == 1
        assert len(plan.retire) == 1

    @pytest.mark.asyncio
    async def test_audit_written_even_if_apply_raises(self, tmp_path) -> None:  # noqa: ANN001
        """F1: the run that mutated rows must never be the run with no audit."""
        raw = _store()
        store = AsyncHistoryStore(raw)

        async def _boom(*_a, **_k):  # noqa: ANN002, ANN003, RUF029
            raise RuntimeError("boom")

        original = passes_mod.apply_hygiene
        passes_mod.apply_hygiene = _boom  # ty: ignore[invalid-assignment]
        try:
            with pytest.raises(RuntimeError):
                await execute_sleep(
                    store=store,
                    llm=FakeLLMClient(replies=[_reply()]),
                    familiar_id="fam",
                    familiar_display_name="Sapphire",
                    audit_dir=tmp_path,
                    apply=True,
                )
        finally:
            passes_mod.apply_hygiene = original
        # audit artifact still landed
        audits = list(tmp_path.glob("fam-*.json"))
        assert len(audits) == 1

    @pytest.mark.asyncio
    async def test_apply_mutates_and_advances_watermark(self, tmp_path) -> None:  # noqa: ANN001
        raw = _store()
        store = AsyncHistoryStore(raw)
        _, audit_path = await execute_sleep(
            store=store,
            llm=FakeLLMClient(replies=[_reply()]),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            audit_dir=tmp_path,
            apply=True,
        )
        assert "noise" not in {
            f.text for f in raw.recent_facts(familiar_id="fam", limit=10)
        }
        assert raw.get_sleep_watermark(familiar_id="fam") is not None
        assert json.loads(audit_path.read_text())["applied"] is True


def _dream_store() -> HistoryStore:
    store = HistoryStore(":memory:")
    store.append_turn(
        familiar_id="fam",
        channel_id=1,
        role="assistant",
        content="lo-fi is real music, fight me",
        author=None,
    )
    return store


def _dream_replies() -> list[str]:
    return [
        json.dumps({"candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]}),
        json.dumps({
            "opinions": [
                {
                    "text": "Sapphire is fiercely protective of lo-fi as real music.",
                    "source_turn_ids": [1],
                    "reason": "defended it",
                }
            ]
        }),
    ]


class TestExecuteDream:
    @pytest.mark.asyncio
    async def test_dry_run_writes_audit_without_minting(self, tmp_path) -> None:  # noqa: ANN001
        raw = _dream_store()
        store = AsyncHistoryStore(raw)
        plan, audit_path = await execute_dream(
            store=store,
            llm=FakeLLMClient(replies=_dream_replies()),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            display_tz="UTC",
            audit_dir=tmp_path,
            apply=False,
        )
        assert raw.recent_facts(familiar_id="fam", limit=10) == []  # nothing minted
        assert raw.get_sleep_watermark(familiar_id="fam") is None
        audit = json.loads(audit_path.read_text())
        assert audit["kind"] == "dream"
        assert audit["applied"] is False
        # audit renders the grounding excerpt inline
        assert "lo-fi is real music" in audit["opinions"][0]["grounding"][0]["excerpt"]
        assert len(plan.opinions) == 1

    @pytest.mark.asyncio
    async def test_apply_mints_self_facts(self, tmp_path) -> None:  # noqa: ANN001
        raw = _dream_store()
        store = AsyncHistoryStore(raw)
        await execute_dream(
            store=store,
            llm=FakeLLMClient(replies=_dream_replies()),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            display_tz="UTC",
            audit_dir=tmp_path,
            apply=True,
        )
        facts = raw.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert is_self_key(facts[0].subjects[0].canonical_key)
        wm = raw.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        assert wm.last_turn_id == 1
