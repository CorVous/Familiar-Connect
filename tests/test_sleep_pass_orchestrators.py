"""Sleep pass orchestrators — plan → apply, dry-run vs apply.

``execute_consolidation`` / ``execute_opinion_formation`` wrap
plan+apply for both the activity-engine lifecycle path and any ad-hoc
caller. Tests pin the dry-run-never-mutates contract and the
rejected-proposal log signal (rail violations the code blocked, routed
to a WARNING).
"""

from __future__ import annotations

import json
import logging

import pytest

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import is_ego_key
from familiar_connect.sleep.maintenance import (
    execute_consolidation,
    execute_opinion_formation,
)
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


class TestExecuteConsolidation:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_mutate(self) -> None:
        raw = _store()
        store = AsyncHistoryStore(raw)
        plan = await execute_consolidation(
            store=store,
            llm=FakeLLMClient(replies=[_reply()]),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            apply=False,
        )
        # live facts untouched
        assert {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)} == {
            "noise",
            "Aria likes tea.",
        }
        # no watermark written on dry-run
        assert raw.get_sleep_watermark(familiar_id="fam") is None
        assert len(plan.retire) == 1

    @pytest.mark.asyncio
    async def test_apply_mutates_and_advances_watermark(self) -> None:
        raw = _store()
        store = AsyncHistoryStore(raw)
        await execute_consolidation(
            store=store,
            llm=FakeLLMClient(replies=[_reply()]),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            apply=True,
        )
        assert "noise" not in {
            f.text for f in raw.recent_facts(familiar_id="fam", limit=10)
        }
        assert raw.get_sleep_watermark(familiar_id="fam") is not None

    @pytest.mark.asyncio
    async def test_rail_rejection_is_logged_with_rail_name(self, caplog) -> None:  # noqa: ANN001
        """A rail-blocked proposal lands as a WARNING naming the rail.

        The audit JSON is gone; this log line is the preserved signal of
        what the LLM proposed that the code rails refused.
        """
        raw = _store()
        store = AsyncHistoryStore(raw)
        # fact id 999 does not exist → the unknown_id rail blocks the retire
        reply = json.dumps({
            "retire": [{"fact_ids": [999], "reason": "phantom"}],
            "rewrite": [],
        })
        with caplog.at_level(logging.WARNING):
            await execute_consolidation(
                store=store,
                llm=FakeLLMClient(replies=[reply]),
                familiar_id="fam",
                familiar_display_name="Sapphire",
                apply=False,
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("unknown_id" in r.getMessage() for r in warnings)


def _opinion_store() -> HistoryStore:
    store = HistoryStore(":memory:")
    store.append_turn(
        familiar_id="fam",
        channel_id=1,
        role="assistant",
        content="lo-fi is real music, fight me",
        author=None,
    )
    return store


def _opinion_replies() -> list[str]:
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


class TestExecuteOpinionFormation:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_mint(self) -> None:
        raw = _opinion_store()
        store = AsyncHistoryStore(raw)
        plan = await execute_opinion_formation(
            store=store,
            llm=FakeLLMClient(replies=_opinion_replies()),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            display_tz="UTC",
            apply=False,
        )
        assert raw.recent_facts(familiar_id="fam", limit=10) == []  # nothing minted
        assert raw.get_sleep_watermark(familiar_id="fam") is None
        assert len(plan.opinions) == 1

    @pytest.mark.asyncio
    async def test_apply_mints_self_facts(self) -> None:
        raw = _opinion_store()
        store = AsyncHistoryStore(raw)
        await execute_opinion_formation(
            store=store,
            llm=FakeLLMClient(replies=_opinion_replies()),
            familiar_id="fam",
            familiar_display_name="Sapphire",
            display_tz="UTC",
            apply=True,
        )
        facts = raw.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert is_ego_key(facts[0].subjects[0].canonical_key)
        wm = raw.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        assert wm.last_turn_id == 1
