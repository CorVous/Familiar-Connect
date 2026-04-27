"""Tests for :class:`PeopleDossierLayer`.

Renders cached per-person dossiers into the system prompt for the
people who are speaking in or being mentioned in the active channel.
Read-only over ``people_dossiers`` (the worker maintains writes);
prioritises by recency and caps at ``max_people`` — same hard-count
budgeting style as :class:`RecentHistoryLayer.window_size`.
"""

from __future__ import annotations

import pytest

from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.layers import PeopleDossierLayer
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author


def _ctx(*, channel_id: int = 1) -> AssemblyContext:
    return AssemblyContext(familiar_id="fam", channel_id=channel_id)


def _author(uid: str, *, display: str) -> Author:
    return Author(
        platform="discord",
        user_id=uid,
        username=display.lower(),
        display_name=display,
    )


class TestPeopleDossierLayer:
    @pytest.mark.asyncio
    async def test_empty_when_no_turns(self) -> None:
        store = HistoryStore(":memory:")
        layer = PeopleDossierLayer(store=store)
        assert not await layer.build(_ctx())

    @pytest.mark.asyncio
    async def test_empty_when_dossier_missing(self) -> None:
        """Subjects in the channel without a stored dossier are skipped silently."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author("1", display="Cass"),
        )
        layer = PeopleDossierLayer(store=store)
        assert not await layer.build(_ctx())

    @pytest.mark.asyncio
    async def test_renders_dossier_for_recent_speaker(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author("1", display="Cass"),
        )
        store.upsert_account(_author("1", display="Cass"))
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=5,
            dossier_text="Cass enjoys pho.",
        )
        layer = PeopleDossierLayer(store=store)
        out = await layer.build(_ctx())
        assert "Cass" in out
        assert "pho" in out

    @pytest.mark.asyncio
    async def test_renders_dossier_for_mentioned_user(self) -> None:
        """Mentions matter even when the mentioned user hasn't spoken."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hey, what about <@2>?",
            author=_author("1", display="Cass"),
        )
        store.record_mentions(turn_id=1, canonical_keys=["discord:2"])
        store.upsert_account(_author("2", display="Aria"))
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:2",
            last_fact_id=3,
            dossier_text="Aria runs a bakery.",
        )
        layer = PeopleDossierLayer(store=store)
        out = await layer.build(_ctx())
        assert "Aria" in out
        assert "bakery" in out

    @pytest.mark.asyncio
    async def test_prioritises_most_recent_when_capped(self) -> None:
        """When candidates exceed ``max_people``, oldest mentions are dropped."""
        store = HistoryStore(":memory:")
        # Three speakers in turn order: Cass (oldest), Aria, Bo (newest).
        for uid, display in [("1", "Cass"), ("2", "Aria"), ("3", "Bo")]:
            store.upsert_account(_author(uid, display=display))
            store.put_people_dossier(
                familiar_id="fam",
                canonical_key=f"discord:{uid}",
                last_fact_id=1,
                dossier_text=f"{display} dossier.",
            )
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content="hi",
                author=_author(uid, display=display),
            )
        layer = PeopleDossierLayer(store=store, max_people=2)
        out = await layer.build(_ctx())
        # Most recent two kept (Bo, Aria); oldest (Cass) dropped.
        assert "Bo dossier" in out
        assert "Aria dossier" in out
        assert "Cass dossier" not in out

    @pytest.mark.asyncio
    async def test_dedups_repeated_subjects_keeping_most_recent(self) -> None:
        """Cass speaking, then Aria, then Cass again ⇒ Cass is "newest"."""
        store = HistoryStore(":memory:")
        for uid, display in [("1", "Cass"), ("2", "Aria"), ("1", "Cass")]:
            store.upsert_account(_author(uid, display=display))
            store.put_people_dossier(
                familiar_id="fam",
                canonical_key=f"discord:{uid}",
                last_fact_id=1,
                dossier_text=f"{display} dossier.",
            )
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content="hi",
                author=_author(uid, display=display),
            )
        layer = PeopleDossierLayer(store=store, max_people=2)
        out = await layer.build(_ctx())
        # Both kept; both unique candidates.
        assert "Cass dossier" in out
        assert "Aria dossier" in out
        # Order: Cass first (most recent occurrence), Aria second.
        assert out.index("Cass dossier") < out.index("Aria dossier")

    @pytest.mark.asyncio
    async def test_scoped_to_channel(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=99,  # different channel
            role="user",
            content="hi",
            author=_author("1", display="Cass"),
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="Cass dossier.",
        )
        layer = PeopleDossierLayer(store=store)
        # Channel 1 has no turns ⇒ no candidates ⇒ empty.
        assert not await layer.build(_ctx(channel_id=1))

    def test_invalidation_key_changes_when_new_turn(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author("1", display="Cass"),
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="Cass dossier.",
        )
        layer = PeopleDossierLayer(store=store)
        k1 = layer.invalidation_key(_ctx())
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="more",
            author=_author("1", display="Cass"),
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2

    def test_invalidation_key_changes_when_dossier_watermark_moves(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author("1", display="Cass"),
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="v1",
        )
        layer = PeopleDossierLayer(store=store)
        k1 = layer.invalidation_key(_ctx())
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=10,
            dossier_text="v2",
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2
