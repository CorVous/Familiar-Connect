"""Tests for Phase-3 context layers.

Covers :class:`ConversationSummaryLayer` and :class:`RagContextLayer`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.layers import (
    ConversationSummaryLayer,
    RagContextLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import (
    FOCUS_STREAM_CHANNEL_ID,
    FactSubject,
    HistoryStore,
)
from familiar_connect.identity import Author


def _ctx(*, channel_id: int = 1, viewer_mode: str = "voice") -> AssemblyContext:
    return AssemblyContext(
        familiar_id="fam", channel_id=channel_id, viewer_mode=viewer_mode
    )


def _put_focus_summary(
    store: HistoryStore,
    *,
    summary_text: str,
    last_summarised_id: int = 5,
    last_consumed_at: str | None = "2026-06-13T10:00:00+00:00",
) -> None:
    store.put_summary(
        familiar_id="fam",
        channel_id=FOCUS_STREAM_CHANNEL_ID,
        last_summarised_id=last_summarised_id,
        summary_text=summary_text,
        last_consumed_at=last_consumed_at,
    )


class TestConversationSummaryLayer:
    @pytest.mark.asyncio
    async def test_returns_summary_text(self) -> None:
        store = HistoryStore(":memory:")
        _put_focus_summary(store, summary_text="Earlier they talked about foxes.")
        layer = ConversationSummaryLayer(store=AsyncHistoryStore(store))
        out = await layer.build(_ctx())
        assert "foxes" in out

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_summary(self) -> None:
        store = HistoryStore(":memory:")
        layer = ConversationSummaryLayer(store=AsyncHistoryStore(store))
        assert not await layer.build(_ctx())

    @pytest.mark.asyncio
    async def test_build_ignores_ctx_channel_id(self) -> None:
        store = HistoryStore(":memory:")
        _put_focus_summary(store, summary_text="the one true thread")
        layer = ConversationSummaryLayer(store=AsyncHistoryStore(store))
        assert "thread" in await layer.build(_ctx(channel_id=1))
        assert "thread" in await layer.build(_ctx(channel_id=999))

    @pytest.mark.asyncio
    async def test_loads_same_summary_in_text_and_voice(self) -> None:
        store = HistoryStore(":memory:")
        _put_focus_summary(store, summary_text="modality-agnostic thread")
        layer = ConversationSummaryLayer(store=AsyncHistoryStore(store))
        text_out = await layer.build(_ctx(viewer_mode="text"))
        voice_out = await layer.build(_ctx(viewer_mode="voice"))
        assert text_out == voice_out
        assert "modality-agnostic" in text_out

    def test_invalidation_key_tracks_composite_watermark(self) -> None:
        store = HistoryStore(":memory:")
        layer = ConversationSummaryLayer(store=AsyncHistoryStore(store))
        assert layer.invalidation_key(_ctx()) == "none"
        _put_focus_summary(
            store,
            summary_text="v1",
            last_summarised_id=5,
            last_consumed_at="2026-06-13T10:00:00+00:00",
        )
        k1 = layer.invalidation_key(_ctx())
        _put_focus_summary(
            store,
            summary_text="v2",
            last_summarised_id=10,
            last_consumed_at="2026-06-13T11:00:00+00:00",
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2
        assert k1 != "none"


class TestRagContextLayer:
    @pytest.mark.asyncio
    async def test_empty_when_no_cues(self) -> None:
        store = HistoryStore(":memory:")
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_results=5)
        # No cues ⇒ empty contribution.
        assert not await layer.build(_ctx())

    @pytest.mark.asyncio
    async def test_returns_matches_for_current_cue(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="Let's discuss the fox plan tomorrow at noon.",
            author=None,
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="Sure thing.",
            author=None,
        )
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_results=5)
        layer.set_current_cue("fox")
        out = await layer.build(_ctx(channel_id=1))
        assert "fox plan" in out

    @pytest.mark.asyncio
    async def test_scoped_to_familiar(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="other",
            channel_id=1,
            role="user",
            content="Fox on a different familiar.",
            author=None,
        )
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_results=5)
        layer.set_current_cue("fox")
        out = await layer.build(_ctx())
        assert not out

    def test_invalidation_key_reflects_cue_and_watermark(self) -> None:
        store = HistoryStore(":memory:")
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_results=5)
        layer.set_current_cue("fox")
        k1 = layer.invalidation_key(_ctx())
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="new turn",
            author=None,
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2
        layer.set_current_cue("otter")
        k3 = layer.invalidation_key(_ctx())
        assert k3 not in {k1, k2}

    @pytest.mark.asyncio
    async def test_excludes_turns_within_recent_window(self) -> None:
        """RAG must not re-surface turns already in ``RecentHistoryLayer``.

        Otherwise the user's own most-recent message (which is in
        ``fts_turns`` and matches its own cue perfectly) shows up
        twice in the prompt — once verbatim, once as "possibly
        relevant earlier turns".
        """
        store = HistoryStore(":memory:")
        # 30 turns total; the ones inside the recent-window (the
        # last 20) must NOT appear in RAG output.
        for i in range(30):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"strawberry observation number {i}",
                author=None,
            )
        layer = RagContextLayer(
            store=AsyncHistoryStore(store), max_results=10, recent_window_size=20
        )
        layer.set_current_cue("strawberry")
        out = await layer.build(_ctx(channel_id=1))
        # Every numeric mention should be from id 1..10 (older than
        # the 20-turn window's id 11..30).
        for older_idx in range(10):
            assert f"number {older_idx}" in out, (older_idx, out)
        for recent_idx in range(10, 30):
            assert f"number {recent_idx}" not in out, (recent_idx, out)

    @pytest.mark.asyncio
    async def test_fact_with_renamed_subject_gets_annotation(self) -> None:
        """When a fact's subject has since renamed, annotate the rendered fact.

        Render preserves the original fact text (what was actually
        said) and appends a soft hint linking the stale display name
        to the current one. The link is advisory — a real-world mic
        share or relay can break the canonical-key correspondence —
        but it's enough to keep the model from treating "Cass" and
        "peeks" as different people.
        """
        store = HistoryStore(":memory:")
        # User started as "Cass"; an earlier turn from before rename.
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hello from cass",
            author=Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Cass",
            ),
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(
                FactSubject(canonical_key="discord:111", display_at_write="Cass"),
            ),
        )
        # The same user has since renamed to "peeks".
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="still here under a new name",
            author=Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="peeks",
            ),
        )
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_facts=3)
        layer.set_current_cue("pho")
        out = await layer.build(_ctx(channel_id=1))

        assert "Cass likes pho." in out  # original text preserved
        assert "peeks" in out  # current display name surfaced
        assert "Cass" in out  # old name still visible (it's in the text)

    @pytest.mark.asyncio
    async def test_fact_without_rename_renders_without_annotation(self) -> None:
        """If display name hasn't changed, no annotation is added."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hello from cass",
            author=Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Cass",
            ),
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(
                FactSubject(canonical_key="discord:111", display_at_write="Cass"),
            ),
        )
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_facts=3)
        layer.set_current_cue("pho")
        out = await layer.build(_ctx(channel_id=1))

        assert "Cass likes pho." in out
        # No "now known as" / "formerly" / "→" annotation when names match.
        assert "now known as" not in out.lower()
        assert "formerly" not in out.lower()

    @pytest.mark.asyncio
    async def test_legacy_fact_without_subjects_renders_unchanged(self) -> None:
        """Facts written before the subjects feature have no metadata.

        Forward-only fix — they render as plain text, no annotation.
        """
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="An old fact about pho.",
            source_turn_ids=[],
        )
        layer = RagContextLayer(store=AsyncHistoryStore(store), max_facts=3)
        layer.set_current_cue("pho")
        out = await layer.build(_ctx(channel_id=1))

        assert "An old fact about pho." in out
        assert "now known as" not in out.lower()

    @pytest.mark.asyncio
    async def test_renders_with_date_header_and_12h_clock(self) -> None:
        """Earlier turns group under ``YYYY-MM-DD:`` with ``H:MMpm`` lines."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="my brain's dying",
            author=Author(
                platform="discord",
                user_id="111",
                username="peebo",
                display_name="Peebo",
            ),
        )
        # pin the timestamp so the assertion is stable
        ts = datetime(2026, 5, 3, 14, 29, tzinfo=UTC).isoformat()
        store._conn.execute("UPDATE turns SET timestamp = ? WHERE id = 1", (ts,))
        store._conn.commit()
        layer = RagContextLayer(
            store=AsyncHistoryStore(store), max_results=5, context_window=0
        )
        layer.set_current_cue("brain")
        out = await layer.build(_ctx(channel_id=1))
        assert "## Possibly relevant earlier turns" in out
        assert "2026-05-03:" in out
        assert "> [2:29PM Peebo]: my brain's dying" in out

    @pytest.mark.asyncio
    async def test_renders_in_configured_display_tz(self) -> None:
        """Date header + clock localize to display_tz (here crossing midnight)."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="my brain's dying",
            author=Author(
                platform="discord",
                user_id="111",
                username="peebo",
                display_name="Peebo",
            ),
        )
        # 02:29 UTC on May 3 -> May 2 19:29 PDT (date *and* clock shift)
        ts = datetime(2026, 5, 3, 2, 29, tzinfo=UTC).isoformat()
        store._conn.execute("UPDATE turns SET timestamp = ? WHERE id = 1", (ts,))
        store._conn.commit()
        layer = RagContextLayer(
            store=AsyncHistoryStore(store),
            max_results=5,
            context_window=0,
            display_tz="America/Los_Angeles",
        )
        layer.set_current_cue("brain")
        out = await layer.build(_ctx(channel_id=1))
        assert "2026-05-02:" in out
        assert "> [7:29PM Peebo]: my brain's dying" in out

    @pytest.mark.asyncio
    async def test_multiline_message_prefixes_every_line(self) -> None:
        """Multi-line content keeps the blockquote intact on every line."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content=("*small surprised blink*\n\nzero?\n\n*looks at cassidy*"),
            author=Author(
                platform="discord",
                user_id="222",
                username="assistant",
                display_name="assistant",
            ),
        )
        ts = datetime(2026, 5, 2, 3, 2, tzinfo=UTC).isoformat()
        store._conn.execute("UPDATE turns SET timestamp = ? WHERE id = 1", (ts,))
        store._conn.commit()
        layer = RagContextLayer(
            store=AsyncHistoryStore(store), max_results=5, context_window=0
        )
        layer.set_current_cue("blink")
        out = await layer.build(_ctx(channel_id=1))
        assert "> [3:02AM assistant]: *small surprised blink*" in out
        # every continuation line must keep the `>` prefix; blank lines
        # render as bare `>` so the blockquote isn't broken.
        assert "> zero?" in out
        assert "> *looks at cassidy*" in out
        # no continuation line should appear without the `>` marker
        assert "\nzero?" not in out
        assert "\n*looks at cassidy*" not in out

    @pytest.mark.asyncio
    async def test_includes_neighbour_context_per_hit(self) -> None:
        """Each hit pulls ±context_window neighbours, dedup'd, in order."""
        store = HistoryStore(":memory:")
        for content in [
            "warmup chatter A",
            "marker turn that mentions strawberry",
            "follow up reply B",
            "unrelated middle",
            "later mention of strawberry too",
        ]:
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=content,
                author=None,
            )
        layer = RagContextLayer(
            store=AsyncHistoryStore(store), max_results=5, context_window=1
        )
        layer.set_current_cue("strawberry")
        out = await layer.build(_ctx(channel_id=1))
        # hit + immediate neighbours rendered; unrelated middle still
        # appears because it's a neighbour of the second hit too.
        assert "warmup chatter A" in out
        assert "marker turn that mentions strawberry" in out
        assert "follow up reply B" in out
        assert "later mention of strawberry too" in out

    @pytest.mark.asyncio
    async def test_zero_recent_window_keeps_old_behavior(self) -> None:
        """Default ``recent_window_size=0`` disables exclusion.

        Existing tests and any caller that doesn't opt into the
        window-aware mode get the unfiltered RAG semantics.
        """
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="strawberry observation",
            author=None,
        )
        layer = RagContextLayer(
            store=AsyncHistoryStore(store), max_results=5
        )  # default = 0
        layer.set_current_cue("strawberry")
        out = await layer.build(_ctx(channel_id=1))
        assert "strawberry observation" in out


class TestRecentHistoryToolTurns:
    """Past tool turns must replay as plain text, never orphaned ``tool`` msgs.

    Anthropic rejects a ``role=tool`` result with no preceding matching
    ``tool_use`` (HTTP 500). Recent-history replay drops tool-call
    linkage, so any ``tool`` turn in the window would orphan. Render
    such turns as narration text instead.
    """

    @pytest.mark.asyncio
    async def test_tool_turn_rendered_as_text_not_tool_role(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="look at this",
            author=Author(
                platform="discord",
                user_id="111",
                username="cor",
                display_name="Cor",
            ),
        )
        # assistant turn that invoked view_image (empty prose, tool call)
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="",
            author=None,
            tool_calls_json=(
                '[{"id": "call_1", "type": "function", '
                '"function": {"name": "view_image", '
                '"arguments": "{\\"image_id\\":\\"img_0\\"}"}}]'
            ),
        )
        # tool result answering that call — the orphan-prone turn
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="tool",
            content="# Image Description\n\nA red fox spirit.",
            author=None,
            tool_call_id="call_1",
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))

        # no replayed message may carry the bare ``tool`` role — that's
        # the orphan Anthropic 500s on
        assert all(m.role != "tool" for m in msgs)
        # description content survives as context somewhere
        assert any("A red fox spirit." in (m.content or "") for m in msgs)
        # no message carries an unpaired tool_call_id either
        assert all(getattr(m, "tool_call_id", None) is None for m in msgs)

    @pytest.mark.asyncio
    async def test_tool_result_replays_as_user_not_assistant(self) -> None:
        # mimicry guard: replaying tool results as ``assistant`` teaches
        # the model to open fresh replies with ``[tool result] …`` (and
        # even fabricate results). render as non-assistant narration so
        # the model never sees it as its own reply pattern.
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="tool",
            content="# Image Description\n\nA red fox spirit.",
            author=None,
            tool_call_id="call_1",
        )
        layer = RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))

        leaked = [m for m in msgs if "A red fox spirit." in (m.content or "")]
        assert leaked, "tool-result content must survive in context"
        assert all(m.role != "assistant" for m in leaked)
        assert all(m.role == "user" for m in leaked)
