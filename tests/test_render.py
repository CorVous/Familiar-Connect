"""Red-first tests for assemble_chat_messages.

The renderer turns a :class:`PipelineOutput` plus the current
:class:`HistoryStore` contents into a list of
:class:`familiar_connect.llm.Message` ready to hand to the LLM
client. Its job is the one thing the pipeline can't express on its
own: SillyTavern-accurate ``Layer.depth_inject`` placement between
messages, not in the system prompt.

Covers familiar_connect.context.render, which doesn't exist yet.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from familiar_connect.config import ChannelMode
from familiar_connect.context.budget import BudgetResult
from familiar_connect.context.pipeline import PipelineOutput
from familiar_connect.context.render import assemble_chat_messages
from familiar_connect.context.types import (
    ContextRequest,
    Layer,
    Modality,
)
from familiar_connect.history.store import HistoryStore

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CHANNEL = 100
_FAMILIAR = "aria"


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": _FAMILIAR,
        "channel_id": _CHANNEL,
        "guild_id": 1,
        "speaker": "Alice",
        "utterance": "latest thing Alice said",
        "modality": Modality.text,
        "budget_tokens": 2048,
        "deadline_s": 5.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)  # type: ignore[arg-type]


def _store_with(
    tmp_path: Path,
    contents: list[tuple[str, str, str | None]],
) -> HistoryStore:
    """Populate a fresh store with ``(role, content, speaker)`` turns."""
    store = HistoryStore(tmp_path / "history.db")
    for role, content, speaker in contents:
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role=role,
            content=content,
            speaker=speaker,
        )
    return store


def _pipeline_output(
    *,
    request: ContextRequest,
    by_layer: dict[Layer, str] | None = None,
) -> PipelineOutput:
    return PipelineOutput(
        request=request,
        budget=BudgetResult(by_layer=by_layer or {}, dropped=[]),
        outcomes=[],
    )


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_joins_non_history_non_depth_layers(
        self,
        tmp_path: Path,
    ) -> None:
        request = _request()
        store = _store_with(tmp_path, [])
        output = _pipeline_output(
            request=request,
            by_layer={
                Layer.core: "core rails",
                Layer.character: "persona block",
                Layer.content: "relevant snippet",
                Layer.recent_history: "should be ignored at render time",
                Layer.depth_inject: "depth text",
            },
        )

        messages = assemble_chat_messages(output, store=store)

        # The first message is the system prompt; it contains the
        # system-prompt layers but *not* the recent_history contribution
        # text (the renderer re-fetches history turns from the store).
        system = messages[0]
        assert system.role == "system"
        assert "core rails" in system.content
        assert "persona block" in system.content
        assert "relevant snippet" in system.content
        assert "should be ignored" not in system.content
        # depth_inject doesn't live in the system prompt — it's inserted mid-chat.
        assert "depth text" not in system.content

    def test_system_prompt_explains_speaker_prefix_convention(
        self,
        tmp_path: Path,
    ) -> None:
        """Fixed preamble tells the LLM what the ``Speaker:`` prefix means.

        Models that don't honour the OpenAI ``name`` field (most
        non-OpenAI backends through OpenRouter) still need to know
        what the convention means. The preamble is part of the
        rendering contract, not a per-character or per-mode knob.
        """
        store = _store_with(tmp_path, [])
        output = _pipeline_output(request=_request(), by_layer={})

        messages = assemble_chat_messages(output, store=store)

        assert messages[0].role == "system"
        # The preamble must mention the speaker-prefix convention in
        # a way the model can follow without further context.
        lowered = messages[0].content.lower()
        assert "prefix" in lowered or "prefixed" in lowered
        assert "speaker" in lowered or "name" in lowered

    def test_preamble_precedes_layer_content(self, tmp_path: Path) -> None:
        """Preamble sits above layer sections at the top of the system prompt.

        Ensures it's the first thing the model reads when the
        system prompt is assembled.
        """
        store = _store_with(tmp_path, [])
        output = _pipeline_output(
            request=_request(),
            by_layer={Layer.core: "CORE SENTINEL"},
        )

        messages = assemble_chat_messages(output, store=store)

        system = messages[0].content
        assert "CORE SENTINEL" in system
        # Preamble word appears before the core sentinel.
        convention_idx = max(
            system.lower().find("prefix"),
            system.lower().find("prefixed"),
        )
        core_idx = system.find("CORE SENTINEL")
        assert 0 <= convention_idx < core_idx


# ---------------------------------------------------------------------------
# Speaker prefixing on user turns
# ---------------------------------------------------------------------------


class TestSpeakerPrefix:
    def test_user_turn_content_is_prefixed_with_speaker(
        self,
        tmp_path: Path,
    ) -> None:
        """User turns from history get ``Speaker: `` prepended to their content.

        Models that don't honour the OpenAI ``name`` field still
        see who's talking via the content prefix.
        """
        store = _store_with(
            tmp_path,
            [
                ("user", "hi", "Alice"),
                ("assistant", "hello back", None),
                ("user", "how are you?", "Bob"),
            ],
        )
        output = _pipeline_output(
            request=_request(),
            by_layer={Layer.core: "core"},
        )

        messages = assemble_chat_messages(output, store=store)

        # system + 3 history + final user
        assert messages[1].role == "user"
        assert messages[1].content == "Alice: hi"
        # Assistant turns are NOT prefixed — role=assistant is enough
        # to distinguish them, and prefixing them would invite the
        # model to emit the prefix itself in its reply.
        assert messages[2].role == "assistant"
        assert messages[2].content == "hello back"
        assert messages[3].role == "user"
        assert messages[3].content == "Bob: how are you?"

    def test_final_user_turn_is_prefixed_with_request_speaker(
        self,
        tmp_path: Path,
    ) -> None:
        store = _store_with(tmp_path, [])
        request = _request(speaker="Alice", utterance="latest thing Alice said")
        output = _pipeline_output(
            request=request,
            by_layer={Layer.core: "core"},
        )

        messages = assemble_chat_messages(output, store=store)

        assert messages[-1].role == "user"
        assert messages[-1].content == "Alice: latest thing Alice said"

    def test_name_field_still_populated_for_openai_compatible_models(
        self,
        tmp_path: Path,
    ) -> None:
        """The ``Message.name`` field is still populated alongside the prefix.

        OpenAI's own models (which do honour ``name``) get both
        signals. Redundant is fine; silent drop is not.
        """
        store = _store_with(
            tmp_path,
            [("user", "hi", "Alice")],
        )
        output = _pipeline_output(
            request=_request(speaker="Alice"),
            by_layer={Layer.core: "core"},
        )

        messages = assemble_chat_messages(output, store=store)

        assert messages[1].name == "Alice"
        assert messages[-1].name == "Alice"

    def test_user_turn_with_no_speaker_is_not_prefixed(
        self,
        tmp_path: Path,
    ) -> None:
        """A user turn whose speaker sanitised to ``None`` is rendered bare.

        Twitch events and usernames that sanitise to empty still
        need a working ``role="user"`` message — no literal
        ``None: content`` gibberish.
        """
        store = _store_with(
            tmp_path,
            [("user", "from the void", None)],
        )
        output = _pipeline_output(
            request=_request(speaker=None, utterance="latest"),
            by_layer={Layer.core: "core"},
        )

        messages = assemble_chat_messages(output, store=store)

        assert messages[1].content == "from the void"
        assert messages[1].name is None
        assert messages[-1].content == "latest"
        assert messages[-1].name is None


# ---------------------------------------------------------------------------
# Recent history rendering
# ---------------------------------------------------------------------------


class TestRecentHistory:
    def test_history_is_read_from_store_not_contribution(
        self,
        tmp_path: Path,
    ) -> None:
        store = _store_with(
            tmp_path,
            [
                ("user", "hi", "Alice"),
                ("assistant", "hello back", None),
                ("user", "how are you?", "Alice"),
            ],
        )
        request = _request()
        # The contribution has the wrong text on purpose — renderer must ignore it.
        output = _pipeline_output(
            request=request,
            by_layer={
                Layer.core: "core",
                Layer.recent_history: "WRONG TEXT FROM CONTRIBUTION",
            },
        )

        messages = assemble_chat_messages(output, store=store)

        # system + 3 history turns + 1 final user turn (from request.utterance).
        assert len(messages) == 5
        assert messages[0].role == "system"
        # User turns are prefixed with "Speaker: " so non-OpenAI backends
        # (which drop the OpenAI ``name`` field at the OpenRouter boundary)
        # can still see who's talking. See TestSpeakerPrefix for the
        # dedicated coverage.
        assert messages[1].role == "user"
        assert messages[1].content == "Alice: hi"
        assert messages[2].role == "assistant"
        assert messages[2].content == "hello back"
        assert messages[3].role == "user"
        assert messages[3].content == "Alice: how are you?"
        assert messages[-1].role == "user"
        assert messages[-1].content == "Alice: latest thing Alice said"


# ---------------------------------------------------------------------------
# Depth-inject placement (the SillyTavern-accurate piece)
# ---------------------------------------------------------------------------


class TestDepthInject:
    def test_depth_zero_inserts_before_final_user_turn(
        self,
        tmp_path: Path,
    ) -> None:
        store = _store_with(
            tmp_path,
            [
                ("user", "one", "Alice"),
                ("assistant", "two", None),
            ],
        )
        request = _request()
        output = _pipeline_output(
            request=request,
            by_layer={
                Layer.core: "core",
                Layer.depth_inject: "INJECT",
            },
        )

        messages = assemble_chat_messages(
            output,
            store=store,
            depth_inject_position=0,
        )

        # system, "Alice: one", "two", INJECT, final user (prefixed).
        assert messages[-2].role == "system"
        assert messages[-2].content == "INJECT"
        assert messages[-1].content == "Alice: latest thing Alice said"

    def test_depth_two_inserts_two_turns_before_the_end(
        self,
        tmp_path: Path,
    ) -> None:
        store = _store_with(
            tmp_path,
            [
                ("user", "one", "Alice"),
                ("assistant", "two", None),
                ("user", "three", "Alice"),
                ("assistant", "four", None),
            ],
        )
        request = _request(utterance="five")
        output = _pipeline_output(
            request=request,
            by_layer={
                Layer.core: "core",
                Layer.depth_inject: "INJECT",
            },
        )

        messages = assemble_chat_messages(
            output,
            store=store,
            depth_inject_position=2,
        )

        # system, Alice: one, two (assistant), Alice: three, INJECT,
        # four (assistant), Alice: five.
        assert "core" in messages[0].content  # system prompt; preamble + core
        contents = [m.content for m in messages[1:]]
        assert contents == [
            "Alice: one",
            "two",
            "Alice: three",
            "INJECT",
            "four",
            "Alice: five",
        ]

    def test_depth_position_beyond_history_caps_at_front(
        self,
        tmp_path: Path,
    ) -> None:
        """A depth larger than the chat buffer clamps to the top of history."""
        store = _store_with(
            tmp_path,
            [
                ("user", "one", "Alice"),
            ],
        )
        request = _request(utterance="two")
        output = _pipeline_output(
            request=request,
            by_layer={
                Layer.core: "core",
                Layer.depth_inject: "INJECT",
            },
        )

        messages = assemble_chat_messages(
            output,
            store=store,
            depth_inject_position=99,
        )

        # Clamped: system, INJECT, Alice: one, Alice: two
        assert messages[0].role == "system"
        assert messages[1].content == "INJECT"
        assert messages[2].content == "Alice: one"
        assert messages[3].content == "Alice: two"

    def test_depth_inject_absent_when_layer_empty(self, tmp_path: Path) -> None:
        store = _store_with(tmp_path, [])
        request = _request()
        output = _pipeline_output(
            request=request,
            by_layer={Layer.core: "core"},
        )

        messages = assemble_chat_messages(output, store=store)

        # system + final user only
        assert [m.role for m in messages] == ["system", "user"]
        assert "INJECT" not in "".join(m.content for m in messages)

    def test_depth_inject_role_can_be_user(self, tmp_path: Path) -> None:
        store = _store_with(tmp_path, [])
        request = _request()
        output = _pipeline_output(
            request=request,
            by_layer={
                Layer.core: "core",
                Layer.depth_inject: "INJECT",
            },
        )

        messages = assemble_chat_messages(
            output,
            store=store,
            depth_inject_role="user",
        )

        # system, INJECT (as user), final user
        assert messages[1].role == "user"
        assert messages[1].content == "INJECT"


# ---------------------------------------------------------------------------
# History window size
# ---------------------------------------------------------------------------


class TestHistoryWindowSize:
    def test_history_limit_caps_number_of_rendered_turns(
        self,
        tmp_path: Path,
    ) -> None:
        store = _store_with(
            tmp_path,
            [
                ("user", "one", "Alice"),
                ("assistant", "two", None),
                ("user", "three", "Alice"),
                ("assistant", "four", None),
                ("user", "five", "Alice"),
            ],
        )
        request = _request(utterance="six")
        output = _pipeline_output(
            request=request,
            by_layer={Layer.core: "core"},
        )

        messages = assemble_chat_messages(
            output,
            store=store,
            history_window_size=2,
        )

        # system, last 2 ("four" assistant + "Alice: five"), final user
        # ("Alice: six"). The system message carries the preamble plus
        # Layer.core content.
        assert "core" in messages[0].content
        contents = [m.content for m in messages[1:]]
        assert contents == ["four", "Alice: five", "Alice: six"]


# ---------------------------------------------------------------------------
# Mode-filtered history
# ---------------------------------------------------------------------------


class TestModeFilteredHistory:
    def test_only_matching_mode_turns_appear_in_messages(
        self,
        tmp_path: Path,
    ) -> None:
        """When mode is passed, only turns tagged with that mode surface."""
        store = HistoryStore(tmp_path / "history.db")
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="rp action",
            speaker="Alice",
            mode=ChannelMode.full_rp,
        )
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="voice line",
            speaker="Alice",
            mode=ChannelMode.imitate_voice,
        )
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="another rp",
            speaker="Alice",
            mode=ChannelMode.full_rp,
        )

        request = _request(utterance="continue the scene")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.full_rp,
        )

        contents = [m.content for m in messages[1:]]  # skip system prompt
        # Should see only the two full_rp turns + the final user turn.
        assert any("rp action" in c for c in contents)
        assert any("another rp" in c for c in contents)
        assert not any("voice line" in c for c in contents)


# ---------------------------------------------------------------------------
# Per-mode timestamp formatting
# ---------------------------------------------------------------------------


class TestTextConversationTimestamps:
    def test_user_turns_get_hhmm_prefix(self, tmp_path: Path) -> None:
        """In text_conversation_rp, user turns are prefixed [HH:MM]."""
        store = HistoryStore(tmp_path / "history.db")
        # Insert a turn with a known timestamp.
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="hello there",
            speaker="Alice",
            mode=ChannelMode.text_conversation_rp,
        )

        request = _request(utterance="reply")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.text_conversation_rp,
            display_tz="UTC",
        )

        # Find the history user turn (not the final utterance).
        history_msgs = [m for m in messages[1:-1] if m.role == "user"]
        assert len(history_msgs) == 1
        # Should start with [HH:MM] pattern.
        assert re.match(
            r"^\[\d{2}:\d{2}\] Alice: hello there$",
            history_msgs[0].content,
        )

    def test_assistant_turns_unchanged(self, tmp_path: Path) -> None:
        """Assistant turns should NOT get timestamps in any mode."""
        store = HistoryStore(tmp_path / "history.db")
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="assistant",
            content="hey",
            mode=ChannelMode.text_conversation_rp,
        )

        request = _request(utterance="reply")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.text_conversation_rp,
            display_tz="UTC",
        )

        assistant_msgs = [m for m in messages[1:-1] if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "hey"


class TestImitateVoiceGapHints:
    def test_gap_over_30s_adds_breadcrumb(self, tmp_path: Path) -> None:
        """A gap >= 30s between turns produces a time-gap prefix."""
        store = HistoryStore(tmp_path / "history.db")
        # We need control over timestamps, so insert via raw SQL.
        t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        t2 = t1 + timedelta(minutes=3)
        store._conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
            " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
            (
                _FAMILIAR,
                _CHANNEL,
                "user",
                "Alice",
                "first",
                t1.isoformat(),
                "imitate_voice",
            ),
        )
        store._conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
            " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
            (
                _FAMILIAR,
                _CHANNEL,
                "user",
                "Alice",
                "second",
                t2.isoformat(),
                "imitate_voice",
            ),
        )
        store._conn.commit()

        request = _request(utterance="third")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.imitate_voice,
        )

        # The second history turn should have a gap prefix.
        history = messages[1:-1]
        assert len(history) == 2
        assert "(about 3 minutes later)" in history[1].content

    def test_gap_under_30s_no_breadcrumb(self, tmp_path: Path) -> None:
        """Gaps under 30s produce no breadcrumb."""
        store = HistoryStore(tmp_path / "history.db")
        t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        t2 = t1 + timedelta(seconds=10)
        store._conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
            " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
            (
                _FAMILIAR,
                _CHANNEL,
                "user",
                "Alice",
                "first",
                t1.isoformat(),
                "imitate_voice",
            ),
        )
        store._conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
            " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
            (
                _FAMILIAR,
                _CHANNEL,
                "user",
                "Alice",
                "second",
                t2.isoformat(),
                "imitate_voice",
            ),
        )
        store._conn.commit()

        request = _request(utterance="third")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.imitate_voice,
        )

        history = messages[1:-1]
        assert len(history) == 2
        # No gap breadcrumb on the second turn.
        assert "later)" not in history[1].content

    def test_first_turn_has_no_breadcrumb(self, tmp_path: Path) -> None:
        """The very first turn never gets a gap prefix."""
        store = HistoryStore(tmp_path / "history.db")
        t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        store._conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
            " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
            (
                _FAMILIAR,
                _CHANNEL,
                "user",
                "Alice",
                "only one",
                t1.isoformat(),
                "imitate_voice",
            ),
        )
        store._conn.commit()

        request = _request(utterance="reply")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.imitate_voice,
        )

        history = messages[1:-1]
        assert len(history) == 1
        assert "later)" not in history[0].content


class TestFullRpNoTimestamps:
    def test_user_turns_unchanged(self, tmp_path: Path) -> None:
        """full_rp turns get no timestamps, no gap hints."""
        store = HistoryStore(tmp_path / "history.db")
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="she walks in",
            speaker="Alice",
            mode=ChannelMode.full_rp,
        )

        request = _request(utterance="continue")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(
            output,
            store=store,
            mode=ChannelMode.full_rp,
        )

        history_msgs = [m for m in messages[1:-1] if m.role == "user"]
        assert len(history_msgs) == 1
        # No timestamp prefix — just the standard speaker prefix.
        assert history_msgs[0].content == "Alice: she walks in"


# ---------------------------------------------------------------------------
# full_rp cross-context gap breadcrumbs
# ---------------------------------------------------------------------------


class TestFullRpGapBreadcrumbs:
    """full_rp inserts system breadcrumbs between gapped turns.

    A role=system breadcrumb appears when a gap >= 5 minutes exists
    AND other channels had activity during that gap.
    """

    def _seed_gap_scene(
        self,
        store: HistoryStore,
        *,
        gap: timedelta,
    ) -> None:
        """Insert two full_rp turns in the main channel with a controlled gap."""
        t1 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        t2 = t1 + gap
        for ts, content in [(t1, "scene start"), (t2, "scene resume")]:
            store._conn.execute(
                "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
                " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
                (
                    _FAMILIAR,
                    _CHANNEL,
                    "user",
                    "Alice",
                    content,
                    ts.isoformat(),
                    "full_rp",
                ),
            )
        store._conn.commit()

    def _seed_other_channel_activity(
        self,
        store: HistoryStore,
        *,
        activity_time: datetime,
        channel_id: int = 200,
    ) -> None:
        """Insert a turn in another channel and cache a cross-context summary."""
        store._conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, speaker,"
            " content, timestamp, mode) VALUES (?,?,?,?,?,?,?)",
            (
                _FAMILIAR,
                channel_id,
                "user",
                "Bob",
                "hey in text chat",
                activity_time.isoformat(),
                "text_conversation_rp",
            ),
        )
        store._conn.commit()
        # Cache a cross-context summary so the renderer can find it.
        store.put_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=channel_id,
            source_last_id=store.latest_id(familiar_id=_FAMILIAR, channel_id=channel_id)
            or 0,
            summary_text="Alice and Bob discussed plans in the text chat",
        )

    def test_breadcrumb_inserted_when_gap_and_cross_activity(
        self, tmp_path: Path
    ) -> None:
        """A 10-minute gap with cross-channel activity during it gets a breadcrumb."""
        store = HistoryStore(tmp_path / "history.db")
        gap = timedelta(minutes=10)
        self._seed_gap_scene(store, gap=gap)

        # Other channel activity happened during the gap.
        t1 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        activity_time = t1 + timedelta(minutes=5)  # midway through gap
        self._seed_other_channel_activity(store, activity_time=activity_time)

        request = _request(utterance="continue the scene")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(output, store=store, mode=ChannelMode.full_rp)

        # Expect: system, "scene start", BREADCRUMB, "scene resume", final user
        roles = [m.role for m in messages]
        assert roles.count("system") >= 2  # system prompt + breadcrumb
        # Find the breadcrumb — a system message between history turns.
        breadcrumbs = [m for m in messages[1:-1] if m.role == "system"]
        assert len(breadcrumbs) == 1
        assert "Alice and Bob discussed plans" in breadcrumbs[0].content

    def test_no_breadcrumb_when_no_cross_activity(self, tmp_path: Path) -> None:
        """A gap with no other-channel activity produces no breadcrumb."""
        store = HistoryStore(tmp_path / "history.db")
        self._seed_gap_scene(store, gap=timedelta(minutes=10))
        # No other channel activity seeded.

        request = _request(utterance="continue")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(output, store=store, mode=ChannelMode.full_rp)

        breadcrumbs = [m for m in messages[1:-1] if m.role == "system"]
        assert breadcrumbs == []

    def test_no_breadcrumb_when_gap_under_threshold(self, tmp_path: Path) -> None:
        """A gap under 5 minutes produces no breadcrumb even with activity."""
        store = HistoryStore(tmp_path / "history.db")
        gap = timedelta(minutes=2)
        self._seed_gap_scene(store, gap=gap)

        t1 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        self._seed_other_channel_activity(
            store, activity_time=t1 + timedelta(minutes=1)
        )

        request = _request(utterance="continue")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(output, store=store, mode=ChannelMode.full_rp)

        breadcrumbs = [m for m in messages[1:-1] if m.role == "system"]
        assert breadcrumbs == []

    def test_no_breadcrumb_when_activity_outside_gap(self, tmp_path: Path) -> None:
        """Cross-channel activity that predates the gap is not a breadcrumb."""
        store = HistoryStore(tmp_path / "history.db")
        gap = timedelta(minutes=10)
        self._seed_gap_scene(store, gap=gap)

        # Activity happened BEFORE the gap started (before scene start).
        before_gap = datetime(2025, 6, 1, 11, 30, 0, tzinfo=UTC)
        self._seed_other_channel_activity(store, activity_time=before_gap)

        request = _request(utterance="continue")
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(output, store=store, mode=ChannelMode.full_rp)

        breadcrumbs = [m for m in messages[1:-1] if m.role == "system"]
        assert breadcrumbs == []


# ---------------------------------------------------------------------------
# Interruption context injection
# ---------------------------------------------------------------------------


class TestInterruptionContextRendering:
    def test_interruption_context_injected_as_system_message(
        self,
        tmp_path: Path,
    ) -> None:
        """When interruption_context is set, a system message appears.

        Specifically, the system message appears immediately before the
        final user turn.
        """
        request = _request(
            interruption_context=(
                "Alice interrupted while you were forming a response."
            ),
        )
        store = _store_with(tmp_path, [])
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(output, store=store)

        # The last message is the user turn; the one before it should be
        # the interruption context system message.
        assert messages[-1].role == "user"
        assert messages[-2].role == "system"
        assert messages[-2].content == (
            "Alice interrupted while you were forming a response."
        )

    def test_no_interruption_context_no_extra_message(
        self,
        tmp_path: Path,
    ) -> None:
        """Without interruption_context, no extra system message is injected."""
        request = _request()  # interruption_context defaults to None
        store = _store_with(tmp_path, [])
        output = _pipeline_output(request=request)
        messages = assemble_chat_messages(output, store=store)

        # Only the initial system prompt + the final user turn.
        system_messages = [m for m in messages if m.role == "system"]
        assert len(system_messages) == 1  # just the main system prompt
