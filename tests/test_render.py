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

from typing import TYPE_CHECKING, Any

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
