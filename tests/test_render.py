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
        assert messages[1].role == "user"
        assert messages[1].content == "hi"
        assert messages[2].role == "assistant"
        assert messages[2].content == "hello back"
        assert messages[3].role == "user"
        assert messages[3].content == "how are you?"
        # The final user turn is the request's utterance.
        assert messages[-1].role == "user"
        assert messages[-1].content == "latest thing Alice said"


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

        # system, "one", "two", INJECT, final user ("latest thing Alice said").
        assert messages[-2].role == "system"
        assert messages[-2].content == "INJECT"
        assert messages[-1].content == "latest thing Alice said"

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

        # system, one, two, three, INJECT, four, five
        contents = [m.content for m in messages]
        assert contents == ["core", "one", "two", "three", "INJECT", "four", "five"]

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

        # Clamped: system, INJECT, one, two
        assert messages[0].role == "system"
        assert messages[1].content == "INJECT"
        assert messages[2].content == "one"
        assert messages[3].content == "two"

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

        # system, last 2 ("four", "five"), final user ("six")
        contents = [m.content for m in messages]
        assert contents == ["core", "four", "five", "six"]
