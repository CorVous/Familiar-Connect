"""Render a :class:`PipelineOutput` into a list of chat messages.

The pipeline is deliberately ignorant of chat layout — it produces a
per-layer :class:`BudgetResult` and leaves the question of "where
does each layer live in the final message list" to this module. The
renderer is the one place in the codebase that knows:

- Which layers belong in the system prompt (everything except
  :class:`Layer.recent_history` and :class:`Layer.depth_inject`).
- Where recent history comes from (the :class:`HistoryStore`, not
  the ``recent_history`` Contribution's text — the provider writes
  it as prose for the budgeter's accounting, but the renderer wants
  discrete ``Message`` objects).
- Where :class:`Layer.depth_inject` content goes (inserted between
  messages at a configurable position-from-end, per SillyTavern's
  ``@D N`` convention).
- How user turns are labelled so the LLM can tell speakers apart.
  Every user turn's content is prefixed with ``Speaker: `` — the
  same display-name already carried on :attr:`Message.name` for
  OpenAI-compatible backends. The redundant prefix exists because
  OpenRouter silently drops the OpenAI ``name`` field when routing
  to non-OpenAI models (Anthropic, Gemini, most local runners), so
  relying on ``name`` alone means those models never see who's
  talking. Assistant turns are NOT prefixed — the ``assistant``
  role is sufficient to distinguish the bot's own replies, and
  prefixing them would invite the LLM to echo the prefix back into
  the reply it generates.

Keeping all four rules in one file means the "where does what go"
decision moves with the file; adding a new layer is a one-edit
change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.context.types import Layer
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.context.pipeline import PipelineOutput
    from familiar_connect.history.store import HistoryStore

_SECTION_SEPARATOR = "\n\n"

_DEFAULT_HISTORY_WINDOW = 20
"""How many recent turns to pull from the store when the caller doesn't say."""

_SYSTEM_PROMPT_LAYER_ORDER: tuple[Layer, ...] = (
    Layer.core,
    Layer.character,
    Layer.content,
    Layer.history_summary,
    Layer.author_note,
)
"""Natural top-to-bottom order of layers inside the system prompt.

Layers not listed here are deliberately excluded:
:class:`Layer.recent_history` renders as discrete messages;
:class:`Layer.depth_inject` renders mid-chat.
"""

_SPEAKER_PREFIX_PREAMBLE = (
    "User messages in this conversation are prefixed with the speaker's "
    'display name followed by a colon — for example, "Alice: hello". '
    "Different prefixes mean different people. When replying, address "
    "users by name when it's natural; do NOT prefix your own replies with "
    "your name — just write the reply directly."
)
"""Fixed preamble prepended to every system prompt.

This is a property of the rendering contract itself, not something the
character or mode can tune. It tells the LLM how to interpret the
``Speaker: `` prefix the renderer attaches to user turns, and
explicitly instructs it not to echo the prefix back in its own
replies. Kept as a module-level constant (rather than a provider) so
the invariant moves with the code that enforces it."""


def _prefix_user_content(speaker: str | None, content: str) -> str:
    """Prepend ``Speaker: `` to user content when a speaker is known.

    The prefix is redundant with :attr:`Message.name` for backends
    that honour the OpenAI ``name`` field, but it's the only way
    non-OpenAI models routed through OpenRouter ever see the
    speaker — see the module docstring for the rationale. A
    ``None`` speaker (system-generated events, sanitised-to-empty
    display names) renders bare rather than producing literal
    ``None: content``.
    """
    if not speaker:
        return content
    return f"{speaker}: {content}"


def assemble_chat_messages(
    output: PipelineOutput,
    *,
    store: HistoryStore,
    history_window_size: int = _DEFAULT_HISTORY_WINDOW,
    depth_inject_position: int = 0,
    depth_inject_role: str = "system",
) -> list[Message]:
    """Return a list of :class:`Message` ready to hand to the LLM client.

    :param output: The :class:`PipelineOutput` from
        :meth:`ContextPipeline.assemble`.
    :param store: The :class:`HistoryStore` the recent-history window
        is read from. (The :class:`Layer.recent_history`
        contribution's text is ignored — it's a prose rendering for
        the budgeter's token accounting, not for the message list.)
    :param history_window_size: How many of the most-recent turns in
        the request's channel to include as discrete messages.
    :param depth_inject_position: Where to insert the
        :class:`Layer.depth_inject` content, measured as an offset
        from the end of the message list (excluding the final user
        turn). ``0`` means immediately before the final user turn
        (SillyTavern's ``@D 0``). Values larger than the chat
        buffer clamp to the top of history.
    :param depth_inject_role: Either ``"system"`` (default) or
        ``"user"`` — the role assigned to the inserted message.
    """
    request = output.request
    by_layer = output.budget.by_layer

    # 1. System prompt: every layer except recent_history and depth_inject.
    system_prompt = _build_system_prompt(by_layer)
    messages: list[Message] = [Message(role="system", content=system_prompt)]

    # 2. Recent history rendered as discrete messages. User turns are
    # prefixed with ``Speaker: `` so backends that drop the OpenAI
    # ``name`` field can still distinguish speakers; assistant turns
    # are left bare.
    turns = store.recent(
        familiar_id=request.familiar_id,
        channel_id=request.channel_id,
        limit=history_window_size,
    )
    for turn in turns:
        if turn.role == "user":
            messages.append(
                Message(
                    role="user",
                    content=_prefix_user_content(turn.speaker, turn.content),
                    name=turn.speaker,
                ),
            )
        else:
            messages.append(
                Message(role=turn.role, content=turn.content, name=None),
            )

    # 3. The final user turn — the triggering utterance from the request.
    messages.append(
        Message(
            role="user",
            content=_prefix_user_content(request.speaker, request.utterance),
            name=request.speaker,
        ),
    )

    # 4. Depth-inject at position-from-end, computed against the full list
    # (including the final user turn). ``position=0`` means immediately
    # before the final user turn — i.e. ``len - 1`` from the top. Values
    # larger than the chat buffer clamp to just after the system prompt.
    depth_text = by_layer.get(Layer.depth_inject, "").strip()
    if depth_text:
        distance_from_end = max(1, depth_inject_position)
        insert_at = max(1, len(messages) - distance_from_end)
        messages.insert(
            insert_at,
            Message(role=depth_inject_role, content=depth_text),
        )

    return messages


def _build_system_prompt(by_layer: dict[Layer, str]) -> str:
    # The speaker-prefix preamble is always first so the LLM reads the
    # rendering convention before any character or mode content that
    # might assume it.
    sections: list[str] = [_SPEAKER_PREFIX_PREAMBLE]
    for layer in _SYSTEM_PROMPT_LAYER_ORDER:
        text = by_layer.get(layer, "").strip()
        if text:
            sections.append(text)
    return _SECTION_SEPARATOR.join(sections)
