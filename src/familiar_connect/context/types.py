"""Typed data structures for the context pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Layer(Enum):
    """Layers of the assembled system prompt.

    Order here is not priority order (priority lives on Contribution);
    members are listed in top-to-bottom reading order.
    """

    core = "core"
    """Core instructions and safety rails."""

    character = "character"
    """Character-card fields (self/*.md for a familiar)."""

    content = "content"
    """Memory-search snippets from ContentSearchProvider."""

    history_summary = "history_summary"
    """Rolling summary of older turns from HistoryProvider."""

    recent_history = "recent_history"
    """Verbatim recent turns. Rendered as discrete messages, not prose."""

    author_note = "author_note"
    """Per-guild author's note text."""

    depth_inject = "depth_inject"
    """Depth-position injections (inserted N messages from the end)."""


class Modality(Enum):
    """Which interface the user is using for this turn.

    Voice and text have meaningfully different latency budgets and
    tolerable context sizes; providers and processors can branch on
    this, and per-guild config selects which ones run for which
    modality.
    """

    voice = "voice"
    text = "text"


@dataclass(frozen=True)
class Contribution:
    """A single piece of content a provider wants to inject.

    ``priority``: within-layer ordering hint, higher wins under budget
    pressure. ``estimated_tokens``: provider's best guess, re-counted
    by the budgeter before final assembly.
    """

    layer: Layer
    priority: int
    text: str
    estimated_tokens: int
    source: str


@dataclass(frozen=True)
class PendingTurn:
    """A user message buffered by the conversation monitor.

    Carried on :attr:`ContextRequest.pending_turns` so the renderer
    can inject every accumulated message — not just the final trigger —
    into the chat payload sent to the LLM.
    """

    speaker: str | None
    text: str


@dataclass(frozen=True)
class ContextRequest:
    """Input to a single pipeline run.

    Built per incoming event (user utterance, Twitch event, chattiness
    interjection). See ``docs/architecture/configuration-model.md``
    for how ``familiar_id`` / ``channel_id`` / ``guild_id`` partition
    data.

    :param preprocessor_contributions: accumulated by pre-processors;
        merged into budgeter input alongside provider contributions.
        Pre-processors append via ``dataclasses.replace``.
    """

    familiar_id: str
    channel_id: int
    guild_id: int | None
    speaker: str | None
    utterance: str
    modality: Modality
    budget_tokens: int
    deadline_s: float
    pending_turns: tuple[PendingTurn, ...] = ()
    """User messages buffered by the conversation monitor since the last
    response. When non-empty, the renderer appends *all* of these as
    user turns instead of the single ``utterance``. The last entry
    should match ``utterance`` / ``speaker``."""
    preprocessor_contributions: tuple[Contribution, ...] = ()
