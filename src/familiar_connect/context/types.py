"""Typed data structures for the context pipeline.

Small, immutable-ish records the rest of the pipeline passes around.
Kept deliberately boring — no behaviour beyond construction — so tests
against them are cheap and the shape can be reasoned about at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Layer(Enum):
    """The layers that make up the assembled system prompt.

    Order here is *not* priority order — priority is carried on each
    ``Contribution`` — but the members are listed in a natural top-to-
    bottom reading order for the assembled prompt.
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

    ``priority`` is a within-layer ordering hint; higher wins under
    budget pressure. ``estimated_tokens`` is the provider's best guess
    and is re-counted by the budgeter before final assembly. ``source``
    is a short human-readable identifier used for logging and the
    monitoring dashboard.
    """

    layer: Layer
    priority: int
    text: str
    estimated_tokens: int
    source: str


@dataclass(frozen=True)
class ContextRequest:
    """The input to a single run of the context pipeline.

    One of these is built per incoming event (user utterance, Twitch
    event, chattiness-triggered interjection). The pipeline's providers
    each read it, produce a list of ``Contribution``s, and the budgeter
    merges everything into the final ``SystemPromptLayers``.

    :param guild_id: Discord guild (server) id.
    :param familiar_id: Which familiar is replying. Distinct familiars
        in the same guild have distinct memory directories.
    :param channel_id: Discord channel id (text or voice).
    :param speaker: Display name of whoever triggered this turn, or
        ``None`` for system-generated turns like Twitch events.
    :param utterance: The triggering text. For voice this is the final
        transcription; for text it is the message content.
    :param modality: Whether this turn will be delivered via voice or
        text. Providers and processors can branch on this.
    :param budget_tokens: Total budget for the assembled system prompt.
    :param deadline_s: Hard wall-clock deadline for the whole pipeline,
        in seconds. Providers that miss it are dropped, not awaited.
    """

    guild_id: int
    familiar_id: str
    channel_id: int
    speaker: str | None
    utterance: str
    modality: Modality
    budget_tokens: int
    deadline_s: float
