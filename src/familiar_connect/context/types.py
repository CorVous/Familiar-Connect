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
    """The input to a single run of the context pipeline.

    One of these is built per incoming event (user utterance, Twitch
    event, chattiness-triggered interjection). The pipeline's providers
    each read it, produce a list of ``Contribution``s, and the budgeter
    merges everything into the final ``SystemPromptLayers``.

    A Familiar-Connect install runs exactly one familiar at a time —
    see ``docs/architecture/configuration-model.md``. ``familiar_id``
    therefore identifies which character folder on disk is active;
    ``channel_id`` partitions the per-conversation recent history
    window so two simultaneous conversations don't bleed into each
    other; ``guild_id`` is observability only.

    :param familiar_id: Which familiar is replying. Matches the folder
        name under ``data/familiars/``.
    :param channel_id: Discord channel id (text or voice). Used as
        the partition key for the per-conversation recent history
        window.
    :param guild_id: Discord guild (server) id where this turn
        happened. Observability and routing only — never used as a
        partition key for memory or history. ``None`` is permitted
        for non-Discord events (e.g. Twitch).
    :param speaker: Display name of whoever triggered this turn, or
        ``None`` for system-generated turns like Twitch events.
    :param utterance: The triggering text. For voice this is the final
        transcription; for text it is the message content.
    :param modality: Whether this turn will be delivered via voice or
        text. Providers and processors can branch on this.
    :param budget_tokens: Total budget for the assembled system prompt.
    :param deadline_s: Hard wall-clock deadline for the whole pipeline,
        in seconds. Providers that miss it are dropped, not awaited.
    :param preprocessor_contributions: Tuple of :class:`Contribution`
        objects accumulated by pre-processors as they run. The
        pipeline merges these into the budgeter input alongside
        provider contributions, so a pre-processor can carry its own
        output forward without becoming a provider. Defaults to the
        empty tuple; pre-processors append by constructing a fresh
        request via :func:`dataclasses.replace`.
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
    interruption_context: str | None = None
    """Optional system note describing a voice interruption. Populated
    by the Step 8 long-interruption handler (see
    ``docs/roadmap/interruption-flow.md``) with a short annotation
    like ``{speaker} interrupted while you were forming a response.
    They said: "{transcript}"`` so the regenerated reply can
    acknowledge the cutoff. The renderer inserts it as a ``system``
    message immediately before the final user turn. ``None`` or the
    empty string means no note is rendered — the default for every
    non-interrupted turn."""
