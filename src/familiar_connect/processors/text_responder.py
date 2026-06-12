"""Text reply orchestrator.

Consumes ``discord.text`` events; produces LLM reply, posted via
injected ``send_text`` callback. Mirrors :class:`VoiceResponder` but
skips TTS — text channels render assistant string directly.

Owns user-turn writes for ``discord.text`` (single-writer per channel
keeps read-after-write consistency for ``RecentHistoryLayer`` in same
task — separate :class:`HistoryWriter` task would race responder's
``assemble`` call). Cancellation flows through :class:`TurnRouter`
so future barge-in (e.g. user sends follow-up mid-stream) cancels
in-flight LLM work.
"""

from __future__ import annotations

import contextlib
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.activities.engine import GateAction
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.identity import Author
from familiar_connect.llm import LLMDelta, Message
from familiar_connect.silence import SilentDetector

# LLM ping vocabulary: ``[@DisplayName]`` markers in model's
# output. symmetric with form ``RecentHistoryLayer`` rewrites
# inbound ``<@USER_ID>`` mentions into
_PING_MARKER_RE = re.compile(r"\[@([^\]\n]+)\]")

# Thread-reply marker. either ``[↩]`` (matches inbound reply glyph
# read path uses) or ``[reply]`` for tokenizer safety. optional
# whitespace + token after glyph captures target
# ``platform_message_id`` — LLM can point at specific message
# (visible as ``#<id>`` in recent history). bare ``[↩]`` keeps
# legacy meaning: thread to triggering message
_THREAD_MARKER_RE = re.compile(r"\[(?:↩|reply)(?:\s+([^\]\n]+))?\]")

# Defense-in-depth: when model leaks metadata-shaped prefix like
# ``[#1500709436557445449]`` or ``[4:03AM]`` at very start of reply
# (mimicking recent-history rendering format), drop it.
# Conservative: only matches bracketed clump containing ``#``,
# digits + ``:``, or ``AM/PM``, optionally followed by another such
# bracket, so we don't eat a legitimate ``[note]`` opener
_LEAKED_META_PREFIX_RE = re.compile(
    r"^\s*(?:\[[^\]\n]*(?:#\d|\d:\d|[AP]M)[^\]\n]*\]\s*)+"
)

# Short addendum to system prompt explaining two output controls.
# Costs ~5 lines; doesn't enumerate per-channel participants — LLM
# grounds names in recent history, resolver attempts match against
# active speakers at send time
_BOT_OUTPUT_INSTRUCTIONS = (
    "## Output controls\n\n"
    "- The `[H:MM Name #id]` prefix on each user message is read-only "
    "metadata the system adds for you. **Do not** start your replies "
    "with that shape — just write the message body.\n"
    "- Ping a user by writing `[@DisplayName]` using a name that "
    "appears in recent messages. Unrecognised names render as "
    "plain text without pinging.\n"
    "- Optionally prefix your message with `[↩]` to thread it as "
    "a reply to the message you're responding to. Useful when the "
    "channel is busy and it isn't obvious who you're addressing. "
    "Without `[↩]`, your message posts normally.\n"
    "- To reply to a *specific* earlier message, write "
    "`[↩ <message_id>]` using the `#<id>` shown next to that "
    "message in recent history. Unknown ids fall back to the "
    "triggering message."
)


def _rewrite_pings(
    content: str,
    label_to_key: dict[str, str],
) -> tuple[str, tuple[int, ...]]:
    """Rewrite ``[@DisplayName]`` markers; collect resolved user ids.

    Known labels become Discord-native ``<@user_id>`` mentions and
    contribute to returned tuple (passed to ``AllowedMentions``).
    Unknown labels degrade to plain ``@DisplayName`` text — no ping,
    no error. Non-Discord canonical keys also degrade (bot only
    sends to Discord today; future platform support extends this).
    """
    resolved: list[int] = []

    def _sub(match: re.Match[str]) -> str:
        label = match.group(1)
        key = label_to_key.get(label)
        if key is None:
            return f"@{label}"
        platform, _, user_id = key.partition(":")
        if platform != "discord":
            return f"@{label}"
        try:
            resolved.append(int(user_id))
        except ValueError:
            return f"@{label}"
        return f"<@{user_id}>"

    rewritten = _PING_MARKER_RE.sub(_sub, content)
    return rewritten, tuple(resolved)


def _strip_leaked_metadata_prefix(content: str) -> str:
    """Drop a leaked ``[#id]`` / ``[H:MMpm]`` style prefix from head.

    Recent-history layer prefixes every user turn with
    ``[H:MM Name #id]`` clump; some models echo that shape back.
    Regex conservative — only matches bracket clumps smelling like
    metadata, leaving legitimate ``[note]`` openings alone.
    """
    return _LEAKED_META_PREFIX_RE.sub("", content, count=1)


def _consume_thread_marker(content: str) -> tuple[str, bool, str | None]:
    """Strip thread markers; return ``(stripped, wanted_thread, target_id)``.

    Any occurrence anywhere in output triggers threading — lenient
    about placement, since models unreliable about exact formatting.
    Multiple markers collapse to single signal; *first* explicit id
    wins. ``target_id`` is ``None`` when marker is bare (legacy form:
    thread to triggering message). Leading ``#`` sigil stripped —
    recent-history surfaces ids as ``#<id>`` and models routinely
    echo sigil back inside marker. Surrounding whitespace from
    leading marker is cleaned so posted message doesn't start with
    stray newline.
    """
    target_id: str | None = None
    matches = list(_THREAD_MARKER_RE.finditer(content))
    if not matches:
        return content, False, None
    for m in matches:
        captured = m.group(1)
        if captured is not None and captured.strip():
            target_id = captured.strip().lstrip("#").strip() or None
            if target_id:
                break
    stripped = _THREAD_MARKER_RE.sub("", content).lstrip()
    return stripped, True, target_id


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from contextlib import AbstractAsyncContextManager as AsyncContextManager

    from familiar_connect.activities.engine import ActivityEngine
    from familiar_connect.bus.envelope import TurnScope
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.context.assembler import Assembler
    from familiar_connect.focus import FocusManager
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.llm import LLMClient
    from familiar_connect.tools.registry import ToolContext, ToolRegistry
    from familiar_connect.typing_interrupt import TypingInterruptHandler

    # Optional Discord typing-indicator hook. wrapped around
    # streaming + send path so channel shows ``Bot is typing…``
    # while LLM is generating. ``None`` = indicator disabled
    TriggerTyping = Callable[[int], AsyncContextManager[None]]

_logger = logging.getLogger("familiar_connect.processors.text_responder")


class TextResponder:
    """Streams LLM replies for ``discord.text`` events; posts via ``send_text``."""

    name: str = "text-responder"
    topics: tuple[str, ...] = (TOPIC_DISCORD_TEXT,)

    def __init__(
        self,
        *,
        assembler: Assembler,
        llm_client: LLMClient,
        send_text: Callable[
            [int, str, str | None, tuple[int, ...]],
            Awaitable[str | None],
        ],
        history_store: AsyncHistoryStore,
        router: TurnRouter,
        familiar_id: str,
        trigger_typing: TriggerTyping | None = None,
        typing_handler: TypingInterruptHandler | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_context_factory: (
            Callable[[int, str, dict[str, str]], ToolContext] | None
        ) = None,
        post_history_instructions: str = "",
        display_tz: str = "UTC",
        focus_manager: FocusManager | None = None,
        loop_max_iterations: int = 5,
        activity_engine: ActivityEngine | None = None,
    ) -> None:
        self._assembler = assembler
        self._llm = llm_client
        self._send_text = send_text
        self._history = history_store
        self._sync_history = history_store.sync
        self._router = router
        self._familiar_id = familiar_id
        # Per-familiar etiquette appended to the trailing reminder
        # (post-history). empty = omitted.
        self._post_history_instructions = post_history_instructions
        # IANA zone for the final-reminder clock (validated at config load)
        self._display_tz = display_tz
        # Discord ``Bot is typing…`` indicator factory; ``None`` opts out
        self._trigger_typing = trigger_typing
        # Typing-event policy — bot pingpong backoff + user-typing cancel.
        # ``None`` disables both (tests, future non-discord paths)
        self._typing_handler = typing_handler
        # Agentic-loop tool registry. paired with non-None
        # ``tool_context_factory`` and ``llm.tool_calling_enabled``,
        # responder runs ``agentic_loop`` instead of bare ``chat_stream``
        self._tool_registry = tool_registry
        self._tool_context_factory = tool_context_factory
        # Hard cap on agentic-loop rounds per turn ([tools].loop_max_iterations)
        self._loop_max_iterations = loop_max_iterations
        # Attentional focus controller; None = backward-compat (no focus gating)
        self._focus_manager = focus_manager
        # Absence controller; None = no activity gating (zero behavior change)
        self._activity_engine = activity_engine
        # In-process dedup; bus doesn't republish today, cheap insurance
        self._seen: set[str] = set()

    async def handle(self, event: Event, bus: EventBus) -> None:
        if event.topic != TOPIC_DISCORD_TEXT:
            return
        if event.event_id in self._seen:
            return
        payload = event.payload
        if not isinstance(payload, dict):
            return
        if payload.get("familiar_id") != self._familiar_id:
            return
        channel_id = payload.get("channel_id")
        content = payload.get("content") or ""
        # Idle nudge: synthetic wake event has no real user content —
        # it just earns the model a focused turn (see _emit_idle_nudge)
        is_wake = payload.get("wake") is True
        if not isinstance(channel_id, int):
            return
        if not content and not is_wake:
            return
        raw_author = payload.get("author")
        author = raw_author if isinstance(raw_author, Author) else None
        raw_guild = payload.get("guild_id")
        guild_id = raw_guild if isinstance(raw_guild, int) else None
        raw_msg_id = payload.get("message_id")
        message_id = raw_msg_id if isinstance(raw_msg_id, str) else None
        raw_reply_to = payload.get("reply_to_message_id")
        reply_to_message_id = raw_reply_to if isinstance(raw_reply_to, str) else None
        raw_mentions = payload.get("mentions") or ()
        mentions: tuple[Author, ...] = (
            tuple(m for m in raw_mentions if isinstance(m, Author))
            if isinstance(raw_mentions, (tuple, list))
            else ()
        )
        raw_images = payload.get("images")
        images: dict[str, str] = raw_images if isinstance(raw_images, dict) else {}

        self._seen.add(event.event_id)
        # Quiet-clock for idle activity nudges — every handled event
        # counts as traffic, even staged/suppressed ones
        if self._activity_engine is not None:
            self._activity_engine.note_traffic()
        # Honor any active pingpong-backoff (another bot has been
        # typing in this channel) before claiming lane. then reset
        # ladder so future bot-typing events start at initial step
        if self._typing_handler is not None:
            await self._typing_handler.wait_for_backoff(channel_id)
            self._typing_handler.notify_user_message(channel_id=channel_id)
        scope = self._router.begin_turn(
            session_id=event.session_id, turn_id=event.turn_id
        )

        # Refresh canonical identity rows for everyone we know about
        # in this turn. soft annotation: accounts table is "what we
        # most recently saw" cache, not source of truth
        if author is not None:
            await self._history.upsert_account(author)
            if guild_id is not None and author.guild_nick is not None:
                await self._history.upsert_guild_nick(
                    canonical_key=author.canonical_key,
                    guild_id=guild_id,
                    nick=author.guild_nick,
                )
        for m in mentions:
            await self._history.upsert_account(m)
            if guild_id is not None and m.guild_nick is not None:
                await self._history.upsert_guild_nick(
                    canonical_key=m.canonical_key,
                    guild_id=guild_id,
                    nick=m.guild_nick,
                )

        # Activity gate — engine optional (None ⇒ zero behavior change).
        # SUPPRESS: she's away from the screen — record the user turn
        # staged and stop (no typing, no LLM, no reply). JUDGMENT:
        # normal reply flow plus engine state line injected for this
        # turn only.
        gate = (
            self._activity_engine.gate(payload)
            if self._activity_engine is not None
            else None
        )
        suppressed = gate is not None and gate.action is GateAction.SUPPRESS
        judgment = gate is not None and gate.action is GateAction.JUDGMENT

        # Focus-aware staging: when a focus_manager is wired and
        # channel is not focused, persist as staged turn and return —
        # no LLM call, no reply. unread digest surfaces it next turn.
        # Wake events skip staging+persist entirely: they carry no real
        # user content, only earn the model a focused turn.
        focused = self._focus_manager is None or self._focus_manager.is_focused(
            channel_id
        )
        if not is_wake:
            # Persist user turn *before* streaming so RecentHistoryLayer
            # in same task sees it. mirrors VoiceResponder
            user_turn = await self._history.append_turn(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                role="user",
                content=content,
                author=author,
                guild_id=guild_id,
                platform_message_id=message_id,
                reply_to_message_id=reply_to_message_id,
                consumed=focused and not suppressed,  # Staged when unfocused/absent
            )
            if mentions:
                await self._history.record_mentions(
                    turn_id=user_turn.id,
                    canonical_keys=[m.canonical_key for m in mentions],
                )
            if suppressed:
                # live missed-ping capture: covers cross-channel pings
                # and reply-pings the at-return content scan can't see
                if (
                    self._activity_engine is not None
                    and payload.get("pings_bot") is True
                ):
                    self._activity_engine.note_missed_ping(user_turn.id)
                author_label = author.display_name if author else "unknown"
                _logger.info(
                    f"{ls.tag('Activity', ls.G)} suppressed "
                    f"{ls.kv('ch', str(channel_id), vc=ls.LW)} "
                    f"{ls.kv('from', author_label, vc=ls.LW)} "
                    f"{ls.kv('text', ls.trunc(content, 200), vc=ls.LW)}"
                )
                return
            if not focused:
                author_label = author.display_name if author else "unknown"
                _logger.info(
                    f"{ls.tag('📥 Staged', ls.Y)} "
                    f"{ls.kv('ch', str(channel_id), vc=ls.LW)} "
                    f"{ls.kv('from', author_label, vc=ls.LW)} "
                    f"{ls.kv('text', content, vc=ls.LW)}"
                )
                # Focused channel idle long enough → nudge the model so
                # stranded unreads don't starve. model decides via
                # shift_focus; the nudge never moves focus itself.
                if self._focus_manager is not None and self._focus_manager.should_wake(
                    channel_id
                ):
                    await self._emit_idle_nudge(bus)
                return
        elif suppressed:
            # wake event while absent — carries no user content, drop
            return

        # Seed retrieval before assembly so RagContextLayer sees the cue
        self._assembler.set_rag_cue(content)

        # Build label→canonical_key map for resolving any ``[@X]``
        # markers LLM emits. not surfaced in prompt; LLM grounds on
        # names visible in recent history, resolver matches against
        # active speakers at send time
        label_to_key = self._build_ping_resolver(
            channel_id=channel_id, guild_id=guild_id
        )

        # Typing indicator opens lazily inside ``_stream_reply`` once
        # silent-sentinel decision resolves to ``False`` — reasoning
        # tokens routinely precede ``<silent>`` verdict; don't want
        # ``Bot is typing…`` flickering on for those
        reply = await self._stream_reply(
            scope,
            channel_id=channel_id,
            guild_id=guild_id,
            images=images,
            activity_state_line=(
                gate.state_line if gate is not None and judgment else None
            ),
        )
        if reply is None or scope.is_cancelled():
            # silent slip-away: deferred start (if any) still applies —
            # a stale staged start must never leak into a later turn
            if self._activity_engine is not None:
                await self._activity_engine.end_turn()
            return
        # Discord rejects empty / whitespace-only messages (HTTP 400,
        # error code 50006). empty reply usually means LLM stream
        # emitted no deltas — bad model name, content filter, or
        # upstream error frame parser silently dropped
        if not reply.strip():
            _logger.warning(
                f"{ls.tag('Text', ls.Y)} "
                f"{ls.kv('skip', 'empty_reply', vc=ls.LY)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            if self._activity_engine is not None:
                await self._activity_engine.end_turn()
            return

        # Threading opt-in via LLM-emitted marker. strip marker and
        # only pass reply target to ``send_text`` when model
        # deliberately asked to thread. captured id (e.g.
        # ``[↩ msg-001]``) routes to specific message when known;
        # otherwise fall back to triggering message
        unthreaded, wants_thread, target_id = _consume_thread_marker(reply)
        unthreaded = _strip_leaked_metadata_prefix(unthreaded)
        rewritten, mention_user_ids = _rewrite_pings(unthreaded, label_to_key)
        thread_target: str | None = None
        if wants_thread:
            if (
                target_id
                and (
                    await self._history.lookup_turn_by_platform_message_id(
                        familiar_id=self._familiar_id,
                        platform_message_id=target_id,
                    )
                )
                is not None
            ):
                thread_target = target_id
            else:
                thread_target = message_id

        # If the model shifted focus this turn, send to the new channel
        send_channel_id = channel_id
        if self._focus_manager is not None:
            pending = self._focus_manager.pending_text_focus()
            if pending is not None:
                send_channel_id = pending

        try:
            sent_message_id = await self._send_text(
                send_channel_id, rewritten, thread_target, mention_user_ids
            )
        except Exception as exc:  # noqa: BLE001 — surface but don't crash loop
            _logger.warning(
                f"{ls.tag('Text', ls.R)} {ls.kv('send_error', repr(exc), vc=ls.R)}"
            )
            if self._activity_engine is not None:
                await self._activity_engine.end_turn()
            return
        _logger.info(
            f"{ls.tag('💬 Text', ls.G)} "
            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
            f"{ls.kv('chars', str(len(rewritten)), vc=ls.LW)} "
            f"{ls.kv('thread', '1' if wants_thread else '0', vc=ls.LB)} "
            f"{ls.kv('text', ls.trunc(rewritten, 200), vc=ls.LW)}"
        )

        if scope.is_cancelled():
            return

        # Persist what bot actually sent, including whether we
        # threaded. ``reply_to_message_id`` matches what we passed to
        # ``send_text`` — audit trail for "did bot thread this reply?"
        await self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=send_channel_id,
            role="assistant",
            content=rewritten,
            author=None,
            guild_id=guild_id,
            platform_message_id=sent_message_id,
            reply_to_message_id=thread_target,
        )
        self._router.end_turn(scope)
        if self._focus_manager is not None:
            await self._focus_manager.end_turn()
        if self._activity_engine is not None:
            if judgment:
                # real reply on a judgment turn ⇒ came back early
                await self._activity_engine.notify_reply_sent()
            # applies any tool-deferred activity start
            await self._activity_engine.end_turn()

    async def _emit_idle_nudge(self, bus: EventBus) -> None:
        """Publish a synthetic wake event for the focused text channel.

        Earns the model one focused turn so it sees the unread digest
        and can choose to shift_focus. Never moves focus itself.
        """
        fm = self._focus_manager
        if fm is None:
            return
        focus_ch = fm.get_focus("text")
        if focus_ch is None:
            return
        fm.mark_nudge_pending()
        synth_id = uuid.uuid4().hex
        await bus.publish(
            Event(
                event_id=synth_id,
                turn_id=f"idle-wake-{synth_id}",
                session_id=str(focus_ch),
                parent_event_ids=(),
                topic=TOPIC_DISCORD_TEXT,
                timestamp=datetime.now(tz=UTC),
                sequence_number=0,
                payload={
                    "familiar_id": self._familiar_id,
                    "channel_id": focus_ch,
                    "content": "[idle: unread messages waiting elsewhere]",
                    "author": None,
                    "wake": True,
                },
            )
        )
        _logger.info(
            f"{ls.tag('⏰ Nudge', ls.LC)} "
            f"{ls.kv('focus', fm.channel_label(focus_ch), vc=ls.LW)}"
        )

    def _build_ping_resolver(
        self,
        *,
        channel_id: int,
        guild_id: int | None,
    ) -> dict[str, str]:
        """Return ``label → canonical_key`` map for resolving pings.

        Pulls recent distinct authors, resolves labels via
        :meth:`HistoryStore.resolve_label` so keys match names
        ``RecentHistoryLayer`` puts in front of model. Map *not*
        surfaced in prompt — LLM grounds on names in recent history,
        resolver matches what it emits at send time. Ambiguous labels
        (two participants with same nick) keep first-write; warning
        logged.
        """
        authors = self._sync_history.recent_distinct_authors(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            limit=20,
        )
        label_to_key: dict[str, str] = {}
        for a in authors:
            label = self._sync_history.resolve_label(
                canonical_key=a.canonical_key,
                guild_id=guild_id,
                familiar_id=self._familiar_id,
            )
            if label in label_to_key:
                if label_to_key[label] != a.canonical_key:
                    keys_pair = f"{label_to_key[label]},{a.canonical_key}"
                    _logger.warning(
                        f"{ls.tag('Text', ls.Y)} "
                        f"{ls.kv('ambiguous_label', label, vc=ls.LY)} "
                        f"{ls.kv('keys', keys_pair, vc=ls.LW)}"
                    )
                continue
            label_to_key[label] = a.canonical_key
        return label_to_key

    async def _stream_reply(
        self,
        scope: TurnScope,
        *,
        channel_id: int,
        guild_id: int | None = None,
        images: dict[str, str] | None = None,
        activity_state_line: str | None = None,
    ) -> str | None:
        ctx = AssemblyContext(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            viewer_mode="text",
            guild_id=guild_id,
        )
        prompt = await self._assembler.assemble(ctx)
        # Focus context for unread digest + focus directive
        focus_ch = (
            self._focus_manager.get_focus("text") if self._focus_manager else None
        )
        unread_digest: dict[int, int] | None = None
        if self._focus_manager is not None:
            unread_digest = await self._history.staged_channels(
                familiar_id=self._familiar_id
            )
        # Always append short output-controls instruction so model
        # knows ``[@X]`` and ``[↩]`` are routable. costs ~5 lines of
        # context, no per-channel enumeration. final reminder
        # restates current time + special inputs so model doesn't
        # drift on long-lived caches
        ch_names = self._focus_manager.channel_names if self._focus_manager else {}
        reminder = build_final_reminder(
            viewer_mode="text",
            include_time=False,
            focus_channel_id=focus_ch,
            unread_digest=unread_digest,
            channel_names=ch_names,
        )
        system = "\n\n".join(
            s for s in (prompt.system_prompt, _BOT_OUTPUT_INSTRUCTIONS, reminder) if s
        )
        messages: list[Message] = [Message(role="system", content=system)]
        messages.extend(prompt.recent_history)
        # Recency-biased models routinely ignore mode/format
        # directives buried at top of long context. re-emit as
        # trailing ``system`` message so they sit right before
        # assistant's next turn
        trailing = build_final_reminder(
            viewer_mode="text",
            display_tz=self._display_tz,
            include_mode_instruction=True,
            post_history_instructions=self._post_history_instructions,
            focus_channel_id=focus_ch,
            unread_digest=unread_digest,
            channel_names=ch_names,
        )
        if activity_state_line:
            # judgment-turn state line — this turn only, deepest slot
            # (trailing system message wins on recency-biased models)
            trailing = f"{trailing}\n\n{activity_state_line}"
        messages.append(Message(role="system", content=trailing))

        tool_mode = (
            self._tool_registry is not None
            and self._tool_context_factory is not None
            and (
                self._llm.tool_calling_enabled
                or getattr(self._llm, "image_tools_enabled", False)
            )
        )
        if tool_mode:
            return await self._stream_reply_with_tools(
                scope,
                channel_id=channel_id,
                guild_id=guild_id,
                messages=messages,
                images=images or {},
            )

        accumulated: list[str] = []
        silent = SilentDetector()
        # Typing indicator stays closed until ``SilentDetector`` rules
        # out sentinel — keeps ``Bot is typing…`` from flickering
        # during reasoning that resolves to ``<silent>``.
        # ``AsyncExitStack`` owns eventual ``__aexit__`` whether
        # stream finishes, scope is cancelled, or chat_stream raises
        typing_started = False
        async with contextlib.AsyncExitStack() as stack:
            try:
                async for delta in self._llm.chat_stream(messages):
                    if scope.is_cancelled():
                        return None
                    accumulated.append(delta)
                    decision = silent.feed(delta)
                    if decision is True:
                        # Mirror cancellation: no send, no assistant turn.
                        # 'decision=silent' log replaces would-be
                        # empty-reply warning (which targets bad model /
                        # content-filter cases, not deliberate silence)
                        _logger.info(
                            f"{ls.tag('💤 Text', ls.B)} "
                            f"{ls.kv('decision', 'silent', vc=ls.LB)} "
                            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                        )
                        return None
                    if (
                        decision is False
                        and not typing_started
                        and self._trigger_typing is not None
                    ):
                        await stack.enter_async_context(
                            self._trigger_typing(channel_id)
                        )
                        typing_started = True
            except Exception as exc:  # noqa: BLE001 — stream errors shouldn't crash loop
                _logger.warning(
                    f"{ls.tag('Text', ls.R)} "
                    f"{ls.kv('llm_stream_error', repr(exc), vc=ls.R)}"
                )
                return None
            return "".join(accumulated)

    async def _stream_reply_with_tools(
        self,
        scope: TurnScope,
        *,
        channel_id: int,
        guild_id: int | None,
        messages: list[Message],
        images: dict[str, str] | None = None,
    ) -> str | None:
        """Agentic-loop variant of :meth:`_stream_reply`.

        Streams content into silent detector + typing indicator
        exactly like bare path, but defers terminal text to loop
        helper which can fold in tool execution. Intermediate
        assistant turns (with ``tool_calls``) and ``role=tool`` turns
        persisted via ``on_iteration_end``; terminal turn skipped
        here, left to :meth:`handle` to persist alongside
        ``send_text`` call.
        """
        # `_tool_context_factory` guaranteed non-None in this branch
        # (see callsite). same for `_tool_registry`
        ctx_factory = self._tool_context_factory
        registry = self._tool_registry
        if ctx_factory is None or registry is None:  # pragma: no cover — guard
            return None
        ctx = ctx_factory(channel_id, scope.turn_id, images or {})

        # ``agentic_loop`` returns terminal assistant content.
        # Responder's ``handle`` persists it + posts via send_text
        silent = SilentDetector()
        typing_started = False
        bail_silent = False

        async with contextlib.AsyncExitStack() as stack:

            async def _on_delta(delta: LLMDelta) -> None:
                nonlocal typing_started, bail_silent
                if scope.is_cancelled() or bail_silent:
                    return
                if not delta.content:
                    return
                decision = silent.feed(delta.content)
                if decision is True:
                    bail_silent = True
                    return
                if (
                    decision is False
                    and not typing_started
                    and self._trigger_typing is not None
                ):
                    await stack.enter_async_context(self._trigger_typing(channel_id))
                    typing_started = True

            async def _on_iter_end(
                assistant: Message,
                tool_msgs: list[Message],
            ) -> None:
                # Skip persistence for terminal text-only iteration —
                # ``handle()`` writes final assistant turn alongside
                # platform message id from ``send_text``
                if not assistant.tool_calls:
                    return
                import json as _json  # noqa: PLC0415 — local import keeps top minimal

                await self._history.append_turn(
                    familiar_id=self._familiar_id,
                    channel_id=channel_id,
                    role="assistant",
                    content=assistant.content_str,
                    author=None,
                    guild_id=guild_id,
                    tool_calls_json=_json.dumps(assistant.tool_calls),
                )
                for tm in tool_msgs:
                    from familiar_connect.tools.loop import (  # noqa: PLC0415
                        tool_content_as_text,
                    )

                    await self._history.append_turn(
                        familiar_id=self._familiar_id,
                        channel_id=channel_id,
                        role="tool",
                        content=tool_content_as_text(tm.content),
                        tool_call_id=tm.tool_call_id,
                        guild_id=guild_id,
                    )

            try:
                # Local import — avoids dragging tools into module
                # top for callers that never use them
                from familiar_connect.tools.loop import agentic_loop  # noqa: PLC0415

                result = await agentic_loop(
                    llm=self._llm,
                    messages=messages,
                    registry=registry,
                    ctx=ctx,
                    on_delta=_on_delta,
                    on_iteration_end=_on_iter_end,
                    max_iterations=self._loop_max_iterations,
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    f"{ls.tag('Text', ls.R)} "
                    f"{ls.kv('llm_agentic_error', repr(exc), vc=ls.R)}"
                )
                return None

        if bail_silent or result.is_silent:
            _logger.info(
                f"{ls.tag('💤 Text', ls.B)} "
                f"{ls.kv('decision', 'silent', vc=ls.LB)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            return None
        if scope.is_cancelled():
            return None
        return result.final_content
