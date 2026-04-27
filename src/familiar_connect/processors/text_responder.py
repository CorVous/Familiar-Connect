"""Text reply orchestrator.

Consumes ``discord.text`` events and produces an LLM reply, posted via
the injected ``send_text`` callback. Mirrors :class:`VoiceResponder`
but skips TTS — text channels render the assistant string directly.

Owns user-turn writes for ``discord.text`` (single-writer per channel
keeps read-after-write consistency for ``RecentHistoryLayer`` in the
same task — a separate :class:`HistoryWriter` task would race with
the responder's ``assemble`` call). Cancellation flows through
:class:`TurnRouter` so future barge-in (e.g. user sends a follow-up
mid-stream) cancels in-flight LLM work.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.identity import Author
from familiar_connect.llm import Message
from familiar_connect.silence import SilentDetector

# LLM ping vocabulary: ``[@DisplayName]`` markers in the model's
# output. Symmetric with the form ``RecentHistoryLayer`` rewrites
# inbound ``<@USER_ID>`` mentions into.
_PING_MARKER_RE = re.compile(r"\[@([^\]\n]+)\]")


def _rewrite_pings(
    content: str,
    label_to_key: dict[str, str],
) -> tuple[str, tuple[int, ...]]:
    """Rewrite ``[@DisplayName]`` markers and collect resolved user ids.

    Known labels become Discord-native ``<@user_id>`` mentions and
    contribute to the returned tuple (passed to ``AllowedMentions``).
    Unknown labels degrade to plain ``@DisplayName`` text — no ping,
    no error. Non-Discord canonical keys also degrade (the bot only
    sends to Discord today; future platform support would extend
    this).
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


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.bus.envelope import Event, TurnScope
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.context.assembler import Assembler
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.text_responder")


class TextResponder:
    """Stream LLM replies for ``discord.text`` events; post via ``send_text``."""

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
        history_store: HistoryStore,
        router: TurnRouter,
        familiar_id: str,
    ) -> None:
        self._assembler = assembler
        self._llm = llm_client
        self._send_text = send_text
        self._history = history_store
        self._router = router
        self._familiar_id = familiar_id
        # in-process dedup; bus does not republish today, but cheap insurance
        self._seen: set[str] = set()

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
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
        if not isinstance(channel_id, int) or not content:
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

        self._seen.add(event.event_id)
        scope = self._router.begin_turn(
            session_id=event.session_id, turn_id=event.turn_id
        )

        # Refresh canonical identity rows for everyone we know about in
        # this turn. Soft annotation: the accounts table is a "what we
        # most recently saw" cache, not source of truth.
        if author is not None:
            self._history.upsert_account(author)
            if guild_id is not None and author.guild_nick is not None:
                self._history.upsert_guild_nick(
                    canonical_key=author.canonical_key,
                    guild_id=guild_id,
                    nick=author.guild_nick,
                )
        for m in mentions:
            self._history.upsert_account(m)
            if guild_id is not None and m.guild_nick is not None:
                self._history.upsert_guild_nick(
                    canonical_key=m.canonical_key,
                    guild_id=guild_id,
                    nick=m.guild_nick,
                )

        # Persist user turn *before* streaming so RecentHistoryLayer
        # in the same task sees it. Mirrors VoiceResponder.
        user_turn = self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="user",
            content=content,
            author=author,
            guild_id=guild_id,
            platform_message_id=message_id,
            reply_to_message_id=reply_to_message_id,
        )
        if mentions:
            self._history.record_mentions(
                turn_id=user_turn.id,
                canonical_keys=[m.canonical_key for m in mentions],
            )
        # seed retrieval before assembly so RagContextLayer sees the cue
        self._assembler.set_rag_cue(content)

        # Build ping vocabulary for this channel/guild before assembly.
        # ``label_to_key`` is consumed *after* the LLM responds to
        # rewrite ``[@X]`` markers; the addendum string is appended to
        # the system prompt so the LLM knows which names are routable.
        ping_addendum, label_to_key = self._build_ping_vocabulary(
            channel_id=channel_id, guild_id=guild_id
        )

        reply = await self._stream_reply(
            scope,
            channel_id=channel_id,
            guild_id=guild_id,
            ping_addendum=ping_addendum,
        )
        if reply is None or scope.is_cancelled():
            return
        # Discord rejects empty / whitespace-only messages (HTTP 400,
        # error code 50006). An empty reply usually means the LLM
        # stream emitted no deltas — bad model name, content filter,
        # or upstream error frame the parser silently dropped.
        if not reply.strip():
            _logger.warning(
                f"{ls.tag('Text', ls.Y)} "
                f"{ls.kv('skip', 'empty_reply', vc=ls.LY)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            return

        rewritten, mention_user_ids = _rewrite_pings(reply, label_to_key)

        try:
            sent_message_id = await self._send_text(
                channel_id, rewritten, message_id, mention_user_ids
            )
        except Exception as exc:  # noqa: BLE001 — surface but don't crash loop
            _logger.warning(
                f"{ls.tag('Text', ls.R)} {ls.kv('send_error', repr(exc), vc=ls.R)}"
            )
            return
        _logger.info(
            f"{ls.tag('💬 Text', ls.G)} "
            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
            f"{ls.kv('chars', str(len(rewritten)), vc=ls.LW)} "
            f"{ls.kv('text', ls.trunc(rewritten, 200), vc=ls.LW)}"
        )

        if scope.is_cancelled():
            return

        self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="assistant",
            content=rewritten,
            author=None,
            guild_id=guild_id,
            platform_message_id=sent_message_id,
            reply_to_message_id=message_id,
        )
        self._router.end_turn(scope)

    def _build_ping_vocabulary(
        self,
        *,
        channel_id: int,
        guild_id: int | None,
    ) -> tuple[str, dict[str, str]]:
        """Return ``(system_prompt_addendum, label→canonical_key map)``.

        Pulls recent distinct authors and resolves their labels via
        :meth:`HistoryStore.resolve_label` so the names match what
        ``RecentHistoryLayer`` puts in front of the model. Ambiguous
        labels (two participants with the same nick) keep first-write;
        a warning is logged so the operator can investigate.
        """
        authors = self._history.recent_distinct_authors(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            limit=20,
        )
        label_to_key: dict[str, str] = {}
        lines: list[str] = []
        for a in authors:
            label = self._history.resolve_label(
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
            lines.append(f"- [@{label}] = {a.canonical_key}")
        if not lines:
            return "", {}
        addendum = (
            "## Ping vocabulary\n\n"
            "To deliberately ping someone, write `[@DisplayName]` "
            "where DisplayName is one of the names below. The bot "
            "rewrites it to a Discord mention. Names not on this "
            "list render as plain text without pinging.\n\n" + "\n".join(lines)
        )
        return addendum, label_to_key

    async def _stream_reply(
        self,
        scope: TurnScope,
        *,
        channel_id: int,
        guild_id: int | None = None,
        ping_addendum: str = "",
    ) -> str | None:
        ctx = AssemblyContext(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            viewer_mode="text",
            guild_id=guild_id,
        )
        prompt = await self._assembler.assemble(ctx)
        system = prompt.system_prompt
        if ping_addendum:
            system = f"{system}\n\n{ping_addendum}" if system else ping_addendum
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.extend(prompt.recent_history)

        accumulated: list[str] = []
        silent = SilentDetector()
        try:
            async for delta in self._llm.chat_stream(messages):
                if scope.is_cancelled():
                    return None
                accumulated.append(delta)
                if silent.feed(delta) is True:
                    # Mirror cancellation: no send, no assistant turn.
                    # The 'decision=silent' log replaces the would-be
                    # empty-reply warning (which targets bad model /
                    # content-filter cases, not deliberate silence).
                    _logger.info(
                        f"{ls.tag('💤 Text', ls.B)} "
                        f"{ls.kv('decision', 'silent', vc=ls.LB)} "
                        f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                    )
                    return None
        except Exception as exc:  # noqa: BLE001 — stream errors shouldn't crash loop
            _logger.warning(
                f"{ls.tag('Text', ls.R)} "
                f"{ls.kv('llm_stream_error', repr(exc), vc=ls.R)}"
            )
            return None
        return "".join(accumulated)
