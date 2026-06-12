"""Activity engine — global absence state machine.

idle → active → returning → idle. Active row persisted in the
activities table (restart-safe); :meth:`ActivityEngine.start` reloads
it and re-arms the return timer (past-due returns fire at now+floor —
never inline at boot). Start is deferred: ``start_activity`` tool
calls :meth:`defer_start`, responder applies it at :meth:`end_turn`
(FocusManager precedent).

While out: gate suppresses non-pings; a real @ping at the focused
channel on a reachable type earns one judgment turn per author; a
real reply there cuts the activity short
(:meth:`notify_reply_sent`). Her own alarms pierce any absence.
Return flow generates experience text on the background LLM tier,
commits the return (``finish_activity``), then best-effort: writes a
mechanical event-fact, persists the experience as a marked assistant
turn (``[returned from <label>]`` display prefix; ``turns.mode`` tag
``activity_return`` keys the fact-extractor skip), archives long
absences, promotes staged turns since departure, and wakes the model
only with cause (missed pings).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from zoneinfo import ZoneInfo

from familiar_connect import log_style as ls
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.history.store import ActivityRecord
from familiar_connect.identity import Author, format_turn_for_transcript
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from familiar_connect.activities.config import ActivitiesConfig, ActivityType
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)

# return-turn display prefix (history/RAG rendering only)
RETURN_TURN_MARKER_PREFIX = "[returned from "
# ``turns.mode`` tag on return turns — fact-extractor skip keys on it
ACTIVITY_RETURN_MODE = "activity_return"

# boot recovery: past-due return fires at now+floor, never inline —
# bus consumers + Discord login must exist before the wake publishes
_PAST_DUE_RETURN_FLOOR_S = 20.0

# author display name cap in the judgment state line
_STATE_LINE_NAME_LIMIT = 40

# missed-ping wake assembly caps
_SCAN_LIMIT = 500  # turns scanned since departure
_MAX_EXCERPTS = 3  # newest pings that get turns_around excerpts
_EXCERPT_SPAN = 2  # turns each side of the ping anchor (~5 total)
_VISIBLE_TAIL = 10  # pings within last-few visible turns skip excerpts

# model-facing idle-nudge wake content. persists as a synthetic user
# turn (AlarmWaker shape) so the model sees it in recent history; the
# model decides via start_activity — the nudge never starts anything
_NUDGE_CONTENT = (
    "[idle: the channel has been quiet for a while. If nothing needs "
    "your attention, you could head out and do something — "
    "start_activity. Nobody is around: call silent() with it to slip "
    "away without posting a goodbye. Staying is fine too; your call.]"
)

_PROVENANCE_RAIL = (
    "Write a short first-person account (2-4 sentences) of the "
    "experience. Experiences are about places, things, and yourself — "
    "NEVER invent claims, conversations, or encounters involving real "
    "people. Plain prose, no quotation marks, no preamble."
)


class FocusLike(Protocol):
    """Subset of FocusManager the engine needs."""

    def get_focus(self, modality: str) -> int | None: ...


class GateAction(Enum):
    """What TextResponder should do with an inbound text event."""

    NORMAL = "normal"
    SUPPRESS = "suppress"
    JUDGMENT = "judgment"


@dataclass(frozen=True)
class GateDecision:
    """Gate verdict; ``state_line`` set only for JUDGMENT."""

    action: GateAction
    state_line: str | None = None


@dataclass(frozen=True)
class _StagedStart:
    """Deferred start — applied at end_turn."""

    activity_type: ActivityType
    note: str | None
    duration_minutes: int


class ActivityEngine:
    """Per-familiar absence controller — DB-backed, asyncio-driven."""

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        config: ActivitiesConfig,
        llm_clients: Mapping[str, LLMClient],
        bus: EventBus,
        focus_manager: FocusLike,
        presence_cb: Callable[[str, str | None], Awaitable[None] | None],
        familiar_id: str,
        display_tz: str,
        bot_user_id: Callable[[], int | None],
        voice_active_fn: Callable[[], bool] = lambda: False,
        now_fn: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        rng: random.Random | None = None,
        nudge_tick_seconds: float = 60.0,
    ) -> None:
        self._store = store
        self._config = config
        self._llm_clients = llm_clients
        self._bus = bus
        self._focus = focus_manager
        self._presence_cb = presence_cb
        self._familiar_id = familiar_id
        self._tz = ZoneInfo(display_tz)
        # late-bound: run.py wires before Discord login fills the real id
        self._bot_user_id_fn = bot_user_id
        self._voice_active_fn = voice_active_fn
        self._nudge_tick_seconds = nudge_tick_seconds
        self._now = now_fn
        self._rng = rng if rng is not None else random.Random()  # noqa: S311
        # state machine
        self._active: ActivityRecord | None = None
        self._staged: _StagedStart | None = None
        self._return_task: asyncio.Task[None] | None = None
        self._returning = False
        # departure bookkeeping for archive watermark + ping scan
        self._departure_channel_id: int | None = None
        self._departure_turn_id: int | None = None
        # live-gated ping turn ids — see note_missed_ping
        self._missed_ping_turn_ids: set[int] = set()
        # S2 latch: author keys already judged this absence
        self._judged_author_keys: set[str] = set()
        # idle-nudge state (FocusManager.should_wake shape)
        self._last_traffic: datetime = now_fn()
        self._last_nudge: datetime | None = None
        self._last_return: datetime | None = None
        self._nudge_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Reload active row; re-arm return timer (past-due ⇒ now+floor).

        Also arms the idle-nudge loop (period
        ``nudge_tick_seconds``) — runs for the engine's lifetime.
        """
        if self._nudge_task is not None:
            return
        self._nudge_task = asyncio.create_task(
            self._nudge_loop(),
            name=f"activity-nudge-{self._familiar_id}",
        )
        row = await self._store.active_activity(familiar_id=self._familiar_id)
        if row is None:
            return
        self._active = row
        self._departure_channel_id = self._focus.get_focus("text")
        self._departure_turn_id = await self._recover_departure_turn_id(row)
        _logger.info(
            f"{ls.tag('Activity', ls.G)} reloaded "
            f"{ls.kv('label', row.label, vc=ls.G)} "
            f"{ls.kv('planned_return', row.planned_return_at.isoformat(), vc=ls.LW)}"
        )
        if row.planned_return_at <= self._now():
            # never inline at boot — floor delay lets the stack come up
            self._arm_return_timer(
                self._now() + timedelta(seconds=_PAST_DUE_RETURN_FLOOR_S)
            )
        else:
            self._arm_return_timer(row.planned_return_at)

    async def stop(self) -> None:
        """Cancel return timer + nudge loop. Doesn't modify DB — restart-safe."""
        await self._cancel_return_timer()
        task = self._nudge_task
        self._nudge_task = None
        await _cancel_task(task)

    async def _cancel_return_timer(self) -> None:
        """Cancel pending return task only; nudge loop stays armed."""
        task = self._return_task
        self._return_task = None
        await _cancel_task(task)

    @property
    def return_timer_armed(self) -> bool:
        return self._return_task is not None and not self._return_task.done()

    @property
    def nudge_loop_armed(self) -> bool:
        return self._nudge_task is not None and not self._nudge_task.done()

    @property
    def active(self) -> ActivityRecord | None:
        """Current activity row, or None when idle."""
        return self._active

    @property
    def catalog(self) -> tuple[ActivityType, ...]:
        """Configured activity types (read-only; tool schema builds from this)."""
        return self._config.catalog

    # ------------------------------------------------------------------
    # start path — defer at tool call, apply at end_turn
    # ------------------------------------------------------------------

    def defer_start(self, type_id: str, note: str | None = None) -> dict[str, Any]:
        """Stage an activity start; returns ack/error dict for the model."""
        if self._voice_active_fn():
            return {"error": "can't head out while in a voice channel"}
        if self._active is not None or self._staged is not None:
            label = (
                self._active.label if self._active is not None else "another activity"
            )
            return {"error": f"already out ({label}) — finish that first"}
        activity_type = self._type_for(type_id)
        if activity_type is None:
            valid = ", ".join(t.id for t in self._config.catalog)
            return {"error": f"unknown activity type {type_id!r}; valid: {valid}"}
        lo, hi = activity_type.duration_minutes
        duration = self._rng.randint(lo, hi)
        self._staged = _StagedStart(
            activity_type=activity_type,
            note=note,
            duration_minutes=duration,
        )
        return {
            "ack": "ok",
            "label": activity_type.label,
            "duration_minutes": duration,
            "note": (
                f"you'll head out on the {activity_type.label} after this "
                f"reply, back in roughly {duration} minutes"
            ),
        }

    async def end_turn(self) -> None:
        """Apply staged start (FocusManager deferral pattern).

        Never raises into the responder turn — one unguarded await
        here killed the run TaskGroup in a live incident.
        """
        staged = self._staged
        if staged is None:
            return
        self._staged = None
        try:
            now = self._now()
            planned_return = now + timedelta(minutes=staged.duration_minutes)
            activity_id = await self._store.create_activity(
                familiar_id=self._familiar_id,
                type_id=staged.activity_type.id,
                label=staged.activity_type.label,
                started_at=now,
                planned_return_at=planned_return,
                note=staged.note,
            )
            # all fields in hand — no re-fetch round trip
            self._active = ActivityRecord(
                id=activity_id,
                familiar_id=self._familiar_id,
                type_id=staged.activity_type.id,
                label=staged.activity_type.label,
                started_at=now,
                planned_return_at=planned_return,
                note=staged.note,
                status=None,
                actual_return_at=None,
                experience_text=None,
            )
            self._missed_ping_turn_ids.clear()  # fresh absence, no stale ids
            self._judged_author_keys.clear()  # fresh absence, fresh latch
            self._departure_channel_id = self._focus.get_focus("text")
            # global: absence is from the whole screen, archive breaks all
            # channels at this point (turn ids globally monotonic)
            self._departure_turn_id = await self._store.latest_id(
                familiar_id=self._familiar_id
            )
            await self._set_presence("idle", staged.activity_type.label)
            self._arm_return_timer(planned_return)
            _logger.info(
                f"{ls.tag('🚶 Activity', ls.G)} departed "
                f"{ls.kv('label', staged.activity_type.label, vc=ls.G)} "
                f"{ls.kv('duration_min', str(staged.duration_minutes), vc=ls.LW)} "
                f"{ls.kv('activity_id', str(activity_id), vc=ls.LW)}"
            )
        except Exception as exc:  # noqa: BLE001 — must not kill responder turn
            _logger.error(
                f"{ls.tag('Activity', ls.G)} deferred start failed "
                f"{ls.kv('label', staged.activity_type.label, vc=ls.LW)} "
                f"{ls.kv('error', ls.trunc(str(exc), 120), vc=ls.LW)}"
            )
            # coherent state: row committed ⇒ she left — make sure the
            # timer still brings her back; row missing ⇒ she never left
            if self._active is not None and not self.return_timer_armed:
                self._arm_return_timer(self._active.planned_return_at)

    # ------------------------------------------------------------------
    # gate — called by TextResponder before assembly
    # ------------------------------------------------------------------

    def gate(self, payload: dict[str, Any]) -> GateDecision:
        """Decide handling of inbound text event while (not) absent."""
        # her own alarm pierces any absence (AlarmWaker marker)
        if payload.get("alarm") is True:
            return GateDecision(action=GateAction.NORMAL)
        active = self._active
        # mid-return corpse state must not suppress the return wake
        if active is None or self._returning:
            return GateDecision(action=GateAction.NORMAL)
        content = payload.get("content") or ""
        # prefer ingest-computed flag (covers reply-pings, excludes
        # roles/@everyone); fall back to raw-content scan for payloads
        # without it (synthetic events, tests, voice)
        flag = payload.get("pings_bot")
        is_ping = flag if isinstance(flag, bool) else self._is_ping(content)
        if not is_ping:
            return GateDecision(action=GateAction.SUPPRESS)
        activity_type = self._type_for(active.type_id)
        if activity_type is None or not activity_type.reachable:
            return GateDecision(action=GateAction.SUPPRESS)
        # judgment only at the focused channel — unfocused pings get
        # suppressed+noted and surface in the return wake instead
        if payload.get("channel_id") != self._focus.get_focus("text"):
            return GateDecision(action=GateAction.SUPPRESS)
        raw_author = payload.get("author")
        author = raw_author if isinstance(raw_author, Author) else None
        author_key = author.canonical_key if author is not None else "someone"
        # one judgment per author per absence — repeats get noted only
        if author_key in self._judged_author_keys:
            return GateDecision(action=GateAction.SUPPRESS)
        self._judged_author_keys.add(author_key)
        elapsed_min = max(
            0, int((self._now() - active.started_at).total_seconds() // 60)
        )
        name = ls.trunc(_author_label(author), _STATE_LINE_NAME_LIMIT)
        # "do not call start_activity" clause: eval finding — stay-out
        # intent misroutes to start_activity + meta narration
        state_line = (
            f"You are {elapsed_min} min into {active.label} — {name} pinged "
            f"you. Replying means heading back; silent() means you stay out. "
            f"You are already out — do not call start_activity."
        )
        return GateDecision(action=GateAction.JUDGMENT, state_line=state_line)

    async def notify_reply_sent(self) -> None:
        """Judgment turn produced a real reply ⇒ cut-short return.

        Never raises into the responder turn.
        """
        try:
            if self._active is None or self._returning:
                return
            await self._cancel_return_timer()
            await self._run_return(status="cut_short")
        except Exception as exc:  # noqa: BLE001 — must not kill responder turn
            _logger.error(
                f"{ls.tag('Activity', ls.G)} cut-short return failed "
                f"{ls.kv('error', ls.trunc(str(exc), 120), vc=ls.LW)}"
            )

    # ------------------------------------------------------------------
    # idle nudge (FocusManager.should_wake shape)
    # ------------------------------------------------------------------

    def should_nudge(self, now: datetime) -> bool:
        """Idle-nudge eligibility — quiet channel, gap elapsed, in hours."""
        if self._active is not None or self._staged is not None or self._returning:
            return False
        idle = timedelta(minutes=self._config.idle_nudge_minutes)
        if self._last_nudge is not None and (now - self._last_nudge) < idle:
            return False
        if (now - self._last_traffic) < idle:
            return False
        gap = timedelta(minutes=self._config.min_gap_minutes)
        if self._last_return is not None and (now - self._last_return) < gap:
            return False
        return self._in_active_hours(now)

    def mark_nudge_pending(self) -> None:
        """Record nudge timestamp to start debounce window."""
        self._last_nudge = self._now()

    def note_traffic(self) -> None:
        """Record subscribed-channel traffic; resets the quiet clock."""
        self._last_traffic = self._now()

    def note_missed_ping(self, turn_id: int) -> None:
        """Record a live-gated ping for the at-return wake.

        Responder calls this on suppressed turns whose payload carried
        ``pings_bot`` — captures what the content scan can't see:
        cross-channel pings and reply-pings (no ``<@id>`` string in
        stored content). In-memory only; restart loses the set and
        the content-based focused-channel scan in :meth:`_missed_pings`
        is the fallback. Cleared at activity start, merged (deduped)
        with the scan at return time.
        """
        self._missed_ping_turn_ids.add(turn_id)

    async def _nudge_loop(self) -> None:
        """Periodic eligibility check; publishes idle-nudge wake."""
        while True:
            await asyncio.sleep(self._nudge_tick_seconds)
            if self.should_nudge(self._now()):
                await self._publish_nudge()

    async def _publish_nudge(self) -> None:
        """Synthetic wake into focused text channel (AlarmWaker shape)."""
        channel_id = self._focus.get_focus("text")
        if channel_id is None:
            return
        self.mark_nudge_pending()
        await self._publish_synthetic(channel_id, _NUDGE_CONTENT, "activity-nudge")
        _logger.info(
            f"{ls.tag('🚶 Activity', ls.G)} nudge "
            f"{ls.kv('channel', str(channel_id), vc=ls.LW)}"
        )

    def _in_active_hours(self, now: datetime) -> bool:
        hours = self._config.active_hours
        if hours is None:
            return True
        start, end = hours
        local = now.astimezone(self._tz).time()
        if start < end:
            return start <= local < end
        return local >= start or local < end  # wraps midnight

    # ------------------------------------------------------------------
    # returning — timer + cut-short paths
    # ------------------------------------------------------------------

    def _arm_return_timer(self, planned_return_at: datetime) -> None:
        self._return_task = asyncio.create_task(
            self._sleep_then_return(planned_return_at),
            name=f"activity-return-{self._familiar_id}",
        )

    async def _sleep_then_return(self, planned_return_at: datetime) -> None:
        delay = (planned_return_at - self._now()).total_seconds()
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await self._run_return(status="completed")
        except Exception as exc:  # noqa: BLE001 — no orphan task exceptions
            _logger.error(
                f"{ls.tag('Activity', ls.G)} timed return failed "
                f"{ls.kv('error', ls.trunc(str(exc), 120), vc=ls.LW)}"
            )

    async def _run_return(self, *, status: str) -> None:
        """Experience, commit, then best-effort fact/turn/archive/wake.

        ``finish_activity`` is the commit — after it she is back even
        if everything else fails (no replay on restart, no stuck
        absent). Each later step is individually guarded; one failing
        must not kill the rest. All state ``gate()`` consults clears
        BEFORE the wake publish so the responder consuming the wake
        sees NORMAL, never a corpse absence.
        """
        active = self._active
        if active is None or self._returning:
            return
        self._returning = True
        now = self._now()
        try:
            activity_type = self._type_for(active.type_id)
            cut_short = status == "cut_short"
            # internally guarded — always yields fallback text
            experience = await self._generate_experience(
                active, activity_type, cut_short=cut_short
            )
            # COMMIT — she is back after this line
            await self._store.finish_activity(
                activity_id=active.id,
                status=status,
                actual_return_at=now,
                experience_text=experience,
            )
            channel_id = self._departure_channel_id
            departure_turn_id = self._departure_turn_id
            try:
                # mechanical event-fact — no LLM
                await self._store.append_fact(
                    familiar_id=self._familiar_id,
                    channel_id=channel_id,
                    text=self._event_fact_text(active),
                    source_turn_ids=(),
                    valid_from=active.started_at,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort step
                self._log_return_step_failed("event_fact", exc)
            if channel_id is not None:
                try:
                    # experience as marked assistant turn — her own
                    # narration; mode tag keys the fact-extractor skip
                    await self._store.append_turn(
                        familiar_id=self._familiar_id,
                        channel_id=channel_id,
                        role="assistant",
                        content=(
                            f"{RETURN_TURN_MARKER_PREFIX}{active.label}] {experience}"
                        ),
                        mode=ACTIVITY_RETURN_MODE,
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort step
                    self._log_return_step_failed("return_turn", exc)
            try:
                # global archive: absence is from the whole screen — one
                # departure point breaks every channel's window
                absence = now - active.started_at
                threshold = timedelta(minutes=self._config.archive_after_minutes)
                if absence >= threshold and departure_turn_id is not None:
                    await self._store.set_archive_watermark_all(
                        familiar_id=self._familiar_id,
                        turn_id=departure_turn_id,
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort step
                self._log_return_step_failed("archive_watermark", exc)
            if departure_turn_id is not None:
                try:
                    # she reads the screen when she gets back — staged
                    # turns from the absence join the visible window
                    promoted = await self._store.promote_staged_turns_since(
                        familiar_id=self._familiar_id,
                        after_turn_id=departure_turn_id,
                    )
                    if promoted:
                        _logger.info(
                            f"{ls.tag('Activity', ls.G)} promoted staged "
                            f"{ls.kv('turns', str(promoted), vc=ls.LW)}"
                        )
                except Exception as exc:  # noqa: BLE001 — best-effort step
                    self._log_return_step_failed("staged_promotion", exc)
            missed_pings: list[HistoryTurn] = []
            if channel_id is not None:
                try:
                    missed_pings = await self._collect_missed_pings(channel_id)
                except Exception as exc:  # noqa: BLE001 — best-effort step
                    self._log_return_step_failed("missed_ping_scan", exc)
            # C1: clear gate-consulted state BEFORE the wake publish
            self._clear_absence_state(now)
            if missed_pings and channel_id is not None:
                try:
                    await self._publish_wake(
                        channel_id=channel_id,
                        label=active.label,
                        pings=missed_pings,
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort step
                    self._log_return_step_failed("wake_publish", exc)
            await self._set_presence("online", None)  # internally guarded
            _logger.info(
                f"{ls.tag('✨ Activity', ls.G)} returned "
                f"{ls.kv('label', active.label, vc=ls.G)} "
                f"{ls.kv('status', status, vc=ls.LW)} "
                f"{ls.kv('missed_pings', str(len(missed_pings)), vc=ls.LW)}"
            )
        finally:
            # idempotent — exception paths land here with state cleared
            self._clear_absence_state(now)
            self._returning = False

    def _clear_absence_state(self, now: datetime) -> None:
        """Reset everything ``gate()`` and the nudge loop consult."""
        self._active = None
        self._departure_channel_id = None
        self._departure_turn_id = None
        self._return_task = None
        self._last_return = now

    def _log_return_step_failed(self, step: str, exc: Exception) -> None:
        _logger.error(
            f"{ls.tag('Activity', ls.G)} return step failed "
            f"{ls.kv('step', step, vc=ls.LW)} "
            f"{ls.kv('error', ls.trunc(str(exc), 120), vc=ls.LW)}"
        )

    async def _generate_experience(
        self,
        active: ActivityRecord,
        activity_type: ActivityType | None,
        *,
        cut_short: bool,
    ) -> str:
        seed = activity_type.seed if activity_type is not None else active.label
        lines = [f"Activity: {active.label}", f"Seed: {seed}"]
        if active.note:
            lines.append(f"Intent noted before leaving: {active.note}")
        if cut_short:
            lines.append(
                "The activity was cut short — someone pinged and the "
                "return home happened early."
            )
        try:
            reply = await self._llm_clients["background"].chat([
                Message(role="system", content=_PROVENANCE_RAIL),
                Message(role="user", content="\n".join(lines)),
            ])
            text = reply.content if isinstance(reply.content, str) else ""
        except Exception as exc:  # noqa: BLE001 — return flow must finish
            _logger.warning(
                f"{ls.tag('Activity', ls.G)} "
                f"{ls.kv('experience_llm_failed', ls.trunc(str(exc), 120), vc=ls.LY)}"
            )
            text = ""
        return text.strip() or f"Back from the {active.label}."

    def _event_fact_text(self, active: ActivityRecord) -> str:
        local = active.started_at.astimezone(self._tz)
        date_str = f"{local.strftime('%b')} {local.day}"
        # spent-<when>-<label> phrasing composes with phrase labels
        # like tending the shrines / out walking the woods
        return (
            f"{self._familiar_id.capitalize()} spent {date_str} "
            f"{_daypart(local.hour)} {active.label}"
        )

    async def _collect_missed_pings(self, channel_id: int) -> list[HistoryTurn]:
        """Live-noted ping ids merged (deduped) with the content scan.

        Live list covers cross-channel + reply-pings; the scan covers
        ``<@id>`` mentions in the focused channel that survive a
        restart. Consumes (clears) the live list.
        """
        scanned = await self._missed_pings(channel_id)
        noted_ids = self._missed_ping_turn_ids - {t.id for t in scanned}
        self._missed_ping_turn_ids = set()
        if not noted_ids:
            return scanned
        noted = await self._store.turns_by_ids(
            familiar_id=self._familiar_id, ids=noted_ids
        )
        merged = {t.id: t for t in (*scanned, *noted)}
        return [merged[i] for i in sorted(merged)]

    async def _missed_pings(self, channel_id: int) -> list[HistoryTurn]:
        """User turns since departure carrying a real bot mention.

        Content-based scan: stored turns don't persist ``pings_bot``,
        so reply-pings without an ``<@id>`` string aren't recoverable
        here.
        """
        departure_id = self._departure_turn_id
        turns = await self._store.recent(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            limit=_SCAN_LIMIT,
        )
        return [
            t
            for t in turns
            if (departure_id is None or t.id > departure_id)
            and t.role == "user"
            and self._is_ping(t.content)
        ]

    async def _publish_wake(
        self,
        *,
        channel_id: int,
        label: str,
        pings: list[HistoryTurn],
    ) -> None:
        """Synthetic discord.text wake event (AlarmWaker pattern)."""
        content = await self._wake_content(
            channel_id=channel_id, label=label, pings=pings
        )
        await self._publish_synthetic(channel_id, content, "activity-return")

    async def _publish_synthetic(
        self, channel_id: int, content: str, turn_prefix: str
    ) -> None:
        """Synthetic discord.text event with authorless payload."""
        synth_event_id = uuid.uuid4().hex
        await self._bus.publish(
            Event(
                event_id=synth_event_id,
                turn_id=f"{turn_prefix}-{synth_event_id}",
                session_id=str(channel_id),
                parent_event_ids=(),
                topic=TOPIC_DISCORD_TEXT,
                timestamp=self._now(),
                sequence_number=0,
                payload={
                    "familiar_id": self._familiar_id,
                    "channel_id": channel_id,
                    "content": content,
                    "author": None,
                    "guild_id": None,
                    "message_id": None,
                    "reply_to_message_id": None,
                    "mentions": (),
                },
            )
        )

    async def _wake_content(
        self,
        *,
        channel_id: int,
        label: str,
        pings: list[HistoryTurn],
    ) -> str:
        """One-line ping list + capped turns_around excerpts.

        Newest ``_MAX_EXCERPTS`` pings get a ~5-turn excerpt; older
        pings collapse to one-liners; pings already inside the
        last-few visible window skip excerpts.
        """
        visible = await self._store.recent(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            limit=_VISIBLE_TAIL,
        )
        visible_ids = {t.id for t in visible}
        excerpt_ids = {t.id for t in pings[-_MAX_EXCERPTS:]} - visible_ids
        lines = [f"[returned from {label} — missed pings while away]"]
        for ping in pings:
            name = _author_label(ping.author)
            lines.append(f"- {name}: {ls.trunc(ping.content, 160)}")
            if ping.id in excerpt_ids:
                window = await self._store.turns_around(
                    familiar_id=self._familiar_id,
                    # live-noted pings may live in another channel —
                    # anchor the excerpt where the ping happened
                    channel_id=ping.channel_id or channel_id,
                    turn_id=ping.id,
                    before=_EXCERPT_SPAN,
                    after=_EXCERPT_SPAN,
                )
                lines.extend(
                    "    " + format_turn_for_transcript(t.role, t.author, t.content)
                    for t in window
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _type_for(self, type_id: str) -> ActivityType | None:
        for entry in self._config.catalog:
            if entry.id == type_id:
                return entry
        return None

    def _is_ping(self, content: str) -> bool:
        """Real @mention of the bot in raw content; bare name doesn't count.

        Fallback for payloads without ``pings_bot`` and for the
        at-return scan over stored turns (which don't persist the
        flag). Real ``<@id>`` mentions survive in content; reply-pings
        carry no mention string, so the scan can miss those — known
        gap, acceptable for v1.
        """
        bot_id = self._bot_user_id_fn()
        if bot_id is None:
            return False
        return f"<@{bot_id}>" in content or f"<@!{bot_id}>" in content

    async def _set_presence(self, status: str, text: str | None) -> None:
        """Presence is cosmetic — cb failures logged, never raised.

        Live incident 2026-06-12: change_presence on a reconnecting
        websocket raised and the error killed the run TaskGroup.
        """
        try:
            result = self._presence_cb(status, text)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — any cb failure is non-fatal
            _logger.warning(
                f"{ls.tag('Activity', ls.G)} presence update failed "
                f"{ls.kv('status', status, vc=ls.LW)} "
                f"{ls.kv('error', ls.trunc(str(exc), 120), vc=ls.LW)}"
            )

    async def _recover_departure_turn_id(self, row: ActivityRecord) -> int | None:
        """Newest turn id at/before departure — recomputed after restart.

        Global (all channels), matching the global archive semantics.
        """
        return await self._store.latest_id_at_or_before(
            familiar_id=self._familiar_id, ts=row.started_at
        )


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel *task* (if any) and swallow its exit."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _author_label(author: Author | None) -> str:
    """:attr:`Author.label`; "someone" only when author missing."""
    return author.label if author is not None else "someone"


def _daypart(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"
