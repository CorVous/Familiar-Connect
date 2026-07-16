# 11 — twitch + activities — port spec

Source files:
- `src/familiar_connect/twitch.py` — Twitch event types, text formatters, event builders (pure).
- `src/familiar_connect/twitch_watcher.py` — EventSub WebSocket watcher (twitchAPI adapter → `asyncio.Queue`).
- `src/familiar_connect/sources/twitch.py` — queue → bus drain (`TwitchSource`; not in the listed set but it is the queue's only consumer; documented here, file lives in 10's sources package).
- `src/familiar_connect/activities/config.py` — `activities.toml` sidecar loader.
- `src/familiar_connect/activities/engine.py` — `ActivityEngine`, the global-absence state machine.
- `src/familiar_connect/activities/__init__.py` — re-exports.

Docs: `docs/architecture/activities.md` (authoritative narrative; also `sleep.md` for the sleep-cycle half owned by 04).

## Role

Two loosely-related features. (a) Twitch: a thin adapter that converts Twitch EventSub callbacks (follows, subs, gift subs, resubs, cheers, channel-point redemptions, ad breaks) into normalized `TwitchEvent` values on an `asyncio.Queue`, plus a source that drains the queue onto the bus topic `twitch.event`. **The Twitch pipeline is currently dormant**: nothing in `commands/run.py` constructs `TwitchWatcher`/`TwitchSource`, and no processor subscribes to `TOPIC_TWITCH_EVENT` (it appears only in the debug-logger topic list). (b) Activities: the `ActivityEngine` lets the familiar "get up from the screen" — a `start_activity` tool call stages a departure, the responder applies it after the reply ships, inbound messages are gated (suppress / judgment / normal) while she is out, and a return timer drives experience generation, a mechanical event-fact, a marked return turn, archive watermarking, staged-turn promotion, a missed-ping wake, and presence updates. The reserved `sleep` catalog entry rides the same machinery with an engine-owned wall-clock schedule and dream/maintenance passes (passes themselves owned by 04).

## Public API surface

### twitch.py (pure, no IO)

```python
@dataclass TwitchEvent:
    channel: str
    text: str
    priority: Literal["normal", "immediate"]
    timestamp: datetime          # UTC-aware
    viewer: Author | None = None

    def to_message(self) -> Message
        # Message(role="user", content=f"[Twitch] {text}",
        #         name=viewer.openai_name if viewer else "Twitch")

@dataclass TwitchWatcherConfig:
    subscriptions_enabled: bool = True
    cheers_enabled: bool = True
    follows_enabled: bool = True
    ads_enabled: bool = True
    ads_immediate: bool = True
    redemption_names: list[str] = []
```

Formatters (exact output strings — conformance targets, see Data formats):
`format_channel_point_redemption(viewer, redemption_name, user_input=None)`,
`format_subscription(viewer, tier)`, `format_gift_subscription(gifter|None, count, tier)`,
`format_resubscription(viewer, months, tier, message)`, `format_cheer(viewer|None, bits, message)`,
`format_follow(viewer)`, `format_ad_start()`, `format_ad_end()`.

Builders (all keyword-only; each returns `TwitchEvent | None`; `None` = event type disabled by config):
`build_channel_point_event(config, channel, viewer, redemption_name, user_input=None)`,
`build_subscription_event(config, channel, viewer, tier)`,
`build_gift_subscription_event(config, channel, gifter, count, tier)`,
`build_resubscription_event(config, channel, viewer, months, tier, message)`,
`build_cheer_event(config, channel, viewer, bits, message)`,
`build_follow_event(config, channel, viewer)`,
`build_ad_start_event(config, channel)`, `build_ad_end_event(config, channel)`.
Timestamp is `datetime.now(UTC)` at build time. `build_ad_end_event` has no watcher caller (ad end is not an EventSub subscription in v1) but is public and tested.

### twitch_watcher.py

```python
class TwitchWatcher:
    def __init__(self, *, config: TwitchWatcherConfig, broadcaster_id: str,
                 channel: str, moderator_id: str | None = None)
    # sync handlers, one per EventSub data object; return TwitchEvent | None
    def handle_follow(data) / handle_subscription(data) / handle_gift_subscription(data)
    def handle_resubscription(data) / handle_cheer(data)
    def handle_channel_point_redemption(data) / handle_ad_break_begin(data)
    async def register_listeners(eventsub, send: asyncio.Queue[TwitchEvent] | None = None)
    async def run(send: asyncio.Queue[TwitchEvent], eventsub: EventSubWebsocket)
```

The `eventsub` argument is duck-typed against twitchAPI's `EventSubWebsocket` (a seam — tests pass mocks). Handlers are synchronous and pure given the data object; all IO lives in `run()`.

### sources/twitch.py

```python
class TwitchSource:
    name = "twitch"
    def __init__(self, *, bus: EventBus, familiar_id: str, queue: asyncio.Queue[object])
    async def run(self) -> None   # forever: q.get() → bus.publish; cancel stops
```

### activities/config.py

```python
CONTENT_SOURCES = frozenset({"authored"})
SLEEP_TYPE_ID = "sleep"

@dataclass(frozen=True) ActivityType:
    id: str; label: str
    duration_minutes: tuple[int, int] | None = None   # (lo, hi), 0 < lo <= hi
    reachable: bool = True
    content_source: str = "authored"
    seed: str = ""
    active_days: frozenset[int] | None = None          # datetime.weekday() ints, Mon=0
    active_hours: tuple[time, time] | None = None      # may wrap midnight

@dataclass(frozen=True) ActivitiesConfig:
    catalog: tuple[ActivityType, ...] = ()
    archive_after_minutes: int = 45
    idle_nudge_minutes: int = 20
    min_gap_minutes: int = 90
    active_hours: tuple[time, time] | None = None
    enabled -> bool  # == bool(catalog); __bool__ mirrors it

def load_activities_config(path, *, defaults_path=None) -> ActivitiesConfig
    # raises ConfigError on invalid content; missing file(s) ⇒ disabled config
```

### activities/engine.py

```python
RETURN_TURN_MARKER_PREFIX = "[returned from "   # display prefix only
ACTIVITY_RETURN_MODE = "activity_return"        # turns.mode — fact extractor SKIPS
SLEEP_RETURN_MODE = "sleep_return"              # turns.mode — extractor dream-frames

class FocusLike(Protocol):                       # seam; FocusManager satisfies it
    def get_focus(self, modality: str) -> int | None
    @property catch_up_limit -> int

class GateAction(Enum): NORMAL | SUPPRESS | JUDGMENT
@dataclass(frozen=True) GateDecision: action: GateAction; state_line: str | None = None

class ActivityEngine:
    def __init__(self, *, store: AsyncHistoryStore, config: ActivitiesConfig,
                 llm_clients: Mapping[str, LLMClient],   # uses only "background"
                 bus: EventBus, focus_manager: FocusLike,
                 presence_cb: Callable[[str, str | None], Awaitable[None] | None],
                 familiar_id: str, display_tz: str,
                 bot_user_id: Callable[[], int | None],   # late-bound (pre-login None)
                 sleep_window: tuple[time, time] | None = None,
                 sleep_grace_minutes: int = 30,
                 voice_active_fn: Callable[[], bool] = lambda: False,
                 now_fn: Callable[[], datetime] = utcnow,   # injected clock
                 rng: random.Random | None = None,
                 nudge_tick_seconds: float = 60.0,
                 familiar_display_name: str | None = None,  # default familiar_id.title()
                 sleep_passes_enabled: bool = False,
                 seed_dream_path: Path | None = None,
                 sleep_prompts: SleepPromptText | None = None)

    async def start(self)              # idempotent; arms nudge loop + reloads active row
    async def stop(self)               # cancels timers/tasks; DB untouched
    async def resync_presence(self)    # re-issue away presence; idle ⇒ no-op
    @property return_timer_armed -> bool
    @property nudge_loop_armed -> bool
    @property active -> ActivityRecord | None
    @property catalog -> tuple[ActivityType, ...]

    def defer_start(self, type_id, note=None) -> dict     # {"ack","label","duration_minutes","note"} or {"error"}
    async def end_turn(self)                              # applies staged start; NEVER raises
    def gate(self, payload: dict) -> GateDecision         # sync, no IO
    async def notify_reply_sent(self)                     # judgment reply ⇒ cut-short return; NEVER raises
    def should_nudge(self, now) -> bool
    def mark_nudge_pending(self)
    def note_traffic(self)
    def note_missed_ping(self, turn_id: int)
```

Swappable seams: `presence_cb` (sync **or** async callable — checked with `inspect.isawaitable`), `bot_user_id`, `voice_active_fn`, `now_fn`, `rng`, `focus_manager: FocusLike`, `store` (monkeypatch-faulted in tests), `llm_clients["background"]`.

### Consumers' contract (defined elsewhere, but load-bearing here)

- `processors/text_responder.py` (06): calls `note_traffic()` for **every** handled event (even suppressed); calls `gate(payload)` before prompt assembly; on SUPPRESS stores the user turn staged (`consumed=False`), never types/streams/replies, and if `payload["pings_bot"] is True` calls `note_missed_ping(turn.id)`; on JUDGMENT appends `state_line` as a trailing **system** message this turn only; after a real reply on a judgment turn calls `notify_reply_sent()` then `end_turn()`; every reply-shipped, silent, empty-reply and send-error exit path calls `end_turn()` (a stale staged start must never leak into a later turn).
- `tools/start_activity.py` (08): tool enum built from `engine.catalog` at registry-build time; if `engine.active is not None` the handler returns the `SILENT_RESULT` sentinel (already-out call = stay-out intent, no channel narration); otherwise validates args and returns `json.dumps(engine.defer_start(...))`. Registered only in the **text** registry and only when an engine exists; never in voice.
- `processors/fact_extractor.py` (07): imports `ACTIVITY_RETURN_MODE` (skip) and `SLEEP_RETURN_MODE` (process dream-framed).
- `bot.py` (10): `build_activity_presence_cb(handle)` — no-op until `bot.is_ready()`; `"idle"|"dnd"` → `change_presence(status, CustomActivity(name=label or "away"))`; `"online"` → focus-presence resync (or plain online without a FocusManager). `on_ready` calls `engine.resync_presence()` **after** the focus presence sync (covers boot-mid-activity and gateway reconnects).
- `commands/run.py` (10): `_build_activity_engine` loads `familiar.root/activities.toml` merged over `data/familiars/_default/activities.toml`; returns `None` when disabled (feature entirely unwired ⇒ byte-for-byte legacy behavior). Prod wiring: `sleep_window`/`sleep_grace_minutes` from character.toml `[sleep]`, `bot_user_id=lambda: familiar.bot_user_id`, `voice_active_fn=lambda: bool(handle.voice_runtime)`, `sleep_passes_enabled=True`, `seed_dream_path=familiar.root/"seed_dream.md"`, prompts via `SleepPromptText.from_config`. `engine.start()` runs pre-Discord-login; `engine.stop()` in shutdown finally with exceptions suppressed.

## Behaviors & invariants

### Twitch event conversion

1. Tier mapping: Twitch tier strings `"1000"/"2000"/"3000"` → ints 1/2/3; **any unknown string → 1**.
2. `handle_subscription` returns `None` when `data.is_gift` is true (gift subs arrive via the dedicated gift handler) — even when subscriptions are enabled.
3. `handle_resubscription` months = `data.cumulative_months` unless it is `None`, then `data.duration_months` (always present). Message text is `data.message.text` (nested object).
4. Anonymous handling: gift-sub gifter and cheer viewer become `None` when `data.is_anonymous`; formatters then use "An anonymous gifter"/"An anonymous cheerer". `event.viewer` is `None` in those cases and for ad events.
5. Channel-point redemptions are an allow-list: the builder returns `None` unless `redemption_name` is (exact string) in `config.redemption_names`. Empty `user_input` is coerced to `None` (`data.user_input or None`), which drops the "and says:" clause.
6. Event priority: everything is `"normal"` except ad start/end which are `"immediate"` when `config.ads_immediate` else `"normal"`.
7. `Author` construction: `Author.from_twitch(user_id=str(user_id or ""), user_login=user_login or "", user_name=user_name or "")` — keyed on `user_id` so repeat viewers resolve to the same identity regardless of display-name changes; Optional fields coerced to `""` (anonymous-capable flows must guard **before** calling).
8. `register_listeners` registration matrix (order matters only for tests asserting call counts): follows → `listen_channel_follow_v2(broadcaster_id, moderator_id, cb)`; subscriptions → `listen_channel_subscribe`, `listen_channel_subscription_gift`, `listen_channel_subscription_message` (all three, broadcaster only); cheers → `listen_channel_cheer`; ads → `listen_channel_ad_break_begin`; redemptions → `listen_channel_points_custom_reward_redemption_add` **iff `redemption_names` non-empty**. `moderator_id` defaults to `broadcaster_id`.
9. Every callback wraps a sync handler; a `None` result (disabled/filtered) sends nothing; otherwise `await queue.put(event)`. `send=None` (test-only) also sends nothing.
10. `run(send, eventsub)`: register listeners → `await eventsub.start()` → sleep forever (`asyncio.Event().wait()`); on cancellation (or any exit) `finally: await eventsub.stop()`. Cancellation is the only shutdown path; there is no retry/reconnect logic in this layer (twitchAPI owns the WS session).
11. `TwitchSource.run`: infinite `queue.get()` → publish; `sequence_number` is a per-source monotonic counter starting at 1; `event_id == turn_id == "twitch-" + uuid4().hex[:12]`; `session_id = f"twitch:{familiar_id}"`; payload `{"familiar_id": ..., "twitch": <TwitchEvent object>}` (the object itself, not serialized). Topic policy is expected UNBOUNDED (volume low, dropping a cheer costly). Stops when the bus publish raises on shutdown or the task is cancelled.
12. **Dormancy invariant**: no production wiring constructs the watcher/source and nothing consumes `twitch.event`. `TwitchEvent.to_message()` has no production caller. Port the pure layer (`twitch.py`) with exact-string conformance; the watcher/source can be a stub crate feature until a consumer exists.

### activities.toml loading/validation

13. Missing target file **and** missing/empty defaults ⇒ disabled config (all defaults, empty catalog, falsy). Knobs-without-catalog parse fine but stay disabled. Target deep-merges **over** defaults (`_deep_merge`, recursive on dict-vs-dict, override wins otherwise; note: `catalog` lists replace wholesale).
14. Present-but-invalid fails loudly with `ConfigError`: malformed TOML (`"failed to parse TOML"` in message); unknown top-level keys (valid: `archive_after_minutes, idle_nudge_minutes, min_gap_minutes, active_hours, catalog`) and unknown entry keys are rejected with sorted-key messages; knobs must be positive ints and **not bool** (Python `bool` is an `int` — replicate the explicit bool rejection); `catalog` must be an array of tables.
15. Entry required keys: `id, label, duration_minutes, seed` — except `duration_minutes` is optional **only** when `id == "sleep"`. `id/label/seed` must be non-empty (after strip) strings. Duplicate ids rejected. `duration_minutes` must be a 2-list of non-bool ints with `0 < lo <= hi` (`lo == hi` legal). `reachable` must be bool (default true). `content_source`: `"adapter"` rejected with a dedicated "reserved" message; anything not in `CONTENT_SOURCES` rejected; default `"authored"`.
16. `active_days`: non-empty list of tokens `mon..sun` → `frozenset` of weekday ints Mon=0..Sun=6; unknown/non-string token or non-list/empty rejected. `active_hours` (both top-level and per-entry) parsed by `parse_hhmm_range`: strict `"HH:MM-HH:MM"` (two-digit fields), may wrap midnight, `start == end` rejected; per-entry errors are prefixed with the entry id.
17. The shipped `_default/activities.toml` is fully commented out (parses to disabled) — pinned by `test_docs`-adjacent test `TestDefaultSkeleton.test_ships_disabled`.

### Engine — start path

18. `defer_start` rejection order (first match wins, each returns `{"error": ...}` without mutating state): (a) `voice_active_fn()` true → "can't head out while in a voice channel"; (b) already active or already staged → `"already out (<label>) — finish that first"` (label falls back to "another activity" when only staged); (c) unknown `type_id` → error listing valid ids; (d) schedule violation (see 20); (e) sleep-only: early-bed guard (see 27); (f) scheduled-entry roll clamp room < lo (see 21).
19. Success stages `_StagedStart(activity_type, note, duration_minutes)` and returns `{"ack": "ok", "label", "duration_minutes", "note": "you'll head out on the <label> after this reply, back in roughly <d> minutes"}`. Duration: window-scheduled (sleep) ⇒ `max(1, (window_end − now) // 60s)` minutes; otherwise `rng.randint(lo, hi)` with `hi` possibly clamped (21). Nothing touches the DB or presence until `end_turn` (pinned).
20. Per-entry availability gate: entries with neither `active_days` nor `active_hours` are always available. Otherwise localize `now` to `display_tz`; require `weekday() in active_days` (if set) **and** local time in `active_hours` (if set; half-open `[start, end)`, wrap-aware: `start > end` means `local >= start or local < end`). Violation message: `"<label> is only available <Mon Tue ...>, <HH:MM-HH:MM>"` (days part and/or hours part as configured). **KNOWN LIMITATION (preserve or document)**: the weekday is taken from the calendar day of `now`, so a midnight-wrapping hours window's post-midnight tail is attributed to the next day's weekday.
21. Scheduled-entry duration clamp (entries with `active_hours` and a rolled duration): `(_, win_end) = window_occurrence(now, hours)`; `room = ((win_end + 15min grace) − now) // 60s`; if `room < lo` → error `"not enough time before the <label> window closes — head out earlier"` (strict `<`; `room == lo` stages with `randint(lo, lo)`); else `hi = min(hi, room)`. Days-only entries are never clamped. Midnight-wrap windows clamp against the **next-day** end (occurrence containing now). Note: the bound is computed at defer-time but the return is computed at end_turn-time, so the real return can overrun by grace + the (seconds-scale) deferral gap — accepted.
22. `end_turn` (applies the staged start): recomputes `now`; window-scheduled ⇒ `planned_return = occurrence end`; else `now + duration`. Calls `_begin_activity`. **Never raises into the responder turn** (live-incident invariant): any exception is logged, and if the DB row was already committed (`self._active` set) but the return timer isn't armed, the timer is re-armed at `planned_return_at` so she still comes back; if the row never committed, state is left clean (staged consumed, next `defer_start` succeeds).
23. `_begin_activity` sequence: `store.create_activity(...)` → build in-memory `ActivityRecord` (no re-fetch; `status/actual_return_at/experience_text` None) → if sleep, spawn passes task (33) → clear `_missed_ping_turn_ids` and `_judged_author_keys` (fresh absence) → `_departure_channel_id = focus.get_focus("text")` → `_departure_turn_id = store.latest_id(familiar_id)` (global, all channels — turn ids are globally monotonic) → presence (`idle` if reachable, `dnd` if not/unknown, with catalog label) → arm return timer.

### Engine — gate

24. `gate(payload)` decision tree, in order: `payload.get("alarm") is True` → NORMAL (her own alarms pierce any absence; replying to one does not cut it short). Idle (`_active is None`) or `_returning` → NORMAL (the mid-return "corpse state" must never suppress the return wake — C1). Ping detection: `payload["pings_bot"]` is **authoritative when a bool** (True = ping even without a mention string; False = not a ping even with one); missing/non-bool falls back to a raw scan of `payload["content"]` (empty-string default) for `<@{bot_id}>` or `<@!{bot_id}>`; `bot_user_id()` returning None (pre-ready) makes the scan always false. Non-ping → SUPPRESS. Type unknown in catalog **or** `reachable == False` → SUPPRESS. `payload["channel_id"] != focus.get_focus("text")` (including focus None) → SUPPRESS (unfocused pings surface in the return wake instead). Author latch: key = `author.canonical_key` if payload author is an `Author` instance else `"someone"` (all authorless pings share one slot); key already judged this absence → SUPPRESS; otherwise the key is added (latch cleared only at the next activity start) and → JUDGMENT.
25. Judgment `state_line` (exact template): `"You are {elapsed} min into {label} — {name} pinged you. Replying means heading back; silent() means you stay out. You are already out — do not call start_activity."` where `elapsed = max(0, (now − started_at) // 60s)` and `name` is `author.label` (or "someone") truncated to 40 chars (`ls.trunc` — head + ellipsis behavior of the log-style helper). The trailing "do not call start_activity" clause is an eval-finding fix and is pinned by tests.
26. `notify_reply_sent`: no-op when idle or `_returning`; else cancel return timer and run the return with `status="cut_short"`. Never raises (guarded like `end_turn`).

### Engine — sleep schedule (tick loop)

27. Sleep is window-scheduled: needs **both** the catalog `sleep` entry and a ctor `sleep_window` — either missing disarms the schedule (`defer_start("sleep")` then follows the normal rolled path only if the entry has a duration; with `duration_minutes=None` and no window it errors "has no duration"). `_window_occurrence(now, window)`: occurrence containing local-now, else the next one; length computed as `(combine(d, end) − combine(d, start)) % 1 day`; candidate starts at local date −1/0/+1 days; first with `start <= local < end or start > local` wins. Early-bed guard in `defer_start`: if `occurrence_start − now > 60 min` → error `"not bedtime — the sleep window starts at {HH:MM local}; head to bed within the hour before it"`; inside the window `start <= now` so the guard passes.
28. `_sleep_schedule_tick(now)` (runs first on every nudge-loop tick, exceptions logged and swallowed so the loop never dies): disarmed or out/staged/returning → return (backstop fires on the first idle tick after she's back, if still in-window); `now < occurrence_start` → return; `_slept_this_window` (latest sleep-typed row with `started_at >= occurrence_start`, **active or finished** — a cut-short sleep blocks re-entry for the night) → return; `now >= start + grace_minutes` → force sleep: `_begin_activity(entry, None, window_end)` directly, no LLM choice, no tool call; else publish the bedtime nudge once per occurrence (debounced by remembering the occurrence start).
29. Wake is **fixed at window end** regardless of start time (both tool-started and force-started sleep).

### Engine — idle nudge

30. `should_nudge(now)` is pure over state: false while active/staged/returning; false within `idle_nudge_minutes` of the last nudge (debounce) or the last traffic; false within `min_gap_minutes` of the last return; else true iff local time is inside top-level `active_hours` (unset = always; wrap-aware). `_last_traffic` initializes to construction time, so a fresh boot needs a full quiet window before the first nudge. `min_gap` gates nudges only — never tool calls.
31. Nudge loop task (`activity-nudge-<familiar_id>`, spawned in `start()`, idempotent — second `start()` returns early): `sleep(nudge_tick_seconds)` (default 60) → sleep tick (28) → if `should_nudge` → `_publish_nudge`: no focused text channel ⇒ skip; else `mark_nudge_pending()` **before** publishing, then a synthetic wake with `_NUDGE_CONTENT` (fixed string mentioning "quiet" and `start_activity`; turn-id prefix `activity-nudge`). The bedtime nudge uses `_BEDTIME_NUDGE_CONTENT` (mentions "sleep", `start_activity`) with prefix `bedtime`. Nudges only earn the model a turn; nothing is auto-started (except the grace backstop).
32. Synthetic wake payload (all three: idle nudge, bedtime nudge, return wake) — topic `discord.text`, `event_id = uuid4().hex`, `turn_id = f"{prefix}-{event_id}"`, `session_id = str(channel_id)`, `sequence_number = 0`, payload exactly: `{"familiar_id", "channel_id", "content", "author": None, "guild_id": None, "message_id": None, "reply_to_message_id": None, "mentions": ()}`. Authorless — the responder treats it as a wake (no `pings_bot`, content-scan fallback applies).

### Engine — sleep passes + dream prose

33. At sleep departure `_kick_sleep_passes` clears `_last_opinion_plan` (one night's material, never reused) and spawns `sleep-passes-<id>`. The task: snapshots `self._active` first (race guard); if `sleep_passes_enabled` is false → return (task exists but does nothing — pinned). Builds `MaintenanceContext(store, llm_clients["background"], familiar_id, display_name, display_tz, apply=True, prompts)` and awaits `run_passes(create_passes(DEFAULT_PASSES, ctx))` (consolidation → opinion sequencing and denylist threading owned by the registry, subsystem 04). Success stores `run.opinion_plan`; failure logs and returns with plan None (wake degrades to seed-only) — never raises.
34. After passes succeed, the task produces + persists the dream immediately (durable within minutes of bedtime): guards — snapshot None or not sleep-typed → return; `self._active` None or id ≠ snapshot id (a return finished/replaced the row mid-passes) → no-op (no stale-row write, no double journal — pinned). Then: generate dream prose → **journal fact first** → `store.set_activity_experience(activity_id, prose)` second (persisted `experience_text` is the single "dream fully produced" signal) → mirror prose into the in-memory record (`dataclasses.replace`) if still current. Any failure logs and degrades to the wake fallback.
35. Dream prose generation order: (a) one-shot seed dream file — if `seed_dream_path` exists, read text, strip, rename to `<stem>.consumed<suffix>`; IO failure or empty text degrades to generation; consumed file is used **verbatim** (no LLM call); (b) else LLM: system=`_DREAM_RAIL`, user=`"Dream seed: <seed>"` plus, when an opinion plan with opinions exists, a "Stances that settled in you tonight…" stanza with `- <opinion.text>` lines (plan consumed/cleared on read); (c) LLM failure or blank → stock line `"Slept deep; whatever I dreamed slipped away on waking."`. The wake never joins the passes task — a fast/forced window can wake before passes finish; seed-only prose there is expected, not a bug.

### Engine — return flow

36. Return timer: one task per departure (`activity-return-<id>`), `delay = planned_return_at − now` computed **once** at arm time against the injected clock, then real `asyncio.sleep(delay)` (skip when ≤ 0) → `_run_return(status="completed")`. All exceptions logged, never orphaned.
37. `_run_return` (single flow for timed and cut-short): entry guard `active is None or _returning` → return; set `_returning = True`; snapshot `now`. Experience: sleep with persisted `experience_text` ⇒ reuse verbatim, mark dream-already-journaled (no LLM, no second journal — pinned including across restart-reload); sleep without ⇒ generate dream prose (35); non-sleep ⇒ LLM with system=`_PROVENANCE_RAIL`, user lines `Activity: <label>` / `Seed: <seed or label if type vanished from catalog>` / optional `Intent noted before leaving: <note>` / cut-short hint line; failure/blank → `"Back from the <label>."`.
38. **COMMIT**: `store.finish_activity(activity_id, status, actual_return_at=now, experience_text)`. After this line she is back. If the commit itself raises, the `finally` still clears in-memory state (she is back in memory; gate returns NORMAL) while the DB row stays active — a restart replays the return (accepted). Everything after the commit is best-effort: each step individually try/except-logged (`return step failed step=<name>`); one failing must not kill the rest.
39. Best-effort steps, in order: (a) mechanical event-fact `store.append_fact(familiar_id, channel_id=departure_channel, text, source_turn_ids=(), valid_from=started_at)` with text `"{familiar_id.capitalize()} spent {%b} {day} {daypart} {label}"` (daypart from local start hour: 5–11 morning, 12–16 afternoon, 17–21 evening, else night; display-tz); (b) return turn — only when a departure channel is known: `append_turn(role="assistant", content=f"[returned from {label}] {experience}", mode=SLEEP_RETURN_MODE|ACTIVITY_RETURN_MODE)` into the **departure-time focused channel**; (c) sleep only, when not already journaled: dream-journal fact (40); (d) archive watermark — when `now − started_at >= archive_after_minutes` and departure turn id known: `set_archive_watermark_all(familiar_id, turn_id=departure_turn_id)` (**every** channel — absence is global); (e) staged promotion — when departure turn id known: `promote_staged_turns_since(familiar_id, after_turn_id, catch_up_limit=focus.catch_up_limit)` (per-channel cap + always-catch mentions live in the store, 03); (f) missed-ping collection (41).
40. Dream-journal fact (stopgap until a dreams table): `append_fact(channel_id=None, text=f"{display_name} dreamed (night of {%b} {day}): {prose}", source_turn_ids=(), subjects=[FactSubject(ego_canonical_key(familiar_id), display_name)], valid_from=started_at)` — dream-framed, `self:`-subject only.
41. Missed pings = content scan ∪ live-noted ids, deduped by turn id, sorted ascending. Scan: `store.recent(channel=departure_channel, limit=500)` filtered to `role == "user"`, `id > departure_turn_id` (or all when departure id None), and `_is_ping(content)` (mention-string scan — reply-pings without `<@id>` are invisible to the scan; that's what the live set covers, including cross-channel). Live set is consumed (cleared) here; ids not already in the scan are fetched via `turns_by_ids`.
42. **C1 ordering invariant**: `_clear_absence_state(now)` (`_active/_departure_*/_return_task = None`, `_last_return = now`) runs **before** the wake publish, so a responder consuming the wake mid-return gates NORMAL — never a judgment off corpse state, no engine state line in the wake prompt (pinned end-to-end with a presence-cb-blocked race test). The `finally` re-runs the clear (idempotent) and resets `_returning`.
43. Wake published only when missed pings exist (no announcement without cause) and a departure channel is known — lands at that channel. Content: header `"[returned from {label} — missed pings while away]"`, then per ping `"- {author.label|someone}: {content trunc 160}"`; the newest 3 pings **not** already inside the channel's last-10 visible turns each get a `turns_around` excerpt (2 turns each side, anchored at `ping.channel_id or wake_channel` — cross-channel live-noted pings excerpt where they happened), rendered indented 4 spaces via `format_turn_for_transcript`. Presence then `("online", None)`.
44. Presence updates never raise (`_set_presence` swallows everything, logs a warning): live incident 2026-06-12 — `change_presence` on a reconnecting websocket killed the run TaskGroup. The cb result may be a plain value or awaitable.

### Engine — restart safety

45. `start()`: reload `active_activity` row (the one with `actual_return_at IS NULL`); if present set `_active`, `_departure_channel_id = current focus` (not the historical one — accepted approximation), `_departure_turn_id = latest_id_at_or_before(started_at)` (recomputed from timestamps, global), re-issue away presence (dropped pre-login by the cb's ready guard in prod; kept so unit tests can exercise the path), and arm the timer: past-due ⇒ `now + 20 s` floor — **never inline at boot** (bus consumers + Discord session must exist before the wake publishes; pinned); future ⇒ at `planned_return_at`. While reloaded-active, `gate` suppresses as usual.
46. `stop()` cancels the return timer, nudge loop and passes task (cancel + await, swallowing `CancelledError`/anything) and does **not** touch the DB — the active row survives for the next boot. A cut-short return must leave the nudge loop armed (pinned).

## Data formats

### DB (owned by 03; consumed here)

```sql
CREATE TABLE IF NOT EXISTS activities (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id        TEXT    NOT NULL,
    type_id            TEXT    NOT NULL,
    label              TEXT    NOT NULL,
    started_at         TEXT    NOT NULL,     -- ISO datetime
    planned_return_at  TEXT    NOT NULL,
    note               TEXT,
    status             TEXT    CHECK(status IN ('completed','cut_short')),
    actual_return_at   TEXT,
    experience_text    TEXT
);
CREATE INDEX IF NOT EXISTS idx_activities_active
    ON activities (familiar_id, actual_return_at, id);

CREATE TABLE IF NOT EXISTS channel_archive_watermark (
    familiar_id  TEXT    NOT NULL,
    channel_id   INTEGER NOT NULL,
    turn_id      INTEGER NOT NULL,
    updated_at   TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, channel_id)
);
```

Append-only activities log; active = `actual_return_at IS NULL`. Store methods used: `create_activity`, `finish_activity`, `set_activity_experience`, `active_activity`, `latest_activity(type_id=)`, `append_fact`, `append_turn(mode=)`, `latest_id`, `latest_id_at_or_before(ts=)`, `set_archive_watermark_all`, `promote_staged_turns_since → PromotionResult{consumed, missed}`, `recent`, `turns_by_ids`, `turns_around`.

### Exact Twitch formatter strings (conformance)

| formatter | output |
|---|---|
| channel point, no input | `{viewer} has redeemed {name}` |
| channel point, input | `{viewer} has redeemed {name} and says: {input}` |
| subscription | `{viewer} has subscribed at tier {tier}` |
| gift sub | `{gifter\|An anonymous gifter} has gifted {count} tier {tier} subscription{s if count!=1}` |
| resub | `{viewer} has subscribed for {months} months at tier {tier} and says: {message}` |
| cheer | `{viewer\|An anonymous cheerer} has cheered with {bits} bits and says: {message}` |
| follow | `{viewer} has followed the channel` |
| ad start / end | `An ad has begun on the channel` / `Ads have ended` |

`TwitchEvent.to_message()` → `Message(role="user", content="[Twitch] " + text, name=viewer.openai_name or "Twitch")`. Viewer names in formatter calls are `Author.label`.

### Bus payloads

- `TOPIC_TWITCH_EVENT = "twitch.event"`; TwitchSource envelope per Behavior 11.
- Synthetic wake payload shape per Behavior 32 (topic `discord.text`).
- Return-wake content format per Behavior 43; nudge contents are the fixed `_NUDGE_CONTENT` / `_BEDTIME_NUDGE_CONTENT` strings (bracketed, mention `start_activity` and `silent()`).

### Marked turns

Return turn: `content = "[returned from {label}] {experience}"`, `role = "assistant"`, `mode = "activity_return"` or `"sleep_return"`. The prefix is display-only; **all downstream keying is on `mode`** (fact-extractor skip / dream framing, pinned in test_fact_extractor + test_activity_engine).

### Engine constants (magic numbers)

`_PAST_DUE_RETURN_FLOOR_S = 20.0`; `_STATE_LINE_NAME_LIMIT = 40`; `_SCAN_LIMIT = 500`; `_MAX_EXCERPTS = 3`; `_EXCERPT_SPAN = 2` (±2 ≈ 5 turns); `_VISIBLE_TAIL = 10`; `_EARLY_BED_MINUTES = 60`; `_SCHEDULE_OVERFLOW_GRACE_MINUTES = 15`; default `nudge_tick_seconds = 60.0`; content trunc 160 in wake lines.

## Config knobs

`data/familiars/<id>/activities.toml` (deep-merged over `data/familiars/_default/activities.toml`; both optional):

| key | default | notes |
|---|---|---|
| `archive_after_minutes` | 45 | positive int; absence ≥ this sets the all-channel archive watermark |
| `idle_nudge_minutes` | 20 | positive int; quiet threshold **and** nudge debounce |
| `min_gap_minutes` | 90 | positive int; post-return nudge gap (nudges only) |
| `active_hours` | unset (always) | `"HH:MM-HH:MM"` in display_tz, may wrap midnight, start≠end |
| `[[catalog]].id/label/seed` | required | non-empty strings; `id` unique; `"sleep"` reserved |
| `[[catalog]].duration_minutes` | required (optional+ignored for `sleep`) | `[lo, hi]`, `0 < lo <= hi` |
| `[[catalog]].reachable` | true | bool |
| `[[catalog]].content_source` | `"authored"` | only valid value; `"adapter"` reserved-rejected |
| `[[catalog]].active_days` | unset (any day) | non-empty list of `mon..sun` |
| `[[catalog]].active_hours` | unset (any time) | same format as top-level |

Consumed from elsewhere (owned by 02): character.toml `[sleep]` `window` (`"HH:MM-HH:MM"`) and `grace_minutes` (default 30) arrive via ctor `sleep_window`/`sleep_grace_minutes`; `display_tz`; sleep prompt-text overrides (`SleepPromptText.from_config`). `familiar.root/seed_dream.md` is the one-shot authored first dream (renamed `seed_dream.consumed.md` after use). `TwitchWatcherConfig` has **no TOML/env source** — programmatic only. No environment variables are read by this subsystem.

## Dependency edges

Imports (module → subsystem):
- `familiar_connect.log_style` — 01 (log formatting helpers incl. `trunc`).
- `familiar_connect.bus.envelope.Event`, `bus.topics` (`TOPIC_DISCORD_TEXT`, `TOPIC_TWITCH_EVENT`), `bus.protocols.EventBus` — 01.
- `familiar_connect.config` (`ConfigError`, `_deep_merge`, `_read_toml`, `parse_hhmm_range`) — 02.
- `familiar_connect.identity` (`Author`, `Author.from_twitch`, `canonical_key`, `label`, `openai_name`, `ego_canonical_key`, `format_turn_for_transcript`) — 02.
- `familiar_connect.history.store` (`ActivityRecord`, `FactSubject`, `HistoryTurn`, schema) / `history.async_store.AsyncHistoryStore` — 03.
- `familiar_connect.llm` (`Message`, `LLMClient`) — 08.
- `familiar_connect.sleep.maintenance` (`DEFAULT_PASSES`, `MaintenanceContext`, `MaintenanceRun`, `SleepPromptText`, `create_passes`, `run_passes`) and `sleep.opinion_formation.OpinionPlan` — 04.

Imported by:
- `processors/text_responder.py` — 06 (gate/GateAction, note_traffic, note_missed_ping, notify_reply_sent, end_turn).
- `processors/fact_extractor.py` — 07 (`ACTIVITY_RETURN_MODE`, `SLEEP_RETURN_MODE`).
- `tools/start_activity.py` — 08 (structural `StartActivityEngine` protocol over the engine).
- `bot.py` and `commands/run.py` — 10 (presence cb, on_ready resync, construction/wiring, shutdown).
- `sources/twitch.py` / `sources/__init__.py` — twitch queue drain (file sits with 10's sources; behavior specced here).

External services: Twitch EventSub WebSocket via the `twitchAPI` Python package (only external dependency; Discord presence goes through 10's callback).

## Test inventory

| test file | behaviors pinned | portability |
|---|---|---|
| `tests/test_twitch.py` | TwitchEvent shape/defaults, every formatter's exact string, builder enable/disable gating, redemption allow-list, ad priority vs `ads_immediate`, `to_message` prefix/name | logic-portable |
| `tests/test_twitch_watcher.py` | ctor field storage + moderator default, per-handler enable/disable + anonymous/gift/tier/months rules, listener registration matrix + argument order, `run()` start/stop-on-cancel + callback→queue delivery | mostly logic-portable; listener/`run` tests need a Rust EventSub mock (duck-typed twitchAPI stand-ins) |
| `tests/test_twitch_source.py` | queue→bus drain, envelope fields, exit on bus shutdown | needs-Rust-mock (bus fake) |
| `tests/test_activities_config.py` | defaults, falsy-when-empty, frozen dataclasses, missing-file disabled, deep-merge, knob/entry validation incl. bool-int rejection, weekday token mapping, hh:mm parsing + wrap, sleep-entry exceptions, `_default` skeleton ships disabled, window/grace no longer catalog keys | logic-portable (frozen-dataclass test is Python-specific-skip) |
| `tests/test_activity_engine.py` (2625 lines — the conformance oracle) | defer/end_turn lifecycle, schedule gate incl. tz localization + half-open boundaries, duration clamp (incl. midnight wrap, room==lo, days-only), gate matrix (flag vs scan, alarm pierce, latch, unfocused, unreachable, name trunc, mid-return NORMAL), return flow (fact text, mode tags, archive all-channels, watermark threshold, wake publish/no-cause-silence, dedupe, cross-channel end-to-end), restart (past-due floor 20 s, presence reload/resync), end_turn/return hardening (fault-injected store/bus, presence boom), staged promotion, should_nudge matrix, nudge loop, late-bound bot id, sleep schedule (window end wake, grace backstop, once-per-occurrence nudge, wrap sides, boot-mid-window), sleep passes (ordering, denylist threading, prompt threading, disabled flag, failure degrade, persist+journal, return-beats-passes no-op), dream return (mode, seed+opinions in prompt, reuse-no-regen, restart reuse, journal-once, stock-line degrade), seed-dream consumable, early-bed guard | logic-portable with Rust fakes (FakeClock/rng/presence recorder/fake LLM, real store + in-proc bus); monkeypatch fault-injection needs a store trait with a faulting test impl |
| `tests/test_text_responder.py::TestActivityGate` | suppress → staged turn + no typing/reply + traffic noted; ping noted only with flag; judgment state line as trailing system msg; judgment reply ⇒ notify+end_turn; silent ⇒ neither notify nor post but end_turn still applies; NORMAL leaves prompt untouched; suppressed log carries server/channel | needs-Rust-mock (responder harness, 06) |
| `tests/test_attentional_tools.py` (start_activity sections) | tool name/schema, enum from catalog, schedule hints in description, description budget, handler arg validation, error passthrough, already-out ⇒ silent sentinel, registry: text-only + engine-gated, never voice | logic-portable |
| `tests/test_bot_interactions.py` (`TestActivityPresenceCb`, `TestOnReadyPresenceResync`) | idle/dnd/online presence mapping, not-ready no-op, on_ready away resync after focus sync, skip without engine | needs-Rust-mock (Discord shell, 10) |
| `tests/test_run_cmd.py` (activity sections) | `_build_activity_engine` returns None without catalog, builds with catalog, handle.activity_engine wiring | needs-Rust-mock (wiring, 10) |
| `tests/test_fact_extractor.py` (mode sections) | extractor skips `activity_return`, dream-frames `sleep_return`, watermark advances over skipped | belongs to 07; contract constants pinned here |

## Rust port notes

- **Task model**: nudge loop, return timer, and sleep-passes are three independently-cancellable tasks → `tokio::task::JoinHandle` stored in the engine; `stop()` = `abort()` + await-ignore (Python swallows *all* exceptions on cancel, not just CancelledError). Name tasks via `tokio::task::Builder` or tracing spans for parity with the Python task names.
- **Injected clock is load-bearing**: `now_fn` drives every decision, but `_sleep_then_return` still calls real `asyncio.sleep(delay)` with delay computed once from the injected clock. Tests never wait — they cancel the timer and drive `_run_return` directly, or use past-due (delay ≤ 0). Port as a `Clock` trait providing `now()` and `sleep_until()`; `tokio::time::pause()` covers most tests, but keep the "compute delay once at arm time" semantic (a clock jump after arming does not re-fire).
- **Engine is effectively single-threaded**: `gate`/`defer_start`/`note_*`/`should_nudge` are sync and lock-free in Python (GIL + event loop affinity). In Rust the state (`_active`, `_staged`, `_returning`, latch sets, nudge clocks) needs interior mutability shared across tasks — a single `Mutex<EngineState>` (or run the engine as an actor with a message channel) is the honest mapping. Watch the C1 ordering invariant (clear-before-wake-publish) across whatever locking you choose; the pinned race test (presence cb blocking mid-return while a responder consumes the wake) must stay reproducible.
- **"Never raises" hierarchy is the core contract**, not incidental: `end_turn`, `notify_reply_sent`, the nudge loop body, `_sleep_then_return`, `_run_return`'s per-step guards, `_set_presence`, and the passes task each have their own catch-log-continue layer with specific recovery (e.g. end_turn re-arms the timer iff the row committed). In Rust, model every step as `Result` and enumerate the recovery in match arms — do not let `?` unify them.
- **Payload typing**: `gate` reads a `dict[str, Any]` with keys `alarm: bool`, `content: str`, `pings_bot: bool` (tri-state: absent/true/false — absence triggers the content-scan fallback), `channel_id: int`, `author: Author-or-anything`. The `isinstance(raw_author, Author)` duck-check → in Rust make the bus payload a typed struct/enum with `Option<bool>` for `pings_bot` and `Option<Author>` for author; the "non-Author author ⇒ `someone`" behavior collapses naturally.
- **Time zones**: `ZoneInfo` → `chrono-tz`; windows are `NaiveTime` pairs; the wrap-length trick `(end − start) % 1 day` needs `rem_euclid` on a `Duration`; weekday Mon=0 matches `chrono::Weekday::num_days_from_monday()`. Preserve (and comment) the documented wrap-day weekday limitation in the schedule gate rather than silently fixing it — tests only cover non-wrapping day+hour combos, but a fix changes model-visible behavior.
- **Month formatting**: `%b`/`local.day` in fact text ("Jun 12") — use chrono's English `%b` and bare day (no zero-pad); `familiar_id.capitalize()` capitalizes only the first char and lowercases the rest — replicate exactly (fact-text tests match substrings).
- **RNG seam**: tests use `random.Random(7)` and an always-return-high stub. Use a `rand::RngCore` box + a `gen_range(lo..=hi)` inclusive both ends (Python `randint` is inclusive).
- **Presence cb sync-or-async** (`inspect.isawaitable`) → just make it `async` in Rust (a boxed `Fn(&str, Option<&str>) -> BoxFuture<Result<()>>`); the dual-arity was Python test convenience.
- **Monkeypatched fault injection** (store method boom, bus publish boom) → the history store and bus must be trait objects here so the hardening tests port; do not take concrete types.
- **Twitch**: `twitchAPI` → the `twitch_api` crate (helix + eventsub types) with `tokio-tungstenite` for the EventSub WS, or keep the watcher unimplemented behind a feature flag given production dormancy. Port `twitch.py` (pure) first — its exact strings are cheap conformance wins. The typed-callback-factory pattern exists only to satisfy the Python type checker; in Rust a generic `fn wrap<E>(handler) -> impl Fn(E) -> Future` collapses it.
- **Redesign candidates**: (1) the `dict`-returning `defer_start` ack/error → a proper `Result<StartAck, StartRefusal>` serialized to JSON only at the tool boundary; (2) `_StagedStart` + end_turn deferral is a tiny state machine — keep it, it is shared precedent with FocusManager (06); (3) the return flow's best-effort step list would be cleaner as an array of named async closures iterated with per-step catch — same order, one log path; (4) seed-dream consumable file rename (`.consumed.md`) is inherently racy on crash — acceptable v1, keep semantics (rename-after-read, IO failure ⇒ generate).
- **Do not port**: the `inspect`-based awaitable sniffing, the `# noqa` exception-breadth pragmas, the pyturso read-visibility workaround noted in `test_passes_persist_dream_prose_and_journal` (spy the append rather than read-back — a Rust SQLite store won't have that flake, but keep the test asserting the *issued* write).
