# 01 bus-and-diagnostics — port spec

Source files: `src/familiar_connect/bus/{__init__,bus,envelope,protocols,router,topics}.py`,
`src/familiar_connect/diagnostics/{__init__,cold_cache,collector,report,spans,voice_budget}.py`,
`src/familiar_connect/macros.py`, `src/familiar_connect/log_style.py`.
Architecture doc: `docs/architecture/streaming-bus.md` (ADR — read it; this spec pins the
implementation-level contracts the ADR does not).

## Role

The in-process event bus is the single data plane of the bot: sources (Discord text, voice
transcripts, Twitch, alarm scheduler) publish topic-addressed `Event` envelopes; processors
(responders, workers, debug logger) consume them via per-subscription queues with per-subscription
backpressure policies. `TurnRouter`/`TurnScope` express barge-in: beginning a new turn in a
session cancels the previous turn's scope, and every long-running pipeline stage polls or awaits
that scope. The diagnostics half is a logs-first timing system: a `@span` decorator and two
singleton recorders (`SpanCollector`, `VoiceBudgetRecorder`) feed both a `/diagnostics` Discord
slash command and a `familiar-connect diagnose` log-grepping CLI, all rendered through the
`log_style` ANSI styling primitives. `macros.py` (SillyTavern macro substitution) and
`log_style.py` are leaf utilities grouped here.

This subsystem is **layer 0**: it imports nothing from any other subsystem (diagnostics imports
only `log_style`; `colorama` is the sole third-party dep). Nearly everything else imports it.

## Public API surface

### bus/protocols.py — the seams

- `BackpressurePolicy` (enum, values are strings): `BLOCK="block"`, `DROP_OLDEST="drop_oldest"`,
  `DROP_NEWEST="drop_newest"`, `UNBOUNDED="unbounded"`. Test pins exactly these four member
  names, no more.
- `EventBus` (Protocol — **the** swappable seam, per ADR a future cross-process impl must drop in):
  - `async start() -> None`
  - `async shutdown() -> None`
  - `async publish(event: Event) -> None`
  - `subscribe(topics: tuple[str, ...], *, policy=BackpressurePolicy.BLOCK, maxsize=0) -> AsyncIterator[Event]`
    — **sync** method returning an async iterator. `maxsize=0` means "default for policy".
- `StreamSource` (runtime-checkable Protocol): `name: str`, `async run(bus: EventBus) -> None`.
  Duck-typed; conformance tests check `isinstance()` structurally (Rust: a trait).
- `Processor` (runtime-checkable Protocol): `name: str`, `topics: tuple[str, ...]`,
  `async handle(event: Event, bus: EventBus) -> None`. `topics` is consulted once at
  registration; no dynamic subscription in v1.

### bus/bus.py

- `Lifecycle` (enum): `STARTING="starting"`, `RUNNING="running"`, `DRAINING="draining"`,
  `STOPPED="stopped"`.
- `InProcessEventBus()` — the only `EventBus` impl. Public attribute `lifecycle: Lifecycle`
  (read by tests and diagnostics). Internal: ordered list of `_Subscription` objects; each
  subscription owns one queue + one policy + a closed flag.

### bus/envelope.py

- `Event` — frozen (immutable, slotted) dataclass:
  `event_id: str`, `turn_id: str`, `session_id: str`, `parent_event_ids: tuple[str, ...]`,
  `topic: str`, `timestamp: datetime` (always tz-aware UTC in practice), `sequence_number: int`,
  `payload: Any`.
- `TurnScope` — mutable dataclass: `turn_id: str`, `session_id: str`, `started_at: float`,
  plus an internal event. Methods: `cancel()` (idempotent), `is_cancelled() -> bool`,
  `async wait_cancelled() -> None`.

### bus/router.py

- `TurnRouter()`:
  - `begin_turn(*, session_id: str, turn_id: str) -> TurnScope` — cancels any prior active
    scope for `session_id`, then registers and returns a fresh scope.
  - `end_turn(scope: TurnScope) -> None` — removes only if the *identical object* is still
    active (identity compare, not turn_id compare); no-op otherwise.
  - `active_scope(session_id: str) -> TurnScope | None`.
  - `shutdown() -> None` — cancels every active scope; **does not clear the map** (post-shutdown
    inspection is a documented feature).
  All methods are synchronous.

### bus/topics.py

Module of string constants (see Data formats). One file, grep-friendly; every routed topic
lives here.

### diagnostics

- `span(name: str)` — decorator factory; wraps sync **or** async callables (detected via
  `iscoroutinefunction`); on every call logs one DEBUG line and records into the global
  `SpanCollector`. Exported as the sole name from `familiar_connect.diagnostics`.
- `SpanRecord` (frozen dataclass): `name: str`, `ms: int`, `status: str`, `at: datetime` (UTC).
- `SpanCollector(maxlen: int = 2000)`:
  - `record(*, name, ms, status)` — thread-safe append, stamps `at=now(UTC)`.
  - `all() -> list[SpanRecord]` — snapshot copy under lock.
  - `by_name() -> dict[str, list[SpanRecord]]` — insertion-ordered grouping.
  - `summary() -> dict[str, dict[str, float]]` — per name: `{count, p50, p95, last_ms}`.
  - `clear()`.
- `get_span_collector(maxlen: int = 2000) -> SpanCollector` — process-wide singleton, created
  on first call (`maxlen` honored only then). `reset_span_collector()` — tests only.
- `VoiceBudgetRecorder(*, max_turns: int = 32)`:
  - `record(*, turn_id: str, phase: str, t: float | None = None)` — stamp a phase timestamp
    (default `time.perf_counter()`); first record per (turn, phase) wins; emits gap spans.
  - `discard(turn_id: str)` — drop a turn's state; no-op if unknown.
  - Phase constants: `PHASE_VAD_END="vad_end"`, `PHASE_STT_FINAL="stt_final"`,
    `PHASE_LLM_FIRST_TOKEN="llm_first_token"`, `PHASE_TTS_FIRST_AUDIO="tts_first_audio"`,
    `PHASE_PLAYBACK_START="playback_start"`; `SPAN_TOTAL="voice.total"`.
  - `get_voice_budget_recorder()` / `reset_voice_budget_recorder()` — singleton pair,
    mirroring the collector.
- `render_summary_table(summary: dict[str, dict[str, float]]) -> str` — Discord-friendly
  code-fenced monospace table; shared by the `/diagnostics` slash command (`bot.py`) and the
  `diagnose` CLI (`commands/diagnose.py`).
- cold_cache (research-phase signal detectors, pure functions):
  - `detect_topic_shift(*, new_text, prior_context, min_overlap=0.15, min_tokens=4) -> bool`
  - `detect_unknown_proper_noun(*, new_text, prior_context, stopwords=_SENTENCE_STARTER_STOPWORDS) -> list[str]`
  - `detect_silence_gap(*, prev_turn_at: datetime | None, current_turn_at: datetime, threshold_seconds=300.0) -> float | None`
  - `log_signals(*, channel_id: int, turn_id: str, new_text, prior_context, prev_turn_at,
    current_turn_at, topic_shift_threshold=0.15, topic_shift_min_tokens=4,
    silence_gap_threshold_s=300.0) -> dict[str, object]` — runs all three, logs one INFO line
    per firing signal, returns which fired. Informational only today (no cache invalidation
    is driven by it) — called by `voice_responder`.

### macros.py

- `MacroContext` dataclass: `char=""`, `user="User"`, `scenario=""`, `personality=""`,
  `description=""`.
- `substitute(text: str, ctx: MacroContext) -> str` — SillyTavern macro subset.
- **Currently dead code in src/** — only `tests/test_macros.py` references it. Port it (tests
  pin it; identity/character-card code in subsystem 02 is its intended consumer) but flag to
  the caller-side spec that nothing wires it yet.

### log_style.py

- `init(strip: bool = False)` — colorama init, called once by `setup_logging` (in `cli.py`).
- Color constants (public, string ANSI escapes): `W C G Y B M R LG LY LC LM LB LW RS`
  (colorama `Fore.*` + `Style.RESET_ALL`).
- `tag(text, color) -> str`, `kv(key, val, *, kc=W, vc=W) -> str`, `word(text, color) -> str`,
  `trunc(text, limit=200) -> str`.
- `StyledFormatter(logging.Formatter)` — repaints leading tag and inserts level label for
  WARNING+; prefixes `DEBUG:` at DEBUG; passthrough at INFO; appends exception/stack text
  like stdlib.

`setup_logging(verbose, level)` lives in `cli.py` (subsystem 02/10 wiring) but its contract is
pinned by this subsystem's `tests/test_logging.py` — see Behaviors 40–42.

## Behaviors & invariants

### Bus lifecycle

1. A new `InProcessEventBus` is in `STARTING`. `start()` transitions to `RUNNING` **only from
   STARTING**; calling it in any other state is a no-op (idempotent). `shutdown()` is a no-op if
   already `STOPPED`; otherwise it sets `DRAINING`, closes every subscription, yields **exactly
   one event-loop tick** (`await asyncio.sleep(0)`), then sets `STOPPED`. It does *not* wait for
   consumers to finish draining.
2. `publish()` is refused (warning log `"publish after stop: topic=%s"`, event dropped, no
   error) only when lifecycle is `STOPPED`. Publishing while `STARTING` or `DRAINING` is
   **not** blocked — events published pre-`start()` are delivered to existing subscriptions.
   Lifecycle is bookkeeping, not a gate (except STOPPED).
3. `subscribe()` never consults lifecycle: it can be called before `start()` and its events
   are received. Registration is synchronous and takes effect for the next `publish()`.
4. **There is no unsubscribe.** The subscription list only grows; subscriptions are closed
   only by bus `shutdown()`. An abandoned iterator's queue keeps accumulating events (bounded
   by policy). The Rust port should keep the observable semantics but may add Drop-based
   cleanup (see Rust notes).

### Fan-out, ordering, backpressure

5. Fan-out is per-subscription queues: every subscription whose `topics` frozenset contains
   `event.topic` gets its own copy (same `Event` object; envelope immutability is what makes
   sharing safe). Subscribers never share a queue.
6. `publish()` awaits `put()` on matching subscriptions **sequentially in subscription
   registration order**. Consequences the port must preserve: (a) per-subscriber FIFO order
   equals publish order; (b) a full BLOCK-policy subscriber stalls the publisher *and* delays
   delivery to all later-registered subscribers. This head-of-line coupling is relied on by
   the BLOCK backpressure test (publisher measurably waits on a slow consumer).
7. Policy semantics on full queue (`maxsize` default 64 for BLOCK/DROP_OLDEST/DROP_NEWEST;
   UNBOUNDED has no bound; `maxsize>0` overrides the default; `maxsize<=0` means
   default-for-policy):
   - `BLOCK` / `UNBOUNDED`: `await queue.put(event)` (UNBOUNDED never actually blocks).
   - `DROP_OLDEST`: evict from the head while full, then non-blocking put; the newest N
     events survive (test: publish 5 into maxsize=2 → receive seq [3, 4]).
   - `DROP_NEWEST`: non-blocking put; incoming event dropped when full (test: publish 5 into
     maxsize=2 → receive seq [0, 1]).
   Drops are **silent** — no log, no counter.
8. `put()` on a closed subscription silently drops the event.
9. Subscription iterator semantics:
   - While open: waits for the next queued event OR the closed signal, whichever first
     (Python implements this by spawning a `get` task and a `closed.wait` task each loop and
     cancelling the loser; a `tokio::select!` is the faithful translation). If an event and
     close race, the event wins and iteration continues.
   - On close: drain everything already queued via non-blocking gets, yield each, then end
     iteration (StopAsyncIteration / `None`). A consumer mid-`await` when shutdown happens
     therefore still receives all queued events before exiting — pinned by
     `test_subscribers_exit_cleanly_on_shutdown`.
   - If the consuming task is cancelled while waiting, both internal tasks are cancelled and
     the cancellation propagates (the run-loop wrappers in `commands/run.py` rely on this to
     unwind under `asyncio.TaskGroup`).
10. Events are **not persisted**; no replay. Durability comes from the `turns` table
    (subsystem 03) — by design (ADR). Losing in-flight events on crash is accepted.

### Envelope contracts

11. `Event` is deeply immutable by convention: the dataclass is frozen; `parent_event_ids`
    is a tuple (hashability pinned by test); `payload` is arbitrary (dicts, dataclasses like
    `Author`, bytes...) and treated as read-only by all consumers.
12. Envelope-field conventions established by producers (cross-module contract; producers
    live in subsystems 08–11 but the format is owned here):
    - `session_id`: `"discord:{channel_id}"` for text events; `"voice:{channel_id}"` for
      voice events. `TurnRouter` keys off these strings (plus a voice-responder extension,
      see 17).
    - `turn_id`: for source events, text sets `turn_id == event_id`; the voice source mints
      one `"voice-" + 12-hex` turn_id per user speech burst and reuses it across
      activity-start → partial(s) → final → activity-end.
    - `event_id`: unique per publish; formats in the wild: `"discord-text-{12 hex}"` (uuid4),
      `"voice-{seq:08d}"`, `"twitch-..."`, `"alarm-..."` — treat as opaque unique strings.
    - `sequence_number`: monotonic **per source instance**, starting at 1 (each source keeps
      its own `_seq` counter). NOT globally monotonic, NOT per-session. Order-sensitive
      consumers must sort or tolerate.
    - `parent_event_ids`: empty tuple for source events; derived events carry lineage.
      In-memory only; derived SQLite rows carry `source_turn_ids` instead (subsystem 03).
    - `timestamp`: `datetime.now(tz=UTC)` at construction.

### TurnScope / TurnRouter

13. `TurnScope.cancel()` is idempotent; `wait_cancelled()` unblocks all waiters once cancelled
    (level-triggered — a waiter that starts after cancel returns immediately).
14. `begin_turn()` cancels the prior scope of the same session **before** registering the new
    one; scopes in different sessions are fully independent. Pinned latency: a task blocked in
    `wait_cancelled()` observes the cancel within 50 ms of the superseding `begin_turn`
    (test asserts < 0.05 s; ADR budget is sub-200 ms barge-in).
15. `begin_turn` sets `started_at` from the running event loop's monotonic clock, or `0.0`
    when called with no running loop (tests construct routers outside a loop).
16. `end_turn(scope)` is idempotent and *identity-guarded*: ending a stale (superseded) scope
    must not clear the newer active scope.
17. Scope-key convention used by callers (contract, not enforced here): `TextResponder` uses
    the event `session_id` directly (`"discord:{channel_id}"`); `VoiceResponder` uses
    `"{session_id}:user:{user_id}"` (per-user barge-in within one voice channel), falling back
    to plain `session_id` when `user_id` is absent. `typing_interrupt` (subsystem 10) calls
    `active_scope(f"discord:{channel_id}")` and `.cancel()` directly — cancel may come from
    outside `begin_turn`.
18. `TurnRouter` has **no locking**; it assumes single-threaded event-loop access. The Rust
    port on a multi-threaded runtime must add synchronization without changing observable
    semantics (see Rust notes).
19. Processor dispatch error contract: the `Processor.handle` docstring promises "raised
    exceptions logged and swallowed by dispatcher", **but the actual dispatch loops in
    `commands/run.py` do not catch anything** — an escaped exception kills the whole
    `asyncio.TaskGroup` (i.e., the process unwinds). In practice responders catch internally.
    The Rust port should resolve this contradiction deliberately: recommend making the
    dispatcher actually log-and-continue, and noting the divergence.

### @span and SpanCollector

20. `@span(name)` wraps sync and async callables transparently (preserving the wrapped
    function's identity/metadata). Timing = wall `perf_counter` delta, reported as
    `int(elapsed_seconds * 1000)` (truncation toward zero). Status is `"ok"` on return,
    `"error"` on **any** raise (BaseException — includes CancelledError); the exception is
    re-raised unchanged. The log/record happens in `finally` — always emitted.
21. The span log line is DEBUG level on logger `"familiar_connect.diagnostics"` and, after
    ANSI stripping, contains `span=<name>`, `ms=<int>`, `status=<ok|error>` in that order
    (rendered via `tag("span", LM)` + three `kv`s). DEBUG (not INFO) is pinned by test —
    visible at `-vv` only.
22. Recording into the singleton collector must **never raise into the caller** — all
    exceptions from `get_span_collector().record(...)` are suppressed. Same rule in
    `VoiceBudgetRecorder._emit` and `llm.py`'s `_span` helper.
23. `SpanCollector` is a bounded ring (deque `maxlen`, default 2000): appending past capacity
    evicts oldest (test: maxlen 3, 5 records → last 3 remain, in order). Appends and reads are
    mutex-protected — callers include non-event-loop threads (Discord voice callbacks).
24. `summary()` percentile: linear interpolation on the sorted values with
    `rank = pct/100 * (n-1)`; empty list → 0.0; single value → that value. Pinned:
    [10..100 step 10] → p50 ≈ 55, p95 ≈ 95.5. `last_ms` is the most recent record's ms
    (insertion order, not max). Empty collector → `{}` (no keys).
25. Singletons: `get_span_collector()` / `get_voice_budget_recorder()` lazily create one
    process-wide instance; `reset_*` swaps in `None` so the next get creates a fresh instance
    (tests pin `a is b` and post-reset `c is not a`). Every producer path
    (`@span`, `VoiceBudgetRecorder`, `llm.py`) fetches the singleton at call time, not at
    import time — resets take effect immediately.

### VoiceBudgetRecorder

26. Phase model: per `turn_id`, first write per phase wins (duplicates silently ignored —
    sentence streaming re-stamps `tts_first_audio` per sentence; only the first counts).
    Gap spans emit at the moment the *later* phase of an adjacent pair is recorded and the
    earlier one exists:
    `vad_end→stt_final = "voice.vad_to_stt"`, `stt_final→llm_first_token = "voice.stt_to_ttft"`,
    `llm_first_token→tts_first_audio = "voice.ttft_to_tts"`,
    `tts_first_audio→playback_start = "voice.tts_to_playback"`. Missing predecessor ⇒ that gap
    silently never emits; the rest of the chain still works (each gap is independent).
27. `"voice.total"` emits once when `playback_start` is recorded AND `stt_final` exists, as
    `playback_start − stt_final` — deliberately anchored at `stt_final` even when `vad_end`
    is present, so historical comparisons hold (pinned by test).
28. Gap ms = `max(0, round(delta_seconds * 1000))` — note Python `round()` is
    banker's/half-to-even; negative gaps (out-of-order stamps) clamp to 0.
29. Turn state is an LRU of `max_turns` (default 32) keyed by `turn_id`: touching an existing
    turn moves it to MRU; inserting a new turn evicts LRU entries beyond capacity. A late
    phase on an evicted turn re-creates the turn with no prior phases → no gap emits (pinned).
    `discard(turn_id)` removes state so later stamps emit nothing.
30. Each emitted gap logs one INFO line on `"familiar_connect.diagnostics.voice_budget"`
    (`[budget] turn=<id> span=<name> ms=<int>`) and records into the shared `SpanCollector`
    with `status="ok"`. Emission happens while holding the recorder's mutex — lock order is
    `VoiceBudgetRecorder.lock → SpanCollector.lock`; never acquire in the other order.
31. Phase call sites (cross-module contract): `sources/voice.py` (09) stamps `vad_end`
    (buffered per-user until the turn_id exists, drained before other phases) and `stt_final`
    *before* publishing the final-transcript event (ordering guarantee: recorder sees
    `stt_final` before the responder can stamp `llm_first_token`); `voice_responder` (06)
    stamps `llm_first_token` on first LLM stream delta; `tts_player/discord_player` (09)
    stamps `tts_first_audio` and `playback_start`. `llm.py` (08) bypasses the budget recorder
    and writes `llm.ttfb*/llm.ttft*/llm.total*` spans straight into the collector.

### Cold-cache signals

32. Tokenization for topic shift: regex `[\w']{3,}` (UNICODE) over the text, lowercased, as a
    set. Shift fires iff both token sets are non-empty AND `len(new_tokens) >= min_tokens`
    (default 4) AND Jaccard `|∩|/|∪| < min_overlap` (default 0.15). The min-token floor exists
    so short voice fragments ("Oh, dear.") can't fire (pinned).
33. Proper-noun detection: regex `\b([A-Z][a-zA-Z]{2,})\b` over `new_text`; skip matches whose
    lowercase form is in the stopword set (default: a frozen set of ~36 capitalized discourse
    markers, stored lowercase — see source; additions cheap by design); dedupe by exact
    surface string preserving first-seen order; report a noun iff its lowercase form does not
    occur as a **substring** of `prior_context.lower()` (substring, not token, match).
34. Silence gap: `None` if no prior turn or gap `< threshold` (gap exactly equal to threshold
    **fires**); otherwise returns the gap in seconds (float).
35. `log_signals` returns `{}` when nothing fires; otherwise keys `topic_shift: True`,
    `unknown_proper_nouns: list[str]`, `silence_gap_s: float`. Each firing signal logs one
    INFO line on `"familiar_connect.diagnostics.cold_cache"` containing `signal=<name>`
    (`topic_shift` | `unknown_proper_noun` | `silence_gap`), `channel=<id>`, `turn=<id>`;
    proper-noun lines add `nouns=<first 5, comma-joined>`; gap lines add `gap_s=<%.1f>`.
    Threshold kwargs plumb through to the detectors (pinned).

### Macros

36. `substitute` pipeline (order matters): (1) strip comments `{{//[^}]*}}` (comment body
    cannot contain `}`); (2) single pass over `{{([^}]+)}}` tokens with the key
    whitespace-trimmed: `trim` sets a flag and replaces with `""`; keys in
    {char, user, scenario, personality, description} replace with context values (empty-string
    defaults except `user` → `"User"`); anything else (e.g. `{{getvar::x}}`) passes through
    **verbatim**; (3) if any `{{trim}}` was seen, `str.strip()` the whole result. Single pass —
    substituted values are never re-scanned. Module-level `_SIMPLE` dict is vestigial dead code;
    do not port.

### log_style / StyledFormatter

37. Exact render formats (these are parsed downstream — treat as wire formats):
    - `tag(text, color)` = `W + "[" + color + text + W + "]" + RS`
    - `kv(key, val, kc=W, vc=W)` = `kc + key + "=" + RS + vc + val + RS` — note the `=` is
      painted in the *key* color and `RS` sits between `=` and the value. The `diagnose` CLI
      regex depends on this: it tolerates zero or more `\x1b[<digits>m` codes immediately
      after `key=`.
    - `word(text, color)` = `color + text + RS`; `trunc(text, limit=200)` = first `limit`
      chars + `…` iff longer.
    - ANSI codes are single-parameter SGR only (`ESC[<n>m`, colorama Fore = 30–37/90–97,
      reset = `ESC[0m`). Multi-parameter codes (`ESC[1;33m`, 256-color) would break both
      `StyledFormatter._TAG_RE` and the diagnose parser — the Rust port must emit the same
      shape.
38. `StyledFormatter.format`:
    - INFO: message passes through untouched.
    - DEBUG: prefixed `LW + "DEBUG" + RS + ": "`.
    - WARNING/ERROR/CRITICAL: if the message *starts with* a `tag()` render (regex
      `^\x1b\[\d+m\[\x1b\[\d+m([^\x1b]+)\x1b\[\d+m\]\x1b\[0m` — inner text may contain
      anything but ESC, including emoji), repaint the tag's inner text in the level color
      (Y for WARNING, R for ERROR and above) and insert the level name after the tag:
      `[Tag] LEVEL rest`. Otherwise fall back to `LEVEL: message`. Exactly one substitution
      (count=1).
    - Exception/stack handling mirrors stdlib `logging.Formatter`: format `exc_info` once
      into `record.exc_text` (cached — formatting the same record twice must not duplicate
      the traceback), append after a newline; then `stack_info` likewise. Without this,
      `logger.exception(...)` drops tracebacks (pinned by three tests).
39. `init(strip=False)` delegates to `colorama.init(strip=strip, autoreset=False)`. In Rust
    this maps to "always emit ANSI unless told to strip" — colorama's Windows translation is
    irrelevant to conformance but stripping support should exist for non-TTY output.
40. `setup_logging` contract (impl in `cli.py`, pinned by `tests/test_logging.py`):
    verbose 0→WARNING, 1→INFO, ≥2→DEBUG; explicit `level` string (case-insensitive, one of
    DEBUG/INFO/WARNING/ERROR/CRITICAL) overrides verbose; unknown level →
    `ValueError("Invalid log level: {level}")`. Reconfigures root (force), installing a
    single stream handler with `StyledFormatter`.
41. Package-logger trick: after configuring root at `log_level`, the `"familiar_connect"`
    logger is set to `min(log_level, INFO)` so package INFO lines flow to the root handler
    even when root is WARNING (handler has no level filter). `-vv` still flips everything
    to DEBUG. Any Rust logging setup must reproduce this visibility split: package INFO
    always visible, package DEBUG only at `-vv`, third-party INFO hidden at default.
42. Named loggers used by this subsystem (log-routing contract):
    `familiar_connect.diagnostics` (spans, DEBUG), `familiar_connect.diagnostics.voice_budget`
    (INFO), `familiar_connect.diagnostics.cold_cache` (INFO), `familiar_connect.bus.bus`
    (module `__name__`, WARNING on publish-after-stop).

### diagnose CLI / report

43. `render_summary_table`: empty summary → exactly ` ```\nno spans recorded yet\n``` `.
    Otherwise triple-backtick fence; rows sorted by span name; header
    `span<pad>  n(>5)  p50(>6)  p95(>6)  last(>6)` with two-space column gaps; name column
    width = max(longest name, 4); count and last rendered as integers, p50/p95 as `%.0f`.
44. `diagnose` subcommand aggregates `span=… ms=… status=…` markers from log files
    (args: one or more paths, `-` = stdin; unreadable file → error log, continue; files
    opened UTF-8 with `errors="replace"`). Parse regex (DOTALL, per line):
    `span=(ANSI)*(?P<name>[\w.-]+).*?ms=(ANSI)*(?P<ms>\d+).*?status=(ANSI)*(?P<status>\w+)`
    where `ANSI = (?:\x1b\[\d+m)*`. Lines missing any of the three keys are skipped. Output:
    the same `render_summary_table` shape computed with the same interpolated percentile
    function (duplicated in `diagnose.py` — port once, share). `status` is captured but not
    used in aggregation (all statuses aggregate together).
45. `/diagnostics` slash command (`bot.py`, subsystem 10) = `render_summary_table(
    get_span_collector().summary())` + focus/unread lines. The collector resets on process
    restart; log files are the durable record — this split is by design.

## Data formats

### Topic constants (exact strings; grep-stable)

| Constant | String | Producer → Consumer (today) |
|---|---|---|
| `TOPIC_DISCORD_TEXT` | `discord.text` | DiscordTextSource, AlarmWaker (synthetic) → TextResponder, DebugLogger |
| `TOPIC_DISCORD_VOICE_STATE` | `discord.voice.state` | *declared, unused* |
| `TOPIC_VOICE_AUDIO_RAW` | `voice.audio.raw` | *declared, unused* (audio bypasses the bus today) |
| `TOPIC_VOICE_TRANSCRIPT_PARTIAL` | `voice.transcript.partial` | VoiceSource → (no subscriber) |
| `TOPIC_VOICE_TRANSCRIPT_FINAL` | `voice.transcript.final` | VoiceSource → VoiceResponder, DebugLogger |
| `TOPIC_VOICE_ACTIVITY_START` | `voice.activity.start` | VoiceSource → VoiceResponder (barge-in), DebugLogger |
| `TOPIC_VOICE_ACTIVITY_END` | `voice.activity.end` | VoiceSource → (no subscriber) |
| `TOPIC_TWITCH_EVENT` | `twitch.event` | TwitchSource → DebugLogger |
| `TOPIC_LLM_RESPONSE_CHUNK` / `_FINAL` | `llm.response.chunk` / `.final` | *declared, unused* |
| `TOPIC_TTS_AUDIO_CHUNK` / `_FINAL` | `tts.audio.chunk` / `.final` | *declared, unused* |
| `TOPIC_ALARM_FIRED` | `alarm.fired` | AlarmScheduler → AlarmWaker |

Test pins: all `TOPIC_*` values are unique strings; `discord.text`, `voice.transcript.final`,
`twitch.event` exact values asserted. Keep declared-but-unused topics — they are the reserved
namespace for planned streaming.

### Payload shapes (established by producers; the bus itself is payload-agnostic)

- `discord.text`: dict `{familiar_id: str, channel_id: int, guild_id: int | None,
  author: Author, content: str, message_id: str | None, reply_to_message_id: str | None,
  mentions: tuple[Author, ...], images: dict[str, str], pings_bot: bool}` (Author from
  subsystem 02).
- `voice.transcript.final`: dict `{text: str, confidence: float, start: float, end: float,
  speaker: str | None, user_id: int | None}`; `.partial`: `{text, confidence, user_id}`;
  `voice.activity.start/end`: `{user_id: int | None}`.
- `alarm.fired`: see subsystem 08 spec (scheduler-owned).
In Rust, make `payload` a closed enum over these shapes rather than a dynamic Any (see notes).

### Log-line wire formats

- Span line (DEBUG): `[span] span=<name> ms=<int> status=<ok|error>` (with ANSI per §37).
- Budget line (INFO): `[budget] turn=<turn_id> span=<voice.*> ms=<int>`.
- Cold-cache line (INFO): `[ColdCache] signal=<name> channel=<int> turn=<id> [nouns=a,b,…|gap_s=<%.1f>]`.
- Span names in the collector namespace (consumed by `/diagnostics` + `diagnose`):
  `voice.vad_to_stt`, `voice.stt_to_ttft`, `voice.ttft_to_tts`, `voice.tts_to_playback`,
  `voice.total`, `llm.ttfb*/llm.ttft*/llm.total*` (from subsystem 08),
  `summary.tick`, `facts.tick`, `reflection.tick`, `fact_embedding.tick`,
  `fact_supersede.tick`, `people_dossier.tick` (from subsystem 07 via `@span`).

### summary() shape

`{span_name: {"count": float, "p50": float, "p95": float, "last_ms": float}}` — flat string
keys, all numeric values (count is an int-valued float in Python; the renderer casts count and
last to int).

## Config knobs

This subsystem reads **no TOML keys and no environment variables**. All tunables are code
constants / function defaults (callers may override per call):

| Knob | Default | Where |
|---|---|---|
| Queue maxsize (BLOCK/DROP_OLDEST/DROP_NEWEST) | 64 | `bus.bus._DEFAULT_MAXSIZE`; `subscribe(maxsize=)` overrides |
| `SpanCollector` ring size | 2000 | `get_span_collector(maxlen=)` first-call only |
| `VoiceBudgetRecorder` LRU turns | 32 | ctor `max_turns` |
| topic-shift `min_overlap` / `min_tokens` | 0.15 / 4 | detector kwargs, plumbed via `log_signals` |
| silence-gap threshold | 300.0 s | detector kwarg |
| `trunc` limit | 200 | `log_style.trunc` |
| verbosity mapping | 0→WARNING, 1→INFO, 2+→DEBUG | `cli.setup_logging` |

Wiring-level choices that live elsewhere but affect this subsystem: all four production
subscriptions in `commands/run.py` use the **default BLOCK/64** policy (the ADR's
"drop-oldest for voice.audio.raw / unbounded for text+twitch" guidance describes intent for
topics not yet on the bus). `FAMILIAR_ID`, `DISCORD_BOT`, `OPENROUTER_API_KEY` are read by
`commands/run.py` (subsystem 10), not here.

## Dependency edges

Internal imports (this subsystem → others): **none**. `diagnostics/*` → `log_style` only;
`bus/*` is self-contained. Third-party: `colorama` (log_style only). This is the bottom layer.

Importers of this subsystem (verified by grep):

| Importer | Subsystem | What it uses |
|---|---|---|
| `familiar.py` | 02 config+identity | constructs `InProcessEventBus`, `TurnRouter`; holds them as `Familiar.bus/.router` (typed as `EventBus` protocol) |
| `config.py`, `identity.py` | 02 | (log_style only, via package) |
| `history/store.py`, `history/fts.py` | 03 | log_style |
| `embedding/fastembed.py`, `sleep/maintenance.py` | 04 | log_style |
| `context/*`, `focus.py`, `prompt_fill.py` | 05 | log_style (`prompt_fill` is 05's own crash-safe fill; distinct from `macros.py`) |
| `processors/text_responder.py`, `voice_responder.py`, `debug_logger.py`, `history_writer.py` | 06 | Event, EventBus, topics, TurnRouter.begin/end/active_scope, voice-budget recorder, cold_cache.log_signals, log_style |
| `processors/{summary,fact_*,reflection,people_dossier}_worker.py` | 07 | `@span`, log_style |
| `llm.py`, `tools/{registry,scheduler,waker,alarm,…}.py` | 08 | get_span_collector (llm), bus publish + topics (scheduler/waker), ToolContext carries `bus` |
| `sources/voice.py`, `stt/*`, `tts.py`, `tts_player/*`, `voice/*`, `sentence_streamer.py` | 09 | Event/EventBus/topics, voice-budget phases, log_style |
| `bot.py`, `commands/run.py`, `commands/diagnose.py`, `cli.py`, `typing_interrupt.py`, `sources/discord_text.py` | 10 | bus wiring + lifecycle, subscribe loops, get_span_collector + render_summary_table (`/diagnostics`), diagnose CLI, StyledFormatter/init, TurnRouter.active_scope().cancel() |
| `sources/twitch.py`, `twitch_watcher.py`, `activities/engine.py` | 11 | Event/EventBus/topics publish, log_style |

`macros.py` currently has zero src importers (tests only).

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `tests/test_bus.py` | Lifecycle transitions (§1); fan-out to N subscribers each seeing every event in order; topic isolation; DROP_OLDEST keeps newest [3,4] of 5 at maxsize 2; DROP_NEWEST keeps [0,1]; BLOCK measurably back-pressures publisher (≥30 ms for 3 events @20 ms consumer); UNBOUNDED holds 100 without loss; consumers exit cleanly on shutdown having received pre-shutdown events (§9) | logic-portable (tokio tests; timing asserts translate) |
| `tests/test_bus_envelope.py` | Event field carriage; frozen immutability; `parent_event_ids` tuple; TurnScope identity fields; cancel idempotence; `wait_cancelled` resolves after cancel | logic-portable (immutability is free in Rust; keep the semantic tests) |
| `tests/test_bus_router.py` | begin/cancel-prior/same-session; cross-session independence; <50 ms cancel propagation; end_turn idempotent + identity-guarded (stale end doesn't clear newer); shutdown cancels all | logic-portable |
| `tests/test_bus_protocols.py` | Structural conformance of dummy source/processor; missing-method rejection; the exact four backpressure variants | Python-specific-skip for `isinstance` structural checks (traits enforce at compile time); keep the enum-variant count test |
| `tests/test_bus_topics.py` | All TOPIC_* unique strings; exact values for discord.text / voice.transcript.final / twitch.event | logic-portable |
| `tests/test_diagnostics_spans.py` | span on async + sync fns; `span=<name>`/`ms=\d+` in ANSI-stripped output; DEBUG level; emits with `status=error` on raise (exception propagates) | logic-portable (assert against captured log output; needs a Rust log-capture harness) |
| `tests/test_span_collector.py` | record/all/by_name; ring eviction order; percentile values (p50=55±1, p95=95.5±1 for 10..100); empty summary `{}`; singleton identity + reset; `@span` integration incl. error status | logic-portable |
| `tests/test_diagnostics_report.py` | empty-table placeholder; rows sorted by name; header columns present | logic-portable (golden-string) |
| `tests/test_voice_budget.py` | gap-on-later-phase; full funnel emission order incl. `voice.total`=700 ms anchored at stt_final; skipped-predecessor no-emit; first-write-wins dedupe; per-turn isolation; LRU eviction (max_turns=2) and `discard`; total skipped without stt_final; optional vad_end chain; singleton + reset; default-clock path | logic-portable |
| `tests/test_cold_cache_signals.py` | topic-shift fire/quiet incl. min_tokens floor + configurability; proper-noun detection, stopword filtering, substring-known suppression, custom stopwords; silence gap incl. no-prior; `log_signals` return-dict keys + `signal=` log lines + `{}`/no-log when quiet | logic-portable |
| `tests/test_macros.py` | comment stripping; trim semantics (flag + whole-string strip); each simple macro + defaults (`user`→"User", others empty); unknown macros verbatim (`{{getvar::…}}`); combined pipeline order | logic-portable |
| `tests/test_logging.py` | setup_logging verbose/level matrix + ValueError; StyledFormatter per-level output (ANSI-stripped golden strings), tag repaint colors (Y/R on inner text + label), untagged fallback, traceback append + no-double-append on repeated format | mostly logic-portable; the `logging.root.level` assertions are Python-specific — re-pin as "effective filter level" of the Rust logger |
| (indirect) `tests/test_debug_logger_processor.py`, `test_discord_text_source.py`, `test_voice_source.py`, `test_typing_interrupt.py`, `test_alarm_*.py`, responder tests (`test_voice_responder.py::TestBargeIn` etc.) | exercise the bus/router/budget through real producers/consumers — conformance oracles for envelope conventions (§12, §17) and barge-in latency | needs-Rust-mock (belong to subsystems 06–10 but will not pass unless this subsystem's semantics match) |

## Rust port notes

- **Bus core.** Do not reach for `tokio::sync::broadcast` (single shared ring, lagging-receiver
  error semantics ≠ per-subscription policies). Model each subscription as its own channel:
  `BLOCK` → bounded `mpsc` with `send().await`; `UNBOUNDED` → `unbounded_channel`;
  `DROP_OLDEST`/`DROP_NEWEST` → either a small custom `Mutex<VecDeque>+Notify` queue or a
  `try_send` + evict-and-retry loop on a bounded channel. Preserve §6's sequential
  publish-in-registration-order (a plain `for sub in subs { sub.put(ev).await }`) — do not
  parallelize delivery, tests depend on the coupling. Subscription list behind
  `Mutex<Vec<...>>` (subscribe is sync-ish and rare; publish iterates a snapshot or holds the
  lock across awaits — snapshot + per-sub closed-flag is safer).
- **Event sharing.** Publish clones an `Arc<Event>` per subscription. `payload: Any` must
  become a closed `enum EventPayload { DiscordText(..), VoiceTranscriptFinal(..), … }` per
  the shapes in Data formats — the payload set is closed and known; do not use
  `Box<dyn Any>`/`serde_json::Value`. `timestamp` → `chrono::DateTime<Utc>`.
- **Iterator → Stream.** Replace the spawn-two-tasks-and-race Python idiom with
  `tokio::select! { ev = rx.recv() => …, _ = closed.cancelled() => drain-then-end }`, then a
  `while let Ok(ev) = rx.try_recv()` drain after close. Ensure a racing event still wins over
  close (§9). Expose as `impl Stream<Item = Arc<Event>>` or an
  `async fn next() -> Option<...>` handle; add `Drop` on the handle that marks the
  subscription closed so the no-unsubscribe leak (§4) is fixed *without* changing test-visible
  behavior.
- **Shutdown tick.** `await sleep(0)` "one loop tick" has no exact tokio equivalent;
  `tokio::task::yield_now().await` is the moral match. The pinned behavior is only: queued
  events remain drainable after shutdown, and shutdown doesn't wait for consumers.
- **TurnScope/TurnRouter.** `TurnScope.cancel` maps 1:1 to `tokio_util::sync::CancellationToken`
  (idempotent, level-triggered `cancelled().await`). `TurnRouter` needs
  `Mutex<HashMap<String, TurnScope>>` (multi-threaded runtime; Python relied on the GIL +
  single loop). `end_turn` identity comparison → compare a unique scope id (e.g. token ptr or a
  u64) — turn_id equality is NOT the contract (two scopes can share a turn_id in theory).
  `started_at` "loop time or 0.0" → `tokio::time::Instant::now()` unconditionally; the 0.0
  fallback exists only because Python tests run without a loop — drop it, adjust the one
  assertion (`started_at > 0`).
- **Singletons.** `get_span_collector`/`get_voice_budget_recorder` → `OnceLock` or
  `static Mutex<Option<Arc<…>>>`. The tests-only `reset_*` needs the resettable form, or
  redesign to dependency-injection (recorder handle threaded through call sites) — DI is
  cleaner but touches every producer in subsystems 06–09; if you keep globals, keep the
  reset seam behind `#[cfg(any(test, feature = "test-util"))]`.
- **@span → macro or tracing.** Two options: (a) a small attribute/proc-macro or
  `time_span!(name, expr)` wrapper that reproduces the exact DEBUG line + collector record;
  (b) adopt `tracing` with a custom `Layer` feeding `SpanCollector`. Either way the **log
  line remains a wire format** (§21, §44): `span=`, `ms=`, `status=` in order, ANSI limited to
  `ESC[<digits>m`, because `diagnose` regex-parses persisted logs (including logs produced by
  the Python version — cross-version compatibility of `diagnose` is desirable). Port the
  percentile/aggregation function once and share it between collector and CLI (Python
  duplicates it).
- **Threading model shift.** In Python only `SpanCollector` and `VoiceBudgetRecorder` are
  lock-protected (called from Discord audio threads); the bus and router are loop-affine and
  lock-free. On tokio everything must be `Send + Sync`. Keep the documented lock order
  (§30: budget lock, then collector lock) or restructure `_emit` to record outside the budget
  lock (safe: gap values are computed before emission; recommended).
- **Numeric edge cases.** `@span` ms uses truncation (`int()`), voice budget uses
  `round()` = half-to-even then `max(0, …)`. `f64::round` is half-away-from-zero — divergence
  only at exact .5 ms boundaries; document and accept, or use `round_ties_even` (stable since
  Rust 1.77) for bit-parity.
- **log_style.** Use raw SGR strings (`"\x1b[33m"` etc.) rather than a color crate that emits
  compound sequences; the formatter's tag-repaint regex and the diagnose parser both assume
  single-parameter codes (§37). `StyledFormatter` becomes a formatting function in whatever
  logging facade is chosen (`tracing-subscriber` custom `FormatEvent` or a `log` crate
  Formatter); replicate the §41 two-tier visibility (package INFO at default verbosity,
  DEBUG at `-vv`, third-party at WARNING) with `EnvFilter`-style directives, e.g.
  `warn,familiar_connect=info`.
- **Processor error contract (§19).** Resolve the docstring-vs-wiring contradiction in the
  port: make the dispatch loop (the Rust equivalent of `_run_*` in run.py) catch, log via the
  styled logger, and continue — that is what the Protocol promises and what a long-running
  bot needs. Flag this as an intentional behavior *fix* in the port changelog.
- **Redesign candidates (do not transliterate):** the per-iteration task-pair race in
  `_Subscription.iterator` (use select); the duplicated percentile code; module-global
  mutable singletons; the unused `_SIMPLE` dict in macros; consider a typed `Topic` newtype or
  enum instead of bare `&'static str` constants (keep the exact string values for logs and any
  serialized form).
