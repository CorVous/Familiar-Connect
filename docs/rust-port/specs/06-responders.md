# 06-responders — port spec

Source files: `processors/text_responder.py`, `processors/voice_responder.py`,
`processors/projectors.py`, `processors/history_writer.py`,
`processors/debug_logger.py`, `silence.py`, `typing_interrupt.py` (~2,165 loc).

## Role

The reply loops. `TextResponder` and `VoiceResponder` consume bus events,
assemble a layered prompt (05), stream an LLM reply (08), gate it through the
`<silent>` sentinel, and deliver it (Discord post via injected callback / TTS
via `TTSPlayer`), persisting user+assistant turns to history (03). Everything
is scoped to a per-turn `TurnScope` so barge-in (new utterance, user typing, a
newer message) cancels in-flight work cooperatively. The subsystem also owns
the projector registry (factory layer over the 07 background workers), the
legacy `HistoryWriter` processor, the `DebugLoggerProcessor`, the
`SilentDetector` stream inspector, and the `TypingInterruptHandler` typing
policy. Concurrency semantics here are the product: every lock, spawn and
cancellation point below is deliberate and regression-tested.

## Public API surface

### Processor contract (implicit protocol, all five processors)

```python
name: str                      # human label for logs / task names
topics: tuple[str, ...]        # bus topics this processor subscribes to
async def handle(event: Event, bus: EventBus) -> None
```

The wiring layer (10, `commands/run.py`) runs one dispatch task per processor:
`async for event in bus.subscribe(proc.topics): await proc.handle(event, bus)`.
`handle` therefore MUST NOT block the dispatch loop for long-running work
(VoiceResponder spawns; TextResponder deliberately runs inline — see B-T1).

### TextResponder

```python
TextResponder(*,
    assembler: Assembler,                       # 05
    llm_client: LLMClient,                      # 08 ("prose" slot in prod)
    send_text: Callable[[int, str, str | None, tuple[int, ...]], Awaitable[str | None]],
        # (channel_id, content, reply_to_message_id, mention_user_ids) -> platform msg id
    history_store: AsyncHistoryStore,           # 03
    router: TurnRouter,                         # 01
    familiar_id: str,
    trigger_typing: Callable[[int], AsyncContextManager[None]] | None = None,
    typing_handler: TypingInterruptHandler | None = None,
    tool_registry: ToolRegistry | None = None,                            # 08
    tool_context_factory: Callable[[int, str, dict[str, str]], ToolContext] | None = None,
        # (channel_id, turn_id, images) -> ToolContext
    post_history_instructions: str = "",
    display_tz: str = "UTC",
    focus_manager: FocusManager | None = None,  # 05
    loop_max_iterations: int = 5,
    activity_engine: ActivityEngine | None = None,  # 11
)
name = "text-responder"; topics = (TOPIC_DISCORD_TEXT,)
```

Every `| None` collaborator is a swappable seam; `None` means "feature off,
zero behavior change" (focus staging, activity gating, typing indicator,
typing policy, tool calling). `send_text`, `trigger_typing`,
`tool_context_factory` are plain callables — Rust: trait objects or fn
pointers, injected at construction.

Module-level helpers (pure, unit-tested, keep as free functions):

- `_rewrite_pings(content, label_to_key) -> (rewritten, tuple[int, ...])`
- `_strip_leaked_metadata_prefix(content) -> str` (imported by tests)
- `_consume_thread_marker(content) -> (stripped, wanted_thread: bool, target_id: str | None)`

### VoiceResponder

```python
VoiceResponder(*,
    assembler, llm_client,                      # "fast" slot in prod
    tts_player: TTSPlayer,                      # 09 protocol: speak(text, *, scope), stop()
    history_store, router, familiar_id,
    member_resolver: Callable[[int, int], Author | None] | None = None,  # SYNC callable
    tool_registry=None, tool_context_factory=None,
    tool_filler_phrases: tuple[str, ...] = ("one sec...", "hold on...", "checking..."),
    post_history_instructions="", display_tz="UTC",
    focus_manager=None, loop_max_iterations=5,
)
name = "voice-responder"
topics = (TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_FINAL)

async def wait_until_idle() -> None   # gather all in-flight final tasks; used by
                                      # tests + graceful shutdown; swallows CancelledError
```

### HistoryWriter (legacy — NOT wired in production run loop)

```python
HistoryWriter(*, store: AsyncHistoryStore, familiar_id: str)
name = "history-writer"; topics = (TOPIC_DISCORD_TEXT,)
```

TextResponder owns the user-turn write in production (read-after-write
consistency for `RecentHistoryLayer` in the same task; a separate writer task
would race the responder's `assemble`). HistoryWriter is retained, exported
from `processors/__init__`, and fully tested — port it, but do not wire it
alongside TextResponder (double-write).

### DebugLoggerProcessor

```python
DebugLoggerProcessor(*, topics: tuple[str, ...])   # topics injected, not fixed
name = "debug-logger"
```

Logs one INFO line per event (topic, event_id, turn_id, session, seq, payload
`repr` truncated to 160 chars). Passive; never republishes.

### Projector registry (`processors/projectors.py`)

```python
class MemoryProjector(Protocol):      # THE seam — third parties implement this
    name: str
    async def run(self) -> None: ...  # forever loop; cancellation stops it

@dataclass(frozen=True)
class ProjectorContext:
    store: AsyncHistoryStore
    llm_clients: dict[str, LLMClient]
    familiar_id: str
    embedder: Embedder | None = None
    memory: MemoryProvidersConfig = MemoryProvidersConfig()
    familiar_display_name: str | None = None
    dream_extraction_clause: str = ""

ProjectorFactory = Callable[[ProjectorContext], MemoryProjector]
register_projector(name: str, factory) -> None    # re-registration overwrites
known_projectors() -> set[str]
create_projectors(*, names: list[str], context) -> list[MemoryProjector]
    # instantiates in `names` order; ValueError "unknown memory projector {name!r}; valid: ..."
DEFAULT_PROJECTORS = ("rolling_summary", "rich_note", "people_dossier",
                      "reflection", "fact_supersede")
```

Six built-ins registered at import time (module side effect — in Rust use an
explicit `builtin_registry()` constructor instead of global mutable state):
`rolling_summary`→SummaryWorker, `rich_note`→FactExtractor,
`people_dossier`→PeopleDossierWorker, `reflection`→ReflectionWorker,
`fact_supersede`→FactSupersedeWorker, `fact_embedding`→FactEmbeddingWorker.
All worker types live in subsystem 07; this file only maps
`[providers.memory.<name>]` knob structs into constructor args and picks the
`"background"` LLM slot for every built-in. `fact_embedding` is registered
but NOT in `DEFAULT_PROJECTORS` (opt-in), and its factory raises
`ValueError("fact_embedding projector requires a configured embedder. ...")`
when `context.embedder is None`. Wiring (10) spawns one task per projector in
the TaskGroup: `tg.create_task(proj.run(), name=proj.name)`.

### SilentDetector (`silence.py`)

```python
SILENT_TOKEN = "<silent>"
class SilentDetector:
    decided: bool | None          # True=silent, False=speak, None=pending
    def feed(delta: str) -> bool | None
```

### TypingInterruptHandler (`typing_interrupt.py`)

```python
TypingInterruptHandler(*,
    config: DiscordTextConfig, router: TurnRouter,
    is_subscribed: Callable[[int], bool],
    bot_user_id_provider: Callable[[], int | None])
def notify_typing(*, channel_id: int, user_id: int, is_bot: bool) -> None   # SYNC
def notify_user_message(*, channel_id: int) -> None                        # SYNC
def backoff_deadline(channel_id: int) -> float | None    # monotonic; lazily expires
def current_backoff_s(channel_id: int) -> float          # test seam
async def wait_for_backoff(channel_id: int) -> None
```

Wired by `bot.py` (10): Discord `on_typing` → `notify_typing`; consumed by
TextResponder (`wait_for_backoff` + `notify_user_message`).

## Behaviors & invariants

### SilentDetector (S)

1. **S1** Algorithm, exactly: append delta to buffer; `stripped =
   buf.lstrip()`; if `stripped.startswith("<silent>")` → decide True; elif
   `len(stripped) >= len("<silent>")` (8 chars) → decide False; elif
   `stripped` non-empty and is not a prefix of `"<silent>"` → decide False;
   else stay None. Length compare is in characters (token is ASCII so bytes
   work if the ≥ compare is done on char count; multibyte content that reaches
   8+ chars decides False either way).
2. **S2** Decision latches: once non-None, `feed` returns the cached value and
   ignores its argument.
3. **S3** Only *leading* whitespace is tolerated; `<silent>` after any content
   is content (`"Sure, " + "<silent>"` → False). Token split across arbitrary
   delta boundaries must still be detected (`"<sil"`,`"ent>"` → True; `"<sil"`,
   `"k"` → False).
4. **S4** Pure-whitespace streams stay pending forever; the responder's
   empty-reply guard (T13/V13) is the backstop.

### TypingInterruptHandler (Y)

1. **Y1** `notify_typing` gate order: (a) `respond_to_typing` off → no-op
   (disables BOTH cancel and backoff); (b) unsubscribed channel → no-op; (c)
   `user_id == bot_user_id_provider()` (when provider returns non-None) →
   no-op (own indicator must not self-cancel); (d) `is_bot` → apply backoff;
   (e) else cancel active turn.
2. **Y2** User-typing cancel: `router.active_scope(f"discord:{channel_id}")`;
   if present, `scope.cancel()` (soft flag — the responder notices at its next
   checkpoint). No active scope → silent no-op.
3. **Y3** Bot-typing backoff ladder, per channel: `next` defaults to
   `typing_backoff_initial_s`; `applied = min(next, max_s)`; `deadline =
   monotonic_now + applied`; `next ← min(applied*2, max_s)`. Sequence for
   initial=1, max=16: 1, 2, 4, … capped at max. Ladder state is in-memory
   only (lost on restart).
4. **Y4** `notify_user_message` resets the ladder (pops both `next` and
   `deadline`); the next bot-typing event starts at initial again.
5. **Y5** `backoff_deadline` lazily cleans expired deadlines (≤ now → pop,
   return None). `wait_for_backoff` reads the deadline ONCE and sleeps once;
   a new bot-typing event landing during the sleep extends the stored deadline
   but does not extend the current waiter. Preserve this single-sleep
   semantic.
6. **Y6** `current_backoff_s` returns the last *applied* window (0.0 when
   never applied); it survives `notify_user_message` until the next apply
   overwrites it. (Python lazily creates this side map via `hasattr` — in
   Rust just make it a normal field.)

### TextResponder.handle (T) — pipeline order matters

1. **T1** `handle` runs the whole turn inline (no spawn). Text barge-in is
   driven by `TurnScope.cancel()` from other actors (typing handler, a newer
   event's `begin_turn` in the same session), not by dispatcher concurrency.
   Consequence: a second `discord.text` event in the same channel queues
   behind the current one at the bus.
2. **T2** Event validation/drop rules, in order: topic mismatch; `event_id`
   already in in-process `_seen` set; payload not a dict; `familiar_id`
   mismatch; `channel_id` not int; empty `content` unless `payload["wake"] is
   True`. Field extraction is defensive: `author` only if `isinstance(...,
   Author)` else None; `guild_id` int-checked; `message_id` /
   `reply_to_message_id` str-checked; `mentions` filtered to Author instances;
   `images` dict-checked else `{}`. The `_seen` set grows unboundedly for the
   process lifetime (bus never republishes today) — a bounded LRU is an
   acceptable Rust substitute.
3. **T3** After validation: add to `_seen`; `activity_engine.note_traffic()`
   on EVERY handled event (including ones that later stage or suppress) —
   feeds the quiet-clock.
4. **T4** Typing policy (when handler wired): `await
   wait_for_backoff(channel_id)` BEFORE claiming the turn, then
   `notify_user_message(channel_id)` (ladder reset). Order: backoff sleep →
   reset → `begin_turn`.
5. **T5** `scope = router.begin_turn(session_id=event.session_id,
   turn_id=event.turn_id)` — session is `discord:<channel_id>`; this cancels
   any prior in-flight turn for the channel (self-supersede).
6. **T6** Identity upserts before gating: `upsert_account(author)` +
   `upsert_guild_nick` (when guild + `author.guild_nick`); same for every
   mention. These are soft "most recently seen" cache writes.
7. **T7** Activity gate (engine optional): `gate = engine.gate(payload)`.
   `SUPPRESS` ⇒ record user turn staged and stop (no typing, no LLM, no
   reply, no `engine.end_turn()`, no `notify_reply_sent`). `JUDGMENT` ⇒
   normal flow plus `gate.state_line` appended (after `\n\n`) to the trailing
   system message for this turn only. `NORMAL` ⇒ prompt untouched.
8. **T8** Focus staging: `focused = fm is None or fm.is_focused(channel_id)`.
   For non-wake events the user turn is persisted BEFORE streaming (so
   RecentHistoryLayer sees it — pinned by test) via `append_turn(...,
   consumed=focused and not suppressed, pings_bot=payload.get("pings_bot") is
   True, platform_message_id, reply_to_message_id, author, guild_id)`;
   mentions recorded via `record_mentions(turn_id, canonical_keys)`.
9. **T9** Suppressed path: if the suppressed turn pinged the bot
   (`pings_bot`), call `engine.note_missed_ping(user_turn.id)` (live capture
   for cross-channel/reply pings); log `Activity suppressed` with ch/srv
   fields; return.
10. **T10** Unfocused (staged) path: log `📥 Staged`; if
    `fm.should_wake(channel_id)` publish an unread nudge (T20); return —
    no LLM call, no reply, no `fm.end_turn()`.
11. **T11** Wake events (`wake: True`): skip persist/staging entirely (no
    synthetic user turn in history — pinned); if suppressed → drop. A wake
    event earns one focused turn; the model decides whether to `shift_focus`.
12. **T12** Prompt build (`_stream_reply`): `assembler.set_rag_cue(content)`
    first (mutates shared assembler state); `assemble(AssemblyContext(...,
    viewer_mode="text", guild_id))`; unread digest =
    `await history.staged_channels(familiar_id)` only when fm wired (shape:
    `dict[channel_id, (unread, pings)]` — `ChannelUnread` NamedTuple unpacks
    structurally). Message list:
    `[system(head)] + prompt.recent_history + [system(trailing)]` where
    head = join(`prompt.system_prompt`, `_BOT_OUTPUT_INSTRUCTIONS`,
    `build_final_reminder(viewer_mode="text", include_time=False,
    focus_channel_id, unread_digest, channel_names)`) with `"\n\n"`;
    trailing = `build_final_reminder(viewer_mode="text", display_tz,
    include_mode_instruction=True, post_history_instructions,
    focus_channel_id, unread_digest, channel_names, guild_name)`. Invariants
    pinned by tests: head omits the timestamp and the server name
    (byte-stable cache prefix); trailing carries "It is now", the text-mode
    directive ("Markdown"/"chatting in a text channel"), `[@DisplayName]`
    sentinel docs, the focus line `Your attention is currently on #<name>.`
    plus `"<Guild>" server` clause only when the guild is known, and
    `post_history_instructions` deepest (after the mode directive). Judgment
    state line is appended after even that.
13. **T13** Bare-stream loop (`tool_mode` false): iterate
    `llm.chat_stream(messages)` (yields `str` deltas). Per delta:
    `scope.is_cancelled()` → return None; `silent.feed(delta)` True → log
    `decision=silent` and return None; on first False decision, if
    `trigger_typing` wired, enter the typing async-context via an
    `AsyncExitStack` (exactly once). The stack guarantees `__aexit__` on
    normal end, cancellation return and exception. Indicator invariants
    (pinned): never opened for `<silent>` turns, reasoning-then-silent turns,
    or zero-delta streams; entered/exited exactly once for real replies. Any
    stream exception → WARN `llm_stream_error` and return None (loop never
    crashes).
14. **T14** `tool_mode = registry and ctx_factory and
    (llm.tool_calling_enabled or getattr(llm, "image_tools_enabled", False))`
    — image-tools alone enters the agentic loop (pinned). Tool path runs
    `agentic_loop(llm, messages, registry, ctx=ctx_factory(channel_id,
    scope.turn_id, images), on_delta, on_iteration_end,
    max_iterations=loop_max_iterations)` (08). `on_delta` mirrors T13's
    silent/typing gating on `delta.content` (a `bail_silent` flag makes later
    deltas no-ops; cancellation makes on_delta a no-op). `on_iteration_end`
    persists ONLY intermediate iterations (`assistant.tool_calls` non-empty):
    an assistant turn with `tool_calls_json=json.dumps(assistant.tool_calls)`
    and one `role="tool"` turn per tool message (`tool_call_id`, content via
    `tool_content_as_text`), both carrying `guild_id`. The terminal text-only
    iteration is NOT persisted there — `handle` persists it after `send_text`
    so the platform message id lands on the row. Pinned history sequence:
    `user, assistant(tool_calls), tool, assistant(text)`.
15. **T15** Empty-completion retry (qwen tool-leak quirk): if the loop result
    has no `final_content`, `is_silent` false, zero `tool_calls_made`, no
    `bail_silent`, and scope not cancelled → log `retry=empty_completion` and
    run `agentic_loop` ONCE more; never a third call (pinned: two LLM calls
    then give up silently). Quirk to preserve or consciously fix: the retry
    call omits `max_iterations`, so it uses the library default (5), not the
    configured cap.
16. **T16** After streaming: `reply is None or scope.is_cancelled()` →
    `engine.end_turn()` (a deferred activity start must still apply on silent
    slip-away — pinned) and return. Whitespace-only reply → WARN
    `skip=empty_reply`, `engine.end_turn()`, return (Discord 400/50006
    defense). No assistant turn in any of these cases.
17. **T17** Output rewriting order (exact): `_consume_thread_marker` →
    `_strip_leaked_metadata_prefix` → `_rewrite_pings`. Contracts:
    - Thread marker regex `\[(?:↩|reply)(?:\s+([^\]\n]+))?\]`: ANY occurrence
      anywhere triggers threading; ALL markers stripped; result `lstrip()`ed;
      first non-empty captured id wins; captured id is `strip().lstrip("#").strip()`
      (models echo the `#` sigil from recent-history rendering); empty after
      stripping ⇒ treated as bare marker.
    - Thread target: explicit id used only if
      `history.lookup_turn_by_platform_message_id(familiar_id, id)` finds a
      row; otherwise (and for bare `[↩]`/`[reply]`) fall back to the
      triggering event's `message_id`. No marker ⇒ `reply_to=None`.
    - Leaked-metadata prefix regex
      `^\s*(?:\[[^\]\n]*(?:#\d|\d:\d|[AP]M)[^\]\n]*\]\s*)+`, applied once at
      the head only; `[note]`-style openers must survive; a metadata clump
      followed by a reply marker (`[14:32 Alice #abc] [↩ msg-1] hi`) leaves
      the marker — but note the marker was already consumed in step 1, so
      order matters.
    - Ping markers `\[@([^\]\n]+)\]`: label found in resolver map AND
      canonical key `discord:<int>` → `<@user_id>` + id appended to
      `mention_user_ids` (in occurrence order); unknown label, non-discord
      platform, or non-int id → plain `@Label`, no ping, no error.
18. **T18** Ping resolver map is built per-turn BEFORE streaming from
    `sync_history.recent_distinct_authors(familiar_id, channel_id, limit=20)`
    + `resolve_label(canonical_key, guild_id, familiar_id)` (synchronous
    store calls). Duplicate label with a different key: first write wins,
    WARN `ambiguous_label`. The map is never surfaced in the prompt.
19. **T19** Send: **per-turn routing** (#170). The target channel is the channel
    a `shift_focus` moved to *during this turn* (recorded turn-locally at the
    tool's call site), else the triggering event channel. The mutable global
    focus (`fm.get_focus("text")`) is **never** read at send time — that read
    was the cross-channel misroute: a slow turn could pick up a concurrent
    turn's shift. Pinned: shift+reply posts to the NEW channel (a turn that
    shifts routes there and persists the assistant turn there); a turn that
    never shifts routes to its own channel regardless of concurrent shifts.
    **Wake turns are shift-or-silent** (#170): a `wake` turn (T20) that produces
    prose but did NOT `shift_focus` this turn is suppressed entirely — no send,
    treated as silent, logged `guard=wake_shift_or_silent action=suppress` then
    `engine.end_turn()` — so a wake reply can only ever reach the channel the
    model deliberately moved to. `send_text` exception → WARN `send_error`,
    `engine.end_turn()`, return — assistant turn NOT persisted and
    `router.end_turn` NOT called (the stale scope is cleaned up by the next
    `begin_turn` on that session). After a successful send: if
    `scope.is_cancelled()` → return WITHOUT persisting the assistant turn.
    Otherwise persist assistant turn (`channel=send_channel_id`,
    `content=rewritten`, `platform_message_id=<send_text return>`,
    `reply_to_message_id=thread_target`, `guild_id`), then
    `router.end_turn(scope)`, then `fm.end_turn()`, then engine bookkeeping:
    `notify_reply_sent()` only on judgment turns that actually replied,
    `end_turn()` always (applies tool-deferred activity starts).
20. **T20** Unread nudge (`_emit_unread_nudge`): no-op unless fm wired and
    text focus set. `fm.mark_nudge_pending()`, then publish a synthetic
    `discord.text` Event: `event_id=uuid4().hex`,
    `turn_id=f"unread-wake-{event_id}"`, `session_id=str(focus_channel)`,
    `sequence_number=0`, payload `{"familiar_id", "channel_id": focus_ch,
    "content": "[unread messages waiting elsewhere]", "author": None,
    "wake": True}`. Debounce/gating lives in `fm.should_wake` (05).
21. **T21** Log-line contracts (tests assert substrings, ANSI-interleaved):
    reply line tag `💬 Text` with `ch=#<name>(id)` (fallback `#<id>` without
    fm) and `srv=<guild>` omitted entirely when unknown — never `srv=None`;
    `📥 Staged`, `Activity ... suppressed`, `💤 Text ... decision ... silent`
    lines carry the same ch/srv fields resolved from the TURN's channel, not
    the focus channel.

### VoiceResponder (V)

1. **V1** `handle` dispatch: `voice.activity.start` → synchronous
   `_on_activity_start`; `voice.transcript.final` → `_spawn_final` (spawn,
   return immediately). This keeps the dispatcher pulling events so a fresh
   `activity.start` can cancel a prior turn parked at an LLM/TTS await
   (pinned by `TestDispatchLoop` — without the spawn, barge-in only lands
   after the old reply finishes).
2. **V2** Scope keys are per (session, user): `f"{session_id}:user:{user_id}"`
   when the payload carries an int `user_id`, else bare `session_id`
   (legacy). `session_id` is `voice:<channel_id>`.
3. **V3** `_on_activity_start` = `router.begin_turn(session_id=scope_key,
   turn_id=event.turn_id)` and NOTHING else. Deliberately no
   `tts.stop()`: Discord exposes one shared voice client per channel, so a
   global stop would cut a DIFFERENT user's in-flight reply (regression
   pinned: Bob's continuous activity.starts must not cut Alice's playback).
   Same-speaker barge-in works because the TTS player's poll loop checks
   `scope.is_cancelled()` and halts within one poll tick (sub-200 ms budget,
   pinned end-to-end).
4. **V4** Cross-user isolation: Alice's scope survives Bob's
   `activity.start` (different keys); same-user start cancels the prior
   scope (self-barge preserved).
5. **V5** `_spawn_final`: if an in-flight task exists for the same scope key
   and is not done, HARD-cancel it (`task.cancel()` — task-level, not scope)
   before spawning `asyncio.create_task(self._run_final(event),
   name=f"voice-final-{turn_id}")` and storing it in `_inflight[scope_key]`.
   The done-callback removes the map entry only if it still owns the slot
   (`_inflight.get(key) is task`) — a newer task may have replaced it.
   `_run_final` wraps `_on_final` and swallows `CancelledError` (expected on
   barge-in). Rust: `JoinHandle::abort()` + owner check on completion.
6. **V6** `_on_final` staleness gate: `router.active_scope(scope_key)` must
   exist AND `scope.turn_id == event.turn_id`, else drop (a FINAL from an
   older utterance after a newer `activity.start`). Also drop: session id
   not parseable as `voice:<int>`; empty `text`.
7. **V7** Author resolution: `member_resolver(channel_id, user_id)` is a
   SYNC call; any exception is swallowed (DEBUG log) and the turn records
   anonymously; resolver `None` likewise. Never block the reply path on
   Discord fetches.
8. **V8** Cold-cache instrumentation (`_emit_cold_cache_signals`) runs
   BEFORE the user turn is appended (so `prev_turn_at` reflects the real
   gap): reads the focus-stream summary
   (`get_summary(familiar_id, FOCUS_STREAM_CHANNEL_ID)`) and
   `recent(limit=1)` via the SYNC store, calls `log_signals(...)` (01).
   Best-effort: every exception swallowed at DEBUG.
9. **V9** User turn appended OUTSIDE the reply gate — observation is never
   gated; every speaker's turn lands even while the bot is mid-reply to
   someone else (pinned: both users' turns recorded when only one reply
   ships).
10. **V10** Per-channel reply gate: one `asyncio.Lock` per voice channel_id,
    lazily created — safe in Python because there is no await between the
    map miss and the insert; Rust must make the map access itself atomic
    (e.g. `DashMap::entry` or a `Mutex<HashMap>`), since two runtime threads
    could otherwise race rival locks. UNDER the lock: `set_rag_cue(text)`
    (mutates shared assembler state a concurrent pipeline would clobber
    mid-assemble) → assemble → stream+speak → empty-reply check →
    assistant-turn append → `router.end_turn(scope)` → `fm.end_turn()`.
    Purpose (pinned by a barrier-LLM test asserting `max_active == 1`): the
    second speaker's pipeline assembles only after the first commits its
    assistant turn, sees that reply in context, and can resolve `<silent>` —
    kills the production double-response. Playback is already serial on the
    shared voice client, so the wait adds no perceived latency.
11. **V11** Prompt build: `viewer_mode="voice"`, no guild in ctx. `tool_mode
    = registry and ctx_factory and llm.tool_calling_enabled` — voice does
    NOT consult `image_tools_enabled` (view_image is never in the voice
    registry). Head system message = join(system_prompt,
    `build_final_reminder(viewer_mode="voice", display_tz,
    tools_enabled=tool_mode)`); note the voice head INCLUDES the timestamp
    (asymmetric with text's `include_time=False`) — preserve. Head appended
    only if non-empty. Trailing reminder: `include_mode_instruction=True`
    ("You are speaking aloud…"), `tools_enabled`,
    `post_history_instructions`; when fm is wired also
    `focus_channel_id=channel_id` (the LIVE channel, not a focus pointer),
    `channel_names=fm.channel_names`,
    `guild_name=fm.guild_name_for(channel_id)`; when fm is None the focus
    kwargs are omitted entirely so the reminder stays byte-stable
    (backward-compat pinned: no "Your attention is currently on" line).
    No unread digest in voice. Ordering pinned: mode directive before
    `post_history_instructions`.
12. **V12** Bare voice streaming (`_stream_and_speak`): per delta —
    cancellation check FIRST (log `decision=preempted` if no decision was
    logged yet, return None); record `PHASE_LLM_FIRST_TOKEN` on the first
    delta (voice-budget recorder, 01); accumulate raw delta; while the gate
    is closed feed `SilentDetector` (True → log `decision=silent`, return
    None; False → gate opens + log `decision=respond` once). Sentences from
    `SentenceStreamer.feed(delta)` (09) buffer into `pending` and drain in
    arrival order only while the gate is open, re-checking
    `scope.is_cancelled()` before EACH `_speak`. Invariant: no sentence
    reaches TTS before the silent gate resolves; buffered sentences drain
    once it opens; on `<silent>` they are dropped.
13. **V13** Stream-end handling: if the stream ended with the detector still
    undecided but accumulated content non-whitespace (very short replies
    like `"just a fragment"`), open the gate and log `respond` — then flush
    `streamer.flush()` tail (if non-whitespace) and drain with per-sentence
    cancel checks. Whitespace-only/empty accumulation returns to `_on_final`
    which logs WARN `skip=empty_reply` and skips the assistant turn
    (Cartesia rejects empty transcripts). Stream exception → WARN
    `llm_stream_error`, return None. Exactly ONE decision line per turn:
    silent | respond | preempted (pinned; `decision_logged` guards
    double-logging on cancel-mid-speak).
14. **V14** `_speak`: skip whitespace-only chunks; record
    `PHASE_TTS_FIRST_AUDIO` before every call (recorder dedupes per turn);
    `await tts.speak(text, scope=scope)` — returns when playback finishes OR
    the scope is cancelled (player contract, 09). Silent turns must record
    NO tts-phase spans (pinned: `voice.ttft_to_tts` absent).
15. **V15** Tool path (`_stream_and_speak_with_tools`): same silent/budget
    gating inside `on_delta` (silent decision clears `pending` and stops
    speaking; detector latch makes later deltas no-ops). `on_before_tools`
    (after the assistant message is built, before handlers run): flush
    streamer tail + drain pending so no half-formed clause is spoken after
    tool execution; then the filler backstop — if `assistant.tool_calls`
    non-empty and spoken content is empty/whitespace, speak the next filler
    phrase (round-robin over `tool_filler_phrases`, index persists across
    turns) unless the tuple is empty or the scope is cancelled. Pinned:
    filler is spoken BEFORE the handler runs. Production wiring currently
    passes `tool_filler_phrases=()` (disabled 2026-06-25 as too chatty) —
    the mechanism must survive, the default prod config disables it.
    `on_iteration_end` persists intermediate assistant(tool_calls)+tool
    turns exactly like T14 but WITHOUT guild_id. After the loop: flush
    tail+drain; undecided-but-content → log respond; if cancelled and no
    decision logged → log preempted and return None; else return the full
    accumulated content (all iterations concatenated) for the assistant-turn
    write. No empty-completion retry in voice (text-only quirk).
16. **V16** `_on_final` post-stream: `reply is None or scope.is_cancelled()`
    → return (no assistant turn, no `end_turn`); `fm.end_turn()` only after
    a committed assistant turn (pinned: not called on silent). Voice has no
    activity-engine integration and no event-id dedup.
17. **V17** `wait_until_idle` gathers a snapshot of not-done in-flight tasks
    with `return_exceptions=True` (never raises); used by tests and graceful
    shutdown.

### HistoryWriter (H)

1. **H1** Drop rules: wrong topic; duplicate `event_id` (in-process set);
   payload not dict; `familiar_id` mismatch; `channel_id` not int; empty
   content. Then `append_turn(role="user", content, author, guild_id if int
   else None)`. Note: `author` is taken from the payload WITHOUT an
   isinstance check (unlike TextResponder) — a duck-typed payload seam.
2. **H2** Dedup on `event_id`: the same Event handled twice writes one row
   (pinned).

### DebugLoggerProcessor (D)

1. **D1** One INFO line per event on its injected topics; payload rendered
   `repr(...)` truncated at 160 chars, `-` when payload is None. Never
   republishes; `topics` stays a tuple.

### Projector registry (P)

1. **P1** `create_projectors` preserves `names` order, raises `ValueError`
   with the sorted valid-name list on unknown names, returns `[]` for `[]`.
2. **P2** Registration is global and last-write-wins; config load (02) also
   validates projector names against `known_projectors()` so typos fail at
   startup, not at wiring.
3. **P3** Knob threading pinned per factory (see tests): rolling_summary
   {turns_threshold, tick_interval_s}; rich_note {batch_size,
   tick_interval_s, participants_max} + familiar_display_name +
   dream_extraction_clause; people_dossier {tick_interval_s} +
   familiar_display_name; reflection {turns_threshold,
   max_reflections_per_tick, max_turns_per_tick, recent_facts_limit,
   tick_interval_s}; fact_supersede {batch_size, tick_interval_s,
   priors_max}; fact_embedding {embedder required}.

## Data formats

### `discord.text` payload (dict; producer: DiscordTextSource, 10)

| key | type | notes |
|---|---|---|
| `familiar_id` | str | responder filters on equality |
| `channel_id` | int | required |
| `guild_id` | int \| None | |
| `author` | `Author` \| None | identity dataclass (02) |
| `content` | str | `""` allowed only with `wake` |
| `message_id` | str | Discord snowflake as string |
| `reply_to_message_id` | str \| None | |
| `mentions` | tuple/list of `Author` | |
| `images` | dict[str,str] | `img_N` → URL; threads into `ToolContext.images` |
| `pings_bot` | bool (absent = False) | persisted on the user turn |
| `wake` | bool (absent = False) | synthetic nudge, no user content |

Synthetic wake event (exact): `event_id=uuid4().hex`,
`turn_id="unread-wake-"+event_id`, `session_id=str(focus_channel_id)`,
`sequence_number=0`, `parent_event_ids=()`, payload
`{"familiar_id", "channel_id": <focus>, "content": "[unread messages waiting
elsewhere]", "author": None, "wake": True}`.

### Voice payloads

- `voice.activity.start`: payload `{"user_id": int}` or `None`;
  `session_id="voice:<channel_id>"`; all events of one utterance share
  `turn_id`.
- `voice.transcript.final`: payload `{"text": str, "confidence": float,
  "start": float, "end": float, "speaker": None, "user_id": int | None}` —
  responder reads only `text` and `user_id`.

### Callback / storage shapes

- `send_text(channel_id: int, content: str, reply_to_message_id: str | None,
  mention_user_ids: tuple[int, ...]) -> str | None` (returned platform
  message id persisted on the assistant turn).
- Assistant tool turns: `tool_calls_json` column = `json.dumps` of the
  OpenAI-style accumulated tool_calls list
  (`[{"index":0,"id","type":"function","function":{"name","arguments":<json str>}}]`);
  tool rows carry `tool_call_id` + stringified content
  (`tool_content_as_text`). History schema itself is subsystem 03.
- LLM sentinel vocabulary in model output: `<silent>` (whole-reply gate),
  `[@DisplayName]` (ping), `[↩]` / `[reply]` / `[↩ <message_id>]` /
  `[↩ #<id>]` (threading). Inbound rendering counterpart
  (`[H:MM Name #id]` prefixes, `<@USER_ID>` → `[@Name]`) lives in 05; the
  two vocabularies must stay symmetric.
- `_BOT_OUTPUT_INSTRUCTIONS` — the "## Output controls" head addendum
  documenting the four rules above (do not start replies with the metadata
  prefix; ping via `[@DisplayName]`; `[↩]` optional threading;
  `[↩ <message_id>]` targeting with fallback). Byte-stable per build; part
  of the cacheable head.
- Ping resolver map: `display label -> canonical_key` where canonical keys
  are `"<platform>:<user_id>"`; only `discord:` keys with integer ids become
  real mentions.

## Config knobs

| Key | Default | Consumer |
|---|---|---|
| `[discord.text].respond_to_typing` | `true` | TypingInterruptHandler master switch (cancel + backoff) |
| `[discord.text].typing_backoff_initial_s` | `1.0` | backoff ladder start (positive float) |
| `[discord.text].typing_backoff_max_s` | `30.0` | ladder cap; config load rejects `max < initial` |
| `[tools].loop_max_iterations` | `5` | both responders' agentic-loop cap (positive int) |
| `display_tz` (top level) | `"UTC"` | trailing-reminder clock; IANA name validated at config load |
| `post_history_instructions` (top level) | `""` | appended deepest in trailing reminder; empty = omitted |
| `[providers.memory].projectors` | `DEFAULT_PROJECTORS` (5 names) | projector selection; unknown names → ConfigError at load |
| `[providers.memory.<name>].*` | per-knob-struct defaults (02) | threaded by factories (P3) |
| `[llm.<slot>].tool_calling` / `.image_tools` | off / off | gates tool_mode (text consults both, voice only tool_calling) |
| `tool_filler_phrases` | constructor `("one sec...", "hold on...", "checking...")` | NOT in TOML; production wiring passes `()` (disabled) |
| focus `unread_nudge_enabled` (`true`) / `nudge_debounce_seconds` (`30.0`) | | live in FocusManager (05); responder only calls `should_wake` |

Production slot wiring (10): voice responder ← `llm_clients["fast"]`, text ←
`llm_clients["prose"]`, all projectors ← `llm_clients["background"]`.

## Dependency edges

Imports (this subsystem → others):

| Module | Subsystem | Used for |
|---|---|---|
| `bus.envelope` (Event, TurnScope), `bus.topics`, `bus.protocols` (EventBus), `bus.router` (TurnRouter) | 01 | event shapes, scope cancel, begin/end/active_scope |
| `log_style`, `diagnostics.cold_cache.log_signals`, `diagnostics.voice_budget` (PHASE_*, recorder) | 01 | log formatting, instrumentation |
| `config` (DiscordTextConfig, MemoryProvidersConfig + knob structs), `identity.Author` | 02 | typing knobs, projector knobs, author payloads |
| `history.async_store.AsyncHistoryStore` (+ `.sync` HistoryStore), `history.store.FOCUS_STREAM_CHANNEL_ID` | 03 | append_turn, upserts, mentions, lookup_turn_by_platform_message_id, staged_channels, recent_distinct_authors, resolve_label, get_summary, recent |
| `context.assembler` (Assembler, AssemblyContext), `context.final_reminder.build_final_reminder`, `focus.FocusManager` | 05 | prompt assembly, trailing reminders, focus gating/staging |
| `llm` (LLMClient, Message, LLMDelta) | 08 | chat_stream / stream_completion contracts |
| `tools.registry` (ToolRegistry, ToolContext), `tools.loop` (agentic_loop, tool_content_as_text) — lazy imports | 08 | agentic loop |
| `sentence_streamer.SentenceStreamer`, `tts_player.protocol.TTSPlayer` | 09 | sentence chunking, speech |
| `activities.engine` (ActivityEngine, GateAction/GateDecision) | 11 | absence gate |
| projectors → `processors.{summary,fact_*,people_dossier,reflection}_worker`, `embedding.protocol.Embedder` | 07 / 04 | factory targets |

Imported by: `commands/run.py` and `bot.py` (10) — construction, dispatch
loops, projector spawn, `on_typing` wiring. Nothing else imports the
responders. Note: spec 01's module map placed `typing_interrupt.py` under 10;
this spec owns it under 06 per the port plan — the Rust crate boundary should
put the policy type with the responders and let the Discord shell (10) depend
on it, matching the Python import direction (`bot.py` → `typing_interrupt`).

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `tests/test_silence.py` | S1-S4: split-token detection, whitespace tolerance, mid-reply token is content, divergence, latching, whitespace-forever-pending | logic-portable |
| `tests/test_typing_interrupt.py` | Y1-Y6: cancel on user typing, self/unsubscribed/disabled no-ops, ladder 1→2→4, cap at max, user-message reset, wait_for_backoff sleeps ≈deadline / returns immediately when idle | logic-portable (timing asserts, generous tolerances) |
| `tests/test_text_responder.py` (1738 loc) | T2 drop rules; T4 backoff parks then replies + ladder reset; T5 scope ended after clean turn; T8 user turn persisted before stream (observing-LLM snapshot), pings_bot round-trip; T13 typing indicator enter/exit exactly once, skipped on silent/empty; silent sentinel skips send+assistant turn (user turn kept), leading whitespace tolerated, mid-reply sentinel is content; T16 empty/whitespace reply → no send; T6 account/guild-nick/mention upserts; T17 threading default-off, `[↩]`→trigger, explicit id, `#`-sigil strip, unknown-id fallback, `[reply]` alias, ping rewrite known/unknown; assistant turn persists sent id + thread target; `_strip_leaked_metadata_prefix` unit cases; T12 trailing reminder content/tz/post-history ordering, head vs trailing server-name split; T21 ch/srv log fields incl. fallbacks; T7/T9 activity gate suppress/judgment/normal + missed-ping capture + note_traffic + end_turn/notify_reply_sent bookkeeping | logic-portable; a few tests inject `responder._typing_handler` / `_focus_manager` post-construction and one patches `build_final_reminder` → needs-Rust-mock (constructor injection) |
| `tests/test_text_responder_tools.py` | T14 full agentic turn: history sequence user/assistant(tool_calls)/tool/assistant(text), tool_calls_json + tool_call_id shapes, final text posted; T15 empty-completion retry once and only once; images threaded into ToolContext; image_tools-only enters loop; loop_max_iterations=1 caps at one LLM call | logic-portable (scripted `stream_completion` stand-in; one test uses `__new__` construction hack → rewrite) |
| `tests/test_voice_responder.py` (1295 loc) | V3 activity.start registers/cancels scopes; full reply spoken + both turns persisted; V11 trailing reminder voice directive, time, post-history ordering, server naming with/without fm; silent sentinel (incl. split `<sil`/`ent>`) skips TTS+assistant turn; empty reply skips TTS; stale final dropped; budget spans present on speech, absent on silent; sentence streaming: ordered per-sentence speaks, first sentence before stream ends, tail flush without punctuation; barge-in mid-stream prevents speech, `decision=preempted` exactly once with no silent/respond for the preempted turn, mid-playback cut < 200 ms; V4 cross-user isolation incl. continuous-starts regression; V10 barrier-LLM proves max_active==1, loser resolves `<silent>`, one assistant turn, both user turns kept; V7 member resolver → author on turn, miss → anon; V1 dispatcher-unblocked (blocking-LLM + real bus subscribe); decision=respond logged exactly once; ch/srv fields | logic-portable; heart of the port's conformance suite — keep timing budgets (200 ms) as integration tests on tokio; MockTTSPlayer needs a Rust equivalent honoring scope-cancel polling |
| `tests/test_voice_responder_tools.py` | V15: spoken content reaches TTS before tool handler runs; filler phrase spoken before handler when iteration has tool_call + empty content; voice registry excludes view_image; text registry gates it on image_tools flag | logic-portable (registry tests belong to 08 but pinned here) |
| `tests/test_attentional_responders.py` | T8/T10 unfocused staged (consumed_at NULL, LLM not invoked), focused replies + fm.end_turn awaited once, no fm → unchanged; T20 nudge published to focused channel with mark_nudge_pending, none when should_wake false; T11 wake event replies without persisting user turn; V16 voice fm.end_turn on completed turn, not on silent; immediate shift_focus: silent peek moves focus, old-channel message then stages, shift+reply posts to new channel | mostly logic-portable; heavy MagicMock/AsyncMock on FocusManager and one `bus.publish` monkeypatch → needs-Rust-mock (trait-based FocusManager + bus test double) |
| `tests/test_history_writer.py` | H1/H2: persist with author, event-id dedup, other-familiar/empty-content ignored, name/topics surface | logic-portable |
| `tests/test_debug_logger_processor.py` | D1 via real bus subscribe loop; topics tuple | logic-portable |
| `tests/test_memory_projectors.py` | P1-P3: defaults set, order preserved, unknown raises, third-party registration, protocol shape, fact_embedding opt-in + embedder requirement, config-load validation of projector lists, every knob-struct → constructor-arg mapping | logic-portable (registry-restore finally block; knob asserts poke private fields → assert via behavior or expose getters in Rust) |

Also relevant: `tests/test_run_cmd.py` (wiring smoke), `tests/test_activity_engine.py`
(GateDecision producer, subsystem 11), `tests/test_final_reminder.py` (05),
`tests/test_sentence_streamer.py` (09), `tests/test_agentic_loop.py` (08).

## Rust port notes

- **Cancellation model.** `TurnScope` is a cooperative flag + waiter, not
  task cancellation → `tokio_util::sync::CancellationToken` maps 1:1
  (`cancel()` idempotent, `is_cancelled()`, `cancelled().await`). Keep the
  discipline: responders CHECK the token at loop tops / before each speak;
  only the voice per-speaker final task is ever hard-cancelled
  (`JoinHandle::abort()` in `_spawn_final`). Do not blanket-`select!` the
  token around whole pipelines — the Python code deliberately runs specific
  sections (persist-after-send, engine.end_turn) even when cancelled, and
  the exact checkpoint positions are observable (assistant turn dropped when
  cancelled between send and persist).
- **Single-thread assumptions.** Python relies on the GIL/event-loop for:
  lazy per-channel lock creation (V10), the `_inflight` map ownership dance
  (V5), `_seen` sets, the filler round-robin index, and detector state. On a
  multithreaded tokio runtime every one of these needs explicit
  synchronization (`Mutex<HashMap<..>>` or `DashMap`; the reply gate itself
  is `Arc<tokio::sync::Mutex<()>>` per channel). Alternatively pin the
  responder to a single-threaded actor/task and message it — arguably the
  cleaner redesign; preserve the observable serialization contracts (V10
  max_active==1, V9 user-turn-outside-gate) either way.
- **AsyncExitStack / typing indicator.** Rust has no async Drop. The typing
  indicator is an async context entered lazily mid-stream and exited on
  normal end, cancel-return, silent-return, and stream error. Model it as a
  guard struct whose `close()` is explicitly awaited on every exit path
  (or a `defer`-style helper that spawns the exit). The pinned invariant is
  entered==exited==1 for real replies, 0 for silent/empty.
- **Streams.** `chat_stream` (bare, yields `String`) and `stream_completion`
  (yields `LLMDelta`) become `impl Stream`; the agentic loop's `on_delta` /
  `on_before_tools` / `on_iteration_end` callbacks are natural async closures
  or a trait with default no-op methods. `getattr(llm,
  "image_tools_enabled", False)` duck-typing → a default-false trait method.
- **Blocking sync-store calls.** `_build_ping_resolver` and
  `_emit_cold_cache_signals` call the SYNC history store from async context
  (cheap SQLite reads in Python). In Rust route them through the same async
  store facade (03's spec) — do not replicate the sync backdoor.
- **Regex.** `regex` crate handles all three patterns; `↩` is multibyte —
  make sure the thread-marker alternation is on the char, not a byte. Port
  the exact patterns and the `count=1` head-anchored metadata strip.
  `SilentDetector` needs no regex; mind char-count vs byte-count in the
  `len(stripped) >= 8` rule (use `chars().count()` or compare against the
  prefix bytes).
- **Time.** Typing backoff uses a monotonic clock (`time.monotonic`) →
  `tokio::time::Instant`. `display_tz` rendering (chrono-tz) lives in 05's
  `build_final_reminder`; this subsystem only passes the string through.
- **Projector registry.** Import-time global registration is a Python-ism;
  replace with an explicit builder (`ProjectorRegistry::with_builtins()`)
  plus a `register(name, factory)` for extensions, and validate names at
  config load exactly like Python (same error text is test-visible).
  `MemoryProjector` → `trait MemoryProjector { fn name(&self) -> &str; async
  fn run(self) -> (); }` spawned as named tokio tasks.
- **Redesign candidates (behavior-preserving):** bound the `_seen` dedup
  sets (LRU); fix the T15 retry to pass the configured `max_iterations`
  (flag as deliberate deviation if changed); consider `router.end_turn` on
  the send-error path (Python leaks the scope until the next begin_turn —
  harmless but sloppy); `wait_for_backoff`'s single-sleep (Y5) could re-check
  the deadline in a loop, but the current semantic is what production runs —
  keep it unless the reviewer signs off.
- **Suggested crates:** tokio, tokio-util (CancellationToken), futures
  (Stream/StreamExt), async-trait, regex, serde_json, tracing, dashmap or
  parking_lot, chrono + chrono-tz (transitively via 05), uuid (wake events).
