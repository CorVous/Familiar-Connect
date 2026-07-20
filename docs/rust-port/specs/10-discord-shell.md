# 10-discord-shell — port spec

Source files: `src/familiar_connect/bot.py` (~1500 loc), `commands/{__init__,run,diagnose,version,example}.py` (~1010 loc),
`sources/{__init__,discord_text,discord_embed_text,twitch,voice}.py` (~430 loc), `cli.py` (117 loc), `__main__.py` (8 loc).
Total ~3060 loc.
Reference docs: `docs/architecture/overview.md` (§ Components, § Voice reply loop), `docs/architecture/security.md`.
Conformance oracle: `tests/test_bot_interactions.py`, `test_voice_intake.py`, `test_image_collect.py`,
`test_message_embeds.py`, `test_message_reactions.py` (dispatch half), `test_discord_text_source.py`,
`test_voice_source.py`, `test_twitch_source.py`, `test_discord_embed_text.py`, `test_cli.py`, `test_run_cmd.py`,
`test_diagnose_cmd.py`, `test_logging.py` (setup_logging half), `test_version.py`.

> Ownership note for the reviewer: spec 01's dependency table assigns `sources/voice.py` to 09 and
> `sources/twitch.py` to 11. This spec covers them fully (they were in its file list and are small);
> if 09/11 also spec them, THIS spec is authoritative for their queue→bus semantics and the 09/11
> versions must not contradict §B-VS/§B-TS below.

## Role

The Discord-facing shell: it owns the py-cord `discord.Bot` client (gateway + REST + voice), registers
the five slash commands and all gateway event handlers, converts inbound Discord traffic into bus events
via `DiscordTextSource`/`VoiceSource`, and exposes the `BotHandle` adapter so bus-only processors (06)
can post/typing/resolve-members without holding a bot reference. It also owns the `/subscribe-voice`
per-channel voice-intake pipeline (recording sink → per-user demux → per-user transcriber clones →
fan-in → `VoiceSource`), and the CLI: argument parsing, logging setup, the `run` wiring/composition root
(`_async_main`), signal handling, and teardown ordering. This is the composition root of the whole
program — nearly every other subsystem is constructed and wired here.

## Public API surface

### bot.py

```python
def create_bot(familiar: Familiar, *, focus_manager: FocusManager | None = None) -> BotHandle
```
Builds the `discord.Bot` with intents = default + `message_content` + `voice_states`
(NO privileged `members` intent — this drives the voice-member side cache, B-VM*).
Constructs `send_text`/`trigger_typing` closures over the bot, a `TypingInterruptHandler`
(config from `familiar.config.discord_text`, subscription check closure, late-bound
`bot_user_id` provider), the `resolve_member` closure, one `DiscordTextSource`, and registers
slash commands + events. Pure construction — no I/O, no login.

```python
@dataclass BotHandle:
    bot: discord.Bot
    send_text: SendText                                  # seam consumed by TextResponder (06)
    voice_runtime: dict[int, VoiceRuntime] = {}          # keyed by voice channel_id; read by DiscordVoicePlayer wiring (09)
    resolve_member: ResolveMember | None = None          # consumed by VoiceResponder (06)
    voice_members: dict[int, Author] = {}                # side cache, user_id -> Author
    trigger_typing: TriggerTyping | None = None          # consumed by TextResponder (06)
    typing_interrupt: TypingInterruptHandler | None      # consumed by on_typing + TextResponder (06)
    focus_manager: FocusManager | None = None            # (05)
    activity_engine: ActivityEngine | None = None        # (11), set later by _async_main
```

Callback seam types (these ARE the swappable-implementation seams; tests inject `AsyncMock`s):

```python
SendText     = (channel_id: int, content: str, reply_to_message_id: str | None,
                mention_user_ids: tuple[int, ...]) -> Awaitable[str | None]
                # returns posted platform message id as str, None on any send failure (never raises)
TriggerTyping = (channel_id: int) -> AsyncContextManager[None]
ResolveMember = (channel_id: int, user_id: int) -> Author | None   # synchronous, never awaits
```

```python
@dataclass VoiceRuntime:            # per-voice-channel pipeline state
    voice_client, sink, audio_queue: Queue[(int, bytes)], result_queue: Queue[TranscriptionResult],
    source: VoiceSource, pump_task, source_task,
    transcribers: dict[int, Transcriber], fanin_tasks: dict[int, Task],
    user_pump_tasks: dict[int, Task], idle_watchdog_task: Task | None,
    endpointers: dict[int, UtteranceEndpointer]
```

Free functions (all unit-tested directly, keep signatures):

```python
async def _defer_interaction(ctx) -> bool          # ACK within Discord's 3s window; NotFound -> warn, False
async def _reply(ctx, message: str) -> None        # ephemeral followup; NotFound swallowed
async def _start_voice_intake(*, handle, familiar, voice_client, channel_id) -> VoiceRuntime | None
async def _stop_voice_intake(*, handle, familiar, channel_id) -> None
async def _prefetch_voice_member(*, handle, channel_id, user_id) -> None
async def _on_recording_done(sink, *args) -> None  # MUST be a coroutine function (pycord threadsafe-schedules it)
async def ingest_event(*, source, familiar, channel_id, guild_id, author, text,
                       message_id=None, reply_to_message_id=None, mentions=(),
                       images=None, pings_bot=False) -> None    # thin seam over source.publish_text
def message_pings_bot(message, bot_user_id: int | None) -> bool
def compose_content_with_embeds(content: str, embeds: Iterable[object]) -> str
def collect_images(*, content, attachments, embeds) -> tuple[str, dict[str, str]]
def apply_message_edit(*, store: HistoryStore, familiar_id, is_subscribed, channel_id,
                       message_id, content, embeds) -> None      # sync, takes SYNC store
def apply_reaction_delta(*, store, familiar_id, is_subscribed, channel_id, message_id, emoji, delta) -> None
def apply_reaction_clear(*, store, familiar_id, is_subscribed, channel_id, message_id, emoji=None) -> None
def _emoji_repr(emoji: PartialEmoji) -> str
def build_activity_presence_cb(handle) -> async (status: str, label: str | None) -> None
def _register_dm_channel(handle, familiar, channel_id) -> None
```

Constants (byte-exact, test-pinned):

```python
DM_BOT_DISCLAIMER = ("⚠ This is a bot, and content may not be isolated solely to this channel- "
                     "treat messages in this conversation as if they were public.")   # verbatim per PR #176 — do not reword
DM_BOT_DISCLAIMER_DELETE_EMOJI = "✅"
DM_BOT_DISCLAIMER_DISMISS_HINT = "\n\n_React ✅ to delete this disclaimer._"
```

### sources/

```python
class DiscordTextSource:
    name = "discord-text"
    def __init__(*, bus: EventBus, familiar_id: str)
    async def publish_text(*, channel_id, guild_id, author, content, message_id=None,
                           reply_to_message_id=None, mentions=(), images=None,
                           pings_bot=False) -> Event

class VoiceSource:
    name = "voice"
    def __init__(*, bus, familiar_id, voice_channel_id, queue: Queue[TranscriptionResult])
    def record_vad_end(*, user_id: int, t: float | None = None) -> None   # sync, callable from any task
    async def run() -> None        # forever; cancellation is the only exit

class TwitchSource:
    name = "twitch"
    def __init__(*, bus, familiar_id, queue: Queue[object])
    async def run() -> None        # forever; cancellation is the only exit

def format_embeds(embeds: Iterable[Any]) -> str    # duck-typed embed flattener (sources/discord_embed_text.py)
```

### cli.py / commands/

```python
def main() -> int                                   # entry; also `python -m familiar_connect`
def create_parser() -> argparse.ArgumentParser
def setup_logging(verbose: int = 0, level: str | None = None) -> None
# each commands/<name>.py exports add_parser(subparsers, common_parser) and a run(args)->int handler
def run(args) -> int                                # commands/run.py
async def _async_main(token: str, familiar: Familiar) -> None
def _install_shutdown_handlers(stop: asyncio.Event) -> Callable[[], None]
async def _wait_for_shutdown(stop) -> None          # raises _GracefulShutdown when stop set
def _resolve_familiar_root(args) -> Path            # --familiar flag > FAMILIAR_ID env; ValueError otherwise
def _default_assembler(familiar, *, window_size, budget, silence_gap_fold_seconds=0.0,
                       embedder=None, focus_manager=None) -> Assembler
def _build_activity_engine(familiar, *, focus_manager, handle) -> ActivityEngine | None
def _first_voice_client(handle) -> discord.VoiceClient | None
def load_opus() -> None
def diagnose(args) -> int                           # commands/diagnose.py
```

## Behaviors & invariants

### Slash commands (B-SC)

1. Every handler ACKs the interaction immediately (`defer(ephemeral=True)`) to claim Discord's 3 s
   response window. A `NotFound (10062)` from defer is benign: log warning
   (`stale_interaction` + command name), return `False`, but the handler still performs its action.
2. `_reply` sends the confirmation as an ephemeral followup; `NotFound` is swallowed (action already
   ran — only the confirmation is lost). Neither helper may ever propagate `NotFound`.
3. `/subscribe-text`: defer → if `ctx.channel_id is None` reply "No channel in context." → else
   `subscriptions.add(channel_id, text, guild_id=ctx.guild_id)` (persisted) → reply
   "Listening in this channel.".
4. `/unsubscribe-text`: defer → channel-None guard as above → `subscriptions.remove(channel_id, text)` →
   "No longer listening here.".
5. `/subscribe-voice`: defer → caller must be in a voice channel (`ctx.author.voice.channel`), else
   "You must be in a voice channel." → `channel.connect(cls=DaveVoiceClient)`; on `DiscordException`
   warn + "Could not join voice." → `_start_voice_intake(...)` (returns `None` when no transcriber:
   bot stays joined for TTS playback only) → `subscriptions.add(channel.id, voice, guild_id)`
   (persisted) → reply `f"Joined {channel.name}."` plus suffix
   `" (playback only — no transcriber)"` when intake is None. Subscription is added even when intake
   fails to build.
6. `/unsubscribe-voice`: no guild → immediate `ctx.respond("Not in a guild.", ephemeral)`;
   no `subscriptions.voice_in_guild(guild.id)` row → "Not in a voice channel here." Otherwise
   defer (errors suppressed) BEFORE teardown — per-user WS closes can exceed the 3 s token deadline —
   then `_stop_voice_intake` → `guild.voice_client.disconnect(force=False)` (suppressed) →
   `subscriptions.remove(channel_id, voice)` → followup "Left voice channel." (suppressed).
7. `/diagnostics`: defer → `get_span_collector().summary()` rendered via `render_summary_table` (01);
   when a focus manager is wired, append `"\nFocus: text=#<id>|unset voice=#<id>|unset"` and, when
   `history_store.staged_channels(familiar_id)` is non-empty,
   `"\nUnreads: #<ch> (<unread>), …"` sorted by channel id.

### on_message ingest (B-OM)

8. Guard order is load-bearing (pinned by 4 tests): (a) drop own echo
   (`author.id == familiar.bot_user_id`), (b) drop any `author.bot`, (c) THEN the DM-allowlist /
   subscription gates. A bot user id colliding with the allowlist must never be admitted (no DM loops).
9. DM (guild is None): admit only `author.id in config.dm_allowlist`. On the FIRST admitted DM per
   user per process (in-memory `set`, deliberately not persisted — restart may re-send): post
   `DM_BOT_DISCLAIMER + DM_BOT_DISCLAIMER_DISMISS_HINT` to the channel, remember the sent message in
   an in-memory `disclaimer_messages: dict[message_id, Message]`, then pre-seed the ✅ reaction on it.
   Then `_register_dm_channel`: `subscriptions.add(channel_id, text, guild_id=None, persist=False)`
   (row must NEVER reach the sidecar file, even across later persisted writes),
   `fm.guild_names[channel_id] = "Private Message"` (`focus.PRIVATE_MESSAGE_GUILD_NAME`), and seed
   text focus via `fm.set_focus_immediately(channel_id, "text")` ONLY when `fm.get_focus("text")` is
   None (never steal an existing focus). Idempotent per DM.
10. Guild message: drop unless a text subscription exists for the channel.
11. `reply_to = str(message.reference.message_id)` when a reference with message_id is present, else None.
12. `mentions` payload = `Author.from_discord_member(u)` for each `message.mentions` entry with
    `bot` falsy. `pings_bot = message_pings_bot(...)` is carried separately precisely because the
    bot is filtered out of `mentions` and reply-pings have no `<@id>` in content.
13. `message_pings_bot`: True iff `bot_user_id` is not None and some entry of `message.mentions` has
    that id. py-cord puts both `<@id>` mentions AND reply-ping targets in `mentions`; role/@everyone
    mentions and bare name-strings never count.
14. Content pipeline (order matters): `text = compose_content_with_embeds(message.content, message.embeds or ())`
    then `text, images = collect_images(content=text, attachments=..., embeds=...)`. Inbound embeds
    are usually empty (Discord unfurls async and fires `on_message_edit` later) but pre-cached
    unfurls do arrive populated and must merge here.
15. Publish via `ingest_event` → `DiscordTextSource.publish_text` (payload in § Data formats). The
    publish is awaited inline in the gateway handler; `discord.text` is an unbounded topic (01) so
    this cannot block on backpressure.

### Message edits / reactions (B-RX)

16. `on_message_edit`: skip own/bot authors; act only when `after.embeds` is non-empty AND differs
    from `before.embeds` (pure text edits are NOT tracked). Dispatch to `apply_message_edit` with the
    SYNCHRONOUS store facade (`familiar.history_store.sync`) — the handler body does no awaiting.
17. `apply_message_edit` no-ops when: channel not text-subscribed, `format_embeds(embeds)` is empty,
    or no stored turn matches `platform_message_id` (bot started after the message). Otherwise writes
    `compose_content_with_embeds(content, embeds)` via `store.update_turn_content_by_message_id`.
18. `on_raw_reaction_add`: if `payload.message_id` is a live disclaimer message: a reaction from a
    non-bot user whose `_emoji_repr(payload.emoji) == "✅"` deletes the disclaimer (pop from dict +
    `message.delete()`); the bot's own pre-seeded checkmark and any other emoji are ignored; in ALL
    disclaimer cases nothing is written to history (no orphan reaction rows). Otherwise
    `apply_reaction_delta(..., delta=+1)`.
19. `on_raw_reaction_remove`: disclaimer messages → total no-op (un-reacting the pre-seeded ✅ must
    not write a −1 row). Otherwise delta −1. `on_raw_reaction_clear` → `apply_reaction_clear` (all
    emoji); `on_raw_reaction_clear_emoji` → scoped to `payload.emoji`.
20. `apply_reaction_delta`/`apply_reaction_clear` check `is_subscribed(channel_id)` FIRST (never
    write rows that will never be read); an empty `_emoji_repr` short-circuits delta. `message_id`
    is stringified before hitting the store. Store-side floor-at-zero / upsert semantics are 03's.
21. `_emoji_repr`: unicode emoji → the character itself (`emoji.id is None` → `emoji.name or ""`);
    custom → `<:name:id>`; animated custom → `<a:name:id>`; custom with `name None` → `""`.

### Typing / voice-state events (B-EV)

22. `on_typing`: forwarded to `typing_interrupt.notify_typing(channel_id=..., user_id=int(user.id),
    is_bot=bool(user.bot))` only when the handler is wired and `channel.id` is an int. Policy itself
    lives in 06.
23. `on_voice_state_update`: ignore self; ignore when `after.channel` is None (leaves); act only when
    the guild's voice subscription exists AND matches `after.channel.id`; then cache
    `voice_members[member.id] = Author.from_discord_member(member)`. This event + the audio-pump
    prefetch are the only two writers of the side cache.

### on_ready / presence (B-PR)

24. `on_ready` sequence (order pinned): set `familiar.bot_user_id = bot.user.id`; bulk-populate
    `fm.channel_names` / `fm.guild_names` from every channel of every guild (only channels with a
    `name` attr); install `fm.on_shift = lambda: _sync_presence(bot, fm)`; await `_sync_presence`;
    THEN `activity_engine.resync_presence()` when an engine is wired. The resync must come after the
    focus sync so a mid-activity away presence (idle/dnd) wins over the online focus presence —
    `engine.start()` runs pre-login and its presence call was dropped by the ready guard.
25. `_sync_presence`: custom-status activity with state `f"✨ {guild} -> {channel}"` when both focus
    labels exist, `f"✨ {channel}"` when only channel, else `None`; `status=online` always.
26. `build_activity_presence_cb` returns an async `(status, label)` callback: no-op while
    `not bot.is_ready()`; `"idle"`/`"dnd"` → matching `discord.Status` + `CustomActivity(label or "away")`;
    anything else ("online") → `_sync_presence` when a focus manager exists, else plain
    `change_presence(status=online)`.

### send_text / trigger_typing (B-TX)

27. `send_text` NEVER raises: channel via `bot.get_channel` then `bot.fetch_channel` fallback
    (cache misses right after startup); fetch failure or non-`Messageable` → warn + `None`.
    Mentions are always restricted: `AllowedMentions(everyone=False, roles=False,
    users=[Object(id) for mention_user_ids])` — stray `<@…>` markers in content must not ping.
    `reply_to_message_id` parsed as int (unparseable → send without reference); reference built with
    `fail_if_not_exists=False` (deleted parent must not fail the send). `DiscordException` on send →
    warn + `None`. Success → `str(sent.id)`.
28. `trigger_typing(channel_id)` is an async context manager factory: cache-miss / non-messageable
    channel or a `DiscordException` from `channel.typing()` → degrade to a bare `yield` (reply still
    posts, indicator just doesn't show); otherwise wrap the yield in the typing CM so the indicator
    persists for the body's duration. A fresh CM per call.

### Voice-member resolution (B-VM)

29. `resolve_member` is synchronous and must never do I/O (audio path cannot tolerate REST
    round-trips): side cache → `guild.get_member` (gateway cache) → `None` (caller records the turn
    anonymously). A `get_member` hit is written back into the side cache.
30. `_prefetch_voice_member` (async, fire-and-forget): bail on cache hit; resolve channel → guild
    (channels without `guild` → return); `get_member` then REST `fetch_member`; ALL exceptions
    swallowed (debug log) — a missing name is recoverable, a crashed prefetch task is not.

### Voice intake pipeline (B-VI) — the concurrency heart

31. `_start_voice_intake` returns `None` when `familiar.transcriber` is None (playback-only join) and
    is idempotent per channel_id (second call returns the existing runtime; `start_recording` called
    exactly once).
32. Topology per channel: `RecordingSink(loop, audio_queue)` (09) attached via
    `voice_client.start_recording(sink, _on_recording_done)`. `_on_recording_done` is a no-op but
    MUST be a coroutine function — pycord schedules `callback(sink, *args)` with
    `asyncio.run_coroutine_threadsafe` from its recording thread; a plain function raises
    `TypeError`. `audio_queue: Queue[(user_id, mono_pcm_bytes)]` is fed cross-thread by the sink.
33. `familiar.transcriber` is a TEMPLATE, never started/stopped by this subsystem's per-channel
    lifecycle: `template.clone()` is called lazily, once per Discord user_id, on that user's first
    audio chunk. Per-user streams exist to kill mixed-stream endpointing (one speaker's pause
    finalizing another's sentence) and inherit exact attribution from Discord's per-SSRC delivery.
34. Router task (`_route_audio`, name `voice-pump-{ch}`): `await audio_queue.get()` → stamp
    `last_audio_time[user_id] = monotonic()` → lazily create per-user unbounded queue + pump task
    (name `voice-user-pump-{ch}-{uid}`) → `q.put_nowait(pcm)`. INVARIANT: no per-chunk awaits after
    the get — a slow user must not head-of-line-block the demux (test: slow user's blocked
    `send_audio` while a fast user's two chunks still arrive in order).
35. Per-user pump: first awaits `_ensure_transcriber(user_id)`; a start failure logs a warning and
    unregisters the pump/queue (stream retried on next audio via lazy re-create). Main loop, FIFO per
    user: `clone.send_audio(pcm)` then (when endpointer present) `ep.feed_audio(pcm)`, each with
    exceptions swallowed; a broken transcriber is recovered by the idle watchdog closing and lazy
    reopen.
36. Idle-flush fallback (Discord client VAD halts RTP during silence, so no downstream endpointer
    ever sees trailing silence): the pump arms an idle timer ONLY while `dirty` (audio sent since
    last flush) — `wait_for(q.get(), timeout=idle_flush_s)`; on timeout, flush:
    `ep.force_complete_if_pending()` when a local endpointer owns endpointing (must NOT call
    `finalize` directly — Smart Turn's hold-through-pause stays authoritative), else
    `clone.finalize()`; then `dirty = False` and loop (re-arms on next chunk; a second idle gap
    flushes again). When not dirty, block on a plain `get()` so long silences cost nothing.
    `idle_flush_s = detector.idle_fallback_s` when an endpointer exists, else module-level
    `DEFAULT_IDLE_FINALIZE_S` (0.5 s, imported from `stt.deepgram`; tests monkeypatch the bot-module
    binding — keep it injectable).
37. `_ensure_transcriber(user_id)`: return existing; else stamp `last_audio_time`; `clone()`; when a
    local turn detector is configured AND the clone exposes `endpointing_ms`, set it to 10 (drive
    Deepgram's hosted endpointer to near-zero so `Finalize` from the local chain rules; duck-typed —
    no-op for backends without the field); `clone.start(per_user_queue)`; when detector set, build
    `detector.make_endpointer(on_turn_complete=cb)` where cb = `source.record_vad_end(user_id)`
    FIRST (so the buffered timestamp reaches VoiceSource before the resulting stt_final), then
    `transcribers[uid].finalize()` (suppressed; skip if stream already closed); spawn fan-in task
    (`voice-fanin-{ch}-{uid}`); spawn fire-and-forget `_prefetch_voice_member` task
    (`voice-prefetch-{ch}-{uid}`), exactly one prefetch per new user_id.
38. Fan-in per user: infinite `result = await q.get(); result.user_id = user_id;
    await result_queue.put(result)` — tags every TranscriptionResult with its Discord user.
39. Idle watchdog (name `voice-idle-watchdog-{ch}`): created only when
    `getattr(template, "_IDLE_CLOSE_S", 0.0) > 0` (duck-typed template attribute; Deepgram/Parakeet/
    faster-whisper default 30.0, configurable). Scan interval `max(idle_close_s / 4, 0.01)`; any user
    with `now - last_audio_time > idle_close_s` gets `_close_user_stream(reason="idle")`. Rationale:
    Deepgram closes silent streams server-side regardless of KeepAlive; closing proactively avoids
    reconnect+replay. Stream reopens lazily on next audio.
40. `_close_user_stream`: pop ALL per-user map entries first (transcribers, fanin, pump, queue,
    endpointer, last_audio_time), then cancel+await pump (suppress), cancel+await fanin (suppress),
    then `clone.stop()` (suppress). Order: producers before the transcriber.
41. `_stop_voice_intake` (unsubscribe/teardown): pop runtime (absent → silent no-op);
    `voice_client.stop_recording()` (suppress); cancel pump, source, watchdog, every user pump,
    every fanin; await pump/source/watchdog with suppress; `gather(*user_pumps,
    return_exceptions=True)`; `gather(*fanins, ...)`; finally `gather(*(clone.stop() for clones))` —
    per-user WS closes run IN PARALLEL, sequential closes would multiply unsubscribe latency past
    the 3 s interaction deadline. The template transcriber is NOT stopped here (only `_async_main`'s
    final teardown stops it).
42. All queues in this pipeline are unbounded; drop-oldest backpressure at the raw-audio boundary is
    owned upstream by the sink/transcriber (09). If a pump wedges, its queue grows until the idle
    watchdog closes the stream.

### Bus sources (B-DS / B-VS / B-TS)

43. `DiscordTextSource.publish_text`: `event_id = f"discord-text-{uuid4().hex[:12]}"`;
    `turn_id == event_id` (source events root their own turn — test-pinned);
    `session_id = f"discord:{channel_id}"`; `sequence_number` monotonic from 1 per source instance;
    `parent_event_ids = ()`; `timestamp = now(UTC)`; topic `discord.text`; returns the Event.
    Omitted optionals default to `message_id=None, reply_to_message_id=None, mentions=(),
    images={}, pings_bot=False` in the payload.
44. `VoiceSource` state machine is keyed per `user_id` (`None` = legacy unattributed slot) —
    `dict[user_id, current_turn_id]`. On a result for an idle user: mint
    `turn_id = f"voice-{uuid4().hex[:12]}"`, publish `voice.activity.start` payload
    `{"user_id": user_id}`. Two interleaved speakers each get their own start + turn_id; a single
    shared slot would drop the second start (test-pinned).
45. Per result, after ensuring a turn: drain `_pending_vad_end.pop(user_id)` (never for `None` key)
    into `voice_budget_recorder.record(turn_id, PHASE_VAD_END, t)` BEFORE any other phase stamp.
    `record_vad_end(user_id, t=None)` buffers `t or perf_counter()`; latest fire wins; a different
    user's final must not consume it (test-pinned).
46. Final result: stamp `PHASE_STT_FINAL` BEFORE publishing (recorder must see stt_final ahead of
    the responder's llm_first_token); publish `voice.transcript.final` payload
    `{"text","confidence","start","end","speaker","user_id"}`, then `voice.activity.end`
    `{"user_id"}`, then clear the user's turn slot (next utterance gets a fresh turn_id). Partial:
    publish `voice.transcript.partial` `{"text","confidence","user_id"}`, turn stays open. Exactly
    ONE activity.start per utterance regardless of partial count; all events of one utterance share
    turn_id. `event_id = f"voice-{seq:08d}"`, `session_id = f"voice:{voice_channel_id}"`.
47. `TwitchSource`: `event_id = f"twitch-{uuid4().hex[:12]}"`, `turn_id == event_id`,
    `session_id = f"twitch:{familiar_id}"`, topic `twitch.event`, payload
    `{"familiar_id": ..., "twitch": <raw queue object>}`. Both queue sources' `run()` loops are
    infinite; task cancellation is the only clean exit (tests assert CancelledError propagates).
48. `format_embeds` (duck-typed — any object with the attributes works): each embed renders as
    `[embed]\n` + lines: header = " — "-join of `(provider.name)` (parenthesized), `author.name`,
    `title` (suppressed when identical to author.name — Tumblr/Bluesky cards); `description`;
    per-field `name: value` (or the non-empty one); footer as `— text`. All values stripped;
    None/blank → skipped. No meaningful lines but a `url` → `[embed]\n[link: {url}]`; nothing at all
    → `""`. Multiple embeds joined by blank line; blank ones dropped.
49. `compose_content_with_embeds`: empty embed text → content unchanged; empty content → embed text
    alone (no leading blank line); else `f"{content}\n\n{embed_text}"`.
50. `collect_images` scan order: image attachments (content_type startswith `image/` OR filename
    extension in {png,jpg,jpeg,gif,webp,bmp,tif,tiff}), then `embed.image` (prefer `proxy_url` over
    `url` — Discord's re-hosted copy fetches more reliably), then inline URLs matching
    `https?://\S+\.(?:png|jpe?g|gif|webp|bmp|tiff?)(?:\?\S+)?` (case-insensitive). Ids assigned
    `img_0, img_1, …` in discovery order; dedupe by exact URL (first source wins). Markers
    `[image: img_N (filename)]` appended one-per-line after content (`content\nmarkers`, or markers
    alone when content empty); filename falls back to last URL path segment (query stripped), then
    `"image"`/`"embed-image"`. No images → `(content, {})` unchanged.

### CLI / run loop (B-CLI)

51. `main()`: `load_dotenv()` first; no subcommand → print help, exit 0; `setup_logging(verbose=args.verbose)`;
    dispatch `args.func(args)`; any exception → `logger.exception` + exit 1. `--version` prints
    `"{prog} {__version__}"`. Prog name from installed package metadata, fallback
    `"familiar-connect"`.
52. `setup_logging`: verbose 0→WARNING, 1→INFO, ≥2→DEBUG; explicit `level` name overrides verbose
    (unknown name → `ValueError`, case-insensitive); reconfigures with `force=True`; the
    `familiar_connect` package logger is pinned to `min(level, INFO)` so package INFO stays visible
    even at root WARNING (log_style formatter itself is 01).
53. `commands/run.run(args)` failure ladder — each returns exit 1 with a specific error log:
    missing `DISCORD_BOT` env; unresolvable familiar (`--familiar` flag beats `FAMILIAR_ID` env;
    directory `data/familiars/<id>` must exist); `ConfigError` from
    `load_character_config(root/"character.toml", defaults_path=root.parent/"_default"/"character.toml")`;
    missing `OPENROUTER_API_KEY`; `KeyError` from `create_llm_clients` (missing LLM slot).
    Degrade-not-fail: `create_tts_client` / `create_transcriber` `ValueError` → warn + None (text
    path still works); local turn detector built only when
    `config.turn_detection.strategy == "ten+smart_turn"`, any exception → warn + None. Then
    `Familiar.load_from_disk(root, llm_clients=, tts_client=, transcriber=, local_turn_detector=)`,
    `load_opus()`, `asyncio.run(_async_main(token, familiar))`. `except* discord.errors.LoginFailure`
    → exit 1 with token-regeneration hint. `KeyboardInterrupt` escaping `asyncio.run` (signal landed
    before the handler armed, or platform without `add_signal_handler`) → quiet exit 0, no traceback.
    Load order is test-pinned: config → llm clients → tts → transcriber → Familiar.
54. `load_opus`: no-op when already loaded; `ctypes.util.find_library("opus")`, else first existing
    path from the hardcoded fallback list (Homebrew/MacPorts/deb/rpm/arch/Windows); none found →
    warning with install hints (voice playback disabled, everything else runs).
55. `_async_main` wiring order (composition root): `bus.start()` → `FocusManager(...)` +
    `initialize()` → startup focus defaults (first subscribed channel per modality, only when that
    modality's focus is unset) → `create_bot(familiar, focus_manager=)` → `create_embedder` →
    voice + text assemblers (`_default_assembler` with per-tier `config.budget_for("voice"/"text")`,
    text also gets `silence_gap_fold_seconds`) → tts_player = `DiscordVoicePlayer(get_voice_client=
    lambda: _first_voice_client(handle))` when `familiar.tts_client` else `LoggingTTSPlayer` →
    `AlarmScheduler` → `_build_activity_engine` (None when `activities.toml` missing/empty catalog);
    `handle.activity_engine = engine` → tool registries: voice registry NEVER contains `view_image`;
    text registry adds image tools only when the `prose` slot exists and has `image_tools` →
    per-turn `ToolContext` factory (description LLM = `llm_clients["__image_description__"]`, bound
    for text kind only) → `AlarmWaker` → `VoiceResponder` (llm slot `fast`,
    `tool_filler_phrases=()` — deliberately disabled 2026-06-25) + `TextResponder` (slot `prose`) →
    projectors from `config.memory_providers.projectors` → `await alarm_scheduler.start()` (pending
    alarms count down before traffic) → `await activity_engine.start()`.
56. Run phase: `stop = asyncio.Event()`; `_install_shutdown_handlers(stop)`; a `TaskGroup` runs
    named tasks: `shutdown-supervisor` (`_wait_for_shutdown`), `debug-logger`
    (DebugLoggerProcessor over topics `discord.text`, `twitch.event`, `voice.activity.start`,
    `voice.transcript.final`), `voice-responder`, `text-responder`, `alarm-waker` (each a
    `async for event in bus.subscribe(proc.topics): await proc.handle(event, bus)` loop), one task
    per projector (named by projector), and `discord-bot` (`bot.start(token)`). Any task failure
    cancels all siblings (TaskGroup semantics). `except* _GracefulShutdown` → log clean shutdown.
57. `finally` teardown ORDER (test-pinned; runs on both failure and graceful paths):
    remove signal handlers → `bot.close()` (first, so py-cord's aiohttp session doesn't leak past
    loop shutdown; suppressed) → `familiar.transcriber.stop()` when set (template; suppressed) →
    `alarm_scheduler.shutdown()` (suppressed) → `activity_engine.stop()` when set (suppressed) →
    `router.shutdown()` (sync, NOT suppressed) → `await bus.shutdown()` → `await client.close()` for
    every LLM client (once per slot, even when slots share a client object).
58. Signal handling: SIGINT + SIGTERM via `loop.add_signal_handler`. FIRST signal: log
    "draining — signal again to force", set `stop` (the supervisor then raises `_GracefulShutdown`,
    unwinding the group so `finally` runs in normal, non-cancelling task state). SECOND signal:
    remove the installed handlers, restoring OS defaults so a wedged shutdown stays force-killable.
    Platforms without `add_signal_handler` (Windows Proactor, non-main thread): skip installation
    (debug log) — `run()`'s `KeyboardInterrupt` fallback covers it. The returned remover is
    idempotent.
59. `_default_assembler` layer order is test-pinned (prompt-cache stability descending):
    CharacterCard, OperatingMode (voice-terse/text-verbose strings hardcoded here), Lorebook,
    ConversationSummary, Reflection, PeopleDossier, RagContext, RecentHistory. Rag must be
    second-to-last with RecentHistory last. Layer construction parameters flow from `budget` /
    `config`; channel/guild name resolvers bind over the focus manager's live maps (populated at
    on_ready).
60. `_build_activity_engine`: loads `root/activities.toml` merged over
    `root.parent/_default/activities.toml`; disabled/empty catalog → None (zero behavior change).
    `voice_active_fn = lambda: bool(handle.voice_runtime)` — a live voice runtime refuses activity
    start; `bot_user_id` late-bound via lambda (on_ready fills it after login).
61. `commands/diagnose`: aggregates `span=<name> … ms=<int> … status=<word>` markers from log files
    (regex tolerates interleaved ANSI `\x1b[Nm` codes and arbitrary intervening tokens, DOTALL);
    per-span summary `{count, p50, p95, last_ms}` with linear-interpolation percentiles
    (single value → itself; rank = pct/100·(n−1)); multiple files concatenate; `-` reads stdin;
    unreadable file → error log, continue with the rest; prints `render_summary_table(summary)` +
    `"\n"` to stdout; returns 0. `commands/version`: styled one-liner with `__version__`; returns 0.
    A `sleep` subcommand must NOT exist (removed; test-pinned).

## Data formats

### `discord.text` bus payload (exact keys)

```python
{
  "familiar_id": str,
  "channel_id": int,
  "guild_id": int | None,          # None for DMs
  "author": Author,                # rich object, not serialized (02)
  "content": str,                  # body + "\n\n" embeds + "\n" image markers
  "message_id": str | None,        # Discord snowflake as string
  "reply_to_message_id": str | None,
  "mentions": tuple[Author, ...],  # non-bot user mentions only
  "images": dict[str, str],        # "img_N" -> URL, {} when none
  "pings_bot": bool,
}
```
Envelope: `event_id = turn_id = "discord-text-" + 12 hex chars`, `session_id = "discord:{channel_id}"`,
`parent_event_ids = ()`, tz-aware UTC timestamp, per-instance 1-based `sequence_number`.

### Voice bus payloads

- `voice.activity.start` / `voice.activity.end`: `{"user_id": int | None}`
- `voice.transcript.partial`: `{"text": str, "confidence": float, "user_id": int | None}`
- `voice.transcript.final`: `{"text", "confidence", "start", "end", "speaker", "user_id"}`
- Envelope: `event_id = f"voice-{seq:08d}"`, shared per-utterance `turn_id = "voice-" + 12 hex`,
  `session_id = "voice:{channel_id}"`.

### `twitch.event` payload

`{"familiar_id": str, "twitch": <opaque TwitchWatcher event object>}`; `turn_id == event_id
= "twitch-" + 12 hex`; `session_id = "twitch:{familiar_id}"`.

### In-content text formats

- Embed block: `"[embed]\n" + lines`, blocks joined by `"\n\n"`; image-only fallback
  `"[embed]\n[link: {url}]"`; footer line `"— {text}"`; header parts joined by `" — "`, provider
  wrapped `"({name})"`.
- Image marker: `"[image: img_{N} ({filename})]"`, one per line, appended after content.
- Presence state strings: `"✨ {guild} -> {channel}"` / `"✨ {channel}"`.
- DM disclaimer: constants in § Public API (byte-exact; core wording immutable per PR #176 review).
- Diagnose input: log lines with `span=`, `ms=`, `status=` KV tokens (ANSI-tolerant);
  output = `render_summary_table` (01) columns count/p50/p95/last_ms.

### Audio framing

`audio_queue` items are `(discord_user_id: int, mono_pcm: bytes)` — the `RecordingSink` (09) has
already downmixed pycord's 48 kHz s16le stereo to mono; this subsystem treats chunks as opaque bytes
and never inspects them. `result_queue` carries `TranscriptionResult` (09) with a mutable `user_id`
field this subsystem stamps.

### On-disk

- `data/familiars/<id>/subscriptions.toml` — written through `SubscriptionRegistry` (02) by the
  subscribe/unsubscribe commands; DM rows never written (`persist=False`).
- Reads `data/familiars/<id>/{character.toml,activities.toml,character.md,lorebook.toml,seed_dream.md}`
  and `data/familiars/_default/{character.toml,activities.toml}` (parsing owned by 02/11).

## Config knobs

| Knob | Where read | Default | Effect here |
|---|---|---|---|
| `DISCORD_BOT` env | `run()` | required | bot token; missing → exit 1 |
| `FAMILIAR_ID` env / `--familiar` flag | `run()` | required (flag wins) | selects `data/familiars/<id>/` |
| `OPENROUTER_API_KEY` env | `run()` | required | LLM client factory; missing → exit 1 |
| `.env` autoload | `main()` | — | `load_dotenv()` before parsing |
| `-v/--verbose` (repeatable) | `cli` | 0 | WARNING/INFO/DEBUG |
| `[discord].dm_allowlist` | on_message | `[]` | int user ids admitted to DM; empty = no DMs |
| `[discord.text].respond_to_typing` | via TypingInterruptHandler (06) | `true` | constructed in `create_bot` |
| `[discord.text].typing_backoff_initial_s` / `_max_s` | 06 | 1.0 / 30.0 | idem |
| `[focus].unread_nudge_enabled` / `nudge_debounce_seconds` / `catch_up_limit` | FocusManager ctor | true / 30.0 / 20 | passed through in `_async_main` |
| `voice_window_size` / `text_window_size`, `budget_for("voice"/"text")` | `_default_assembler` | 02's defaults | window + per-section token caps |
| `text_silence_gap_fold_seconds`, `recent_history_coalesce_max_gap_seconds`, `display_tz`, `post_history_instructions`, `tools.loop_max_iterations` | responder/assembler wiring | 02's defaults | passthrough |
| `[llm.<slot>]` slots `fast`/`prose`/`background` (+`__image_description__`) | `_async_main` | required trio | slot→responder mapping; `prose.image_tools` gates `view_image` |
| `[providers.turn_detection].strategy` | `run()` | `"deepgram"` | `"ten+smart_turn"` → build local detector |
| `[providers.turn_detection.local].idle_fallback_s` | pump idle flush | detector default | replaces `DEFAULT_IDLE_FINALIZE_S` when detector active |
| `DEFAULT_IDLE_FINALIZE_S` (const, `stt.deepgram`) | pump idle flush | 0.5 s | idle gap before forced `finalize()` |
| transcriber `_IDLE_CLOSE_S` (from `[providers.stt.*].idle_close_s`) | idle watchdog | 30.0 (0 disables) | per-user stream close after silence |
| `memory_providers.projectors` | `_async_main` | 02's default list | projector task set |
| `FAMILIARS_ROOT` env | `default_familiars_root()` | platform data dir `…/familiar-connect/familiars` | per-user familiars root; env wins, else the home data dir so `git clean -fdx` can't wipe familiars (issue #201). Legacy `data/familiars/<id>` folders migrate in once on startup |
| `FAMILIAR_DEFAULTS_ROOT` env | `default_defaults_root()` | `data/familiars` | root of the tracked `_default` profile; resolved independently of user state so `_default` never migrates. Pure cores (`resolve_familiars_root` / `resolve_defaults_root`) keep both injectable for tests |
| Opus fallback path list | `load_opus` | hardcoded | libopus discovery |

## Dependency edges

Imports (this subsystem → others):

| Module imported | Subsystem |
|---|---|
| `bus.envelope`, `bus.topics`, `bus.protocols`, `diagnostics.collector/report/voice_budget`, `log_style` | 01 |
| `config` (`load_character_config`, `ConfigError`), `identity.Author`, `familiar.Familiar`, `subscriptions` | 02 |
| `history.store.HistoryStore` (sync facade), `history_store.staged_channels` | 03 |
| `embedding.create_embedder` | 04 |
| `context.*` layers + `Assembler`, `focus.FocusManager` / `PRIVATE_MESSAGE_GUILD_NAME`, `budget.TierBudget` | 05 |
| `processors.{TextResponder,VoiceResponder,DebugLoggerProcessor}`, `typing_interrupt.TypingInterruptHandler`, `processors.projectors` | 06 (projector factories bridge to 07) |
| `sleep.maintenance.SleepPromptText` | 04 |
| `llm.create_llm_clients`, `tools.{builtins,registry,scheduler,waker}` | 08 |
| `stt.{create_transcriber,Transcriber,TranscriptionResult,deepgram.DEFAULT_IDLE_FINALIZE_S}`, `tts.create_tts_client`, `tts_player.*`, `voice.{DaveVoiceClient,recording_sink.RecordingSink,turn_detection}` | 09 |
| `activities.{config,engine}`, `sources.twitch` (TwitchWatcher queue shape) | 11 |

Imported by: nothing else in the tree imports `bot.py`/`cli.py`/`commands/*` (this is the top of the
graph). `sources/*` are imported by `bot.py` and by 06's responders only through bus payloads (no
direct import). External: `discord` (py-cord), `dotenv`, `argparse`, libopus via ctypes.

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `test_bot_interactions.py` | defer/reply NotFound guards (B-SC1-2); `message_pings_bot` truth table (B-OM13); activity presence cb incl. not-ready no-op (B-PR26); on_ready resync ordering — away presence wins after focus sync (B-PR24); DM allowlist gate, guard ordering, ephemeral subscription never touching sidecar, focus seed/keep (B-OM8-10); disclaimer once-per-user, verbatim text + hint, pre-seeded ✅, checkmark-delete lifecycle, no history writes for disclaimer reactions (B-OM9, B-RX18-19) | needs-Rust-mock (SimpleNamespace/AsyncMock Discord objects → trait-object fakes) |
| `test_voice_intake.py` | no-transcriber → None; idempotence; sink attach + lazy clones; per-user dispatch/reuse; slow-user non-blocking (B-VI34); stop cancels tasks + stops every clone; `_on_recording_done` coroutine-ness (B-VI32); idle watchdog close/reopen + zero-disables (B-VI39); idle-finalize fallback incl. re-arm and no-flush-while-flowing (B-VI36); local endpointer fork, finalize-via-callback, vad_end buffering, `force_complete_if_pending` not direct finalize; member cache resolve order + one prefetch per user (B-VM29-30) | needs-Rust-mock; the concurrency assertions (drain-loop ticks, monkeypatched `DEFAULT_IDLE_FINALIZE_S` / `_prefetch_voice_member`) need tokio-time + injected-seam redesign |
| `test_image_collect.py` | attachment/embed/inline collection, content-type + extension filter, dedupe, sequential ids, marker format (B-DS50) | logic-portable |
| `test_message_embeds.py` | `compose_content_with_embeds` edge cases (B-DS49); `apply_message_edit` gates + store write-through (B-RX17) | logic-portable (needs 03 store) |
| `test_message_reactions.py` | `_emoji_repr` forms (B-RX21); dispatch gating + delta/clear plumbing (B-RX20); store-level halves belong to 03 | logic-portable (store halves covered by 03) |
| `test_discord_text_source.py` | topic/session/payload shape, monotonic seq, turn_id==event_id, optional-field defaults, pings_bot round-trip (B-DS43) | logic-portable (needs 01 bus) |
| `test_voice_source.py` | start/final/end ordering, one start per utterance, shared then fresh turn_ids, per-user concurrent turns, user_id in payload, stt_final phase stamp, vad_end buffering + wrong-user isolation (B-VS44-46) | logic-portable |
| `test_twitch_source.py` | queue→bus drain, envelope shape, clean cancellation (B-TS47) | logic-portable |
| `test_discord_embed_text.py` | full `format_embeds` rendering matrix incl. title==author dedupe, image-only fallback, duck-typing (B-DS48) | logic-portable |
| `test_cli.py` | parser shape, verbose counting, `--version`, bare-invocation help, no `sleep` subcommand | logic-portable (clap equivalents); subprocess test portable |
| `test_run_cmd.py` | failure-ladder exit codes; flag>env resolution; config→clients order; transcriber/turn-detector plumbing + degrade; opus discovery; teardown set + order on failure AND graceful signal path; `_wait_for_shutdown`; SIGINT handler set/remove; KeyboardInterrupt→0; assembler layer order; activity engine enable/disable + `voice_active_fn` | mostly Python-specific (deep `unittest.mock.patch` of module attrs); REDESIGN: pin the behaviors via injected-dependency integration tests, don't transliterate the patching |
| `test_diagnose_cmd.py` | span regex incl. ANSI tolerance; multi-file; empty placeholder (B-CLI61) | logic-portable |
| `test_logging.py` | `setup_logging` level ladder, explicit override, invalid level error, pkg-logger INFO floor (B-CLI52); formatter halves belong to 01 | partially portable; formatter tests are 01's |
| `test_version.py` | version output | logic-portable |

## Rust port notes

- **Crate choices** (see `00-rust-ecosystem.md`): serenity for gateway/REST + songbird ≥0.6 for
  voice/DAVE (replaces `DaveVoiceClient` + pycord recording entirely). Slash commands via poise or
  raw serenity interactions — either way preserve the defer-first + ephemeral-followup discipline
  and the NotFound-is-benign contract (B-SC1-2). CLI: `clap` (derive) + `dotenvy`; percentile/regex
  for diagnose: `regex` crate (port `_SPAN_RE` and the interpolated percentile exactly).
- **Recording path changes shape**: pycord's `start_recording(sink, callback)` and the
  coroutine-callback quirk (B-VI32) disappear. Songbird delivers per-SSRC RTP via `VoiceEvent`
  handlers; keep the `(user_id, mono_pcm)` mpsc queue as the seam between 09 and this subsystem so
  the router/pump/fan-in topology (B-VI34-41) ports unchanged. Caution from the ecosystem spec: do
  NOT key per-user state solely off songbird speaking-state events (inconsistently fired) — the
  first-audio-chunk lazy creation + idle-finalize fallback here is the resilient design; keep it.
- **Duck-typed seams → traits**: `getattr(template, "_IDLE_CLOSE_S", 0.0)` and the conditional
  `endpointing_ms = 10` setattr become explicit `Transcriber` trait surface
  (`fn idle_close(&self) -> Option<Duration>`, `fn set_endpointing_hint(&mut self, Duration)` with a
  default no-op impl). `format_embeds`/`collect_images` duck-typing becomes small input structs
  (`EmbedView`, `AttachmentView`) constructed from serenity types — tests then build them literally.
- **BotHandle** is shared mutable state touched from many tasks (`voice_runtime`, `voice_members`,
  late-set `resolve_member`/`activity_engine`). In Rust: `Arc<BotHandle>` with `DashMap`/
  `RwLock<HashMap>` fields; the closures (`send_text`, `trigger_typing`) become `Arc<dyn …>` trait
  objects so 06 keeps its injection seams. `resolve_member` must stay sync/non-blocking (B-VM29).
- **Task model**: Python `Task.cancel()` + `await` with suppress maps to either
  `JoinHandle::abort()` + awaiting the `JoinError`, or (better for the pumps, which hold WS clones)
  a `CancellationToken` + `select!` so teardown can still run the clone `stop()` cleanly. The
  parallel `gather(*clone.stop())` at unsubscribe (B-VI41) is `join_all`; preserve the ordering:
  producers cancelled/awaited before transcriber stops. Name tasks (tracing spans) matching the
  Python task names — logs are part of the operator contract.
- **TaskGroup + except\* → JoinSet/select + typed error**: model `_GracefulShutdown` as a normal
  variant of the supervisor's return, not a panic; first-completion of {signal, any-task-error}
  triggers cancellation of the rest, then the teardown sequence (B-CLI57) runs unconditionally
  (Rust: an explicit shutdown function, not Drop — several steps are async). Signal handling via
  `tokio::signal::unix` (SIGINT+SIGTERM streams); implement the two-stage force-kill semantics
  (second signal resets to default disposition — `libc::signal(SIG_DFL)` or process::exit).
- **Monkeypatch seams to make injectable**: `DEFAULT_IDLE_FINALIZE_S`, the prefetch function, and
  `_DEFAULT_FAMILIARS_ROOT` are monkeypatched in tests — take them as constructor/env parameters.
  Time-based tests (idle watchdog 0.05 s windows) should use `tokio::time::pause`.
- **Do not port**: `commands/example.py` (template), the `getattr` reflection style, py-cord's
  channel-cache `get_channel`/`fetch_channel` split maps to serenity cache + http fallback with the
  same never-raise contract. `on_message_edit`'s use of the *sync* store facade is a Python
  threading artifact — in Rust use the async store, but keep the no-await-in-gateway-handler spirit
  by spawning if the write can block.
- **Verify against real Discord early**: (a) serenity puts reply-ping targets in
  `message.mentions` like py-cord does — `message_pings_bot` depends on it; (b) DAVE receive
  multi-speaker per-SSRC (ecosystem spec flags it least-hardened); (c) `AllowedMentions` restriction
  parity (never ping via content markers).
