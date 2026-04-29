# Architecture overview

A Discord bot shell with two-way plumbing for text and voice, plus a
Twitch EventSub client. Incoming events flow through an in-process
**event bus** to subscribed **processors**.

Phase 2 adds the voice reply loop with sub-200 ms barge-in: a new
utterance cancels the previous reply's `TurnScope`, stopping the
LLM stream and flushing the TTS buffer.

This page covers the *what's wired today* picture. For deeper dives:

- [Memory strategies](memory-strategies.md) — the four families,
  what's implemented, where alternative strategies plug in.
- [Voice pipeline](voice-pipeline.md) — cascaded vs full-duplex,
  two-stage turn detection, sentence-level streaming, swap points.
- [Tuning](tuning.md) — single-point reference for every operator
  knob.
- [Roadmap](roadmap.md) — research-driven priorities for the
  next iteration.

```mermaid
flowchart LR
    dt([Discord text])     --> dts[DiscordTextSource]
    dv([Discord voice])    --> deepg[Deepgram]
    tw([Twitch EventSub])  --> tws[TwitchSource]

    deepg --> vs[VoiceSource]

    dts --> bus{{Event bus}}
    tws --> bus
    vs --> bus

    bus --> dbg[DebugLoggerProcessor]
    bus --> vr[VoiceResponder]
    bus --> tr[TextResponder]

    vr --> asm[Assembler]
    vr --> llm[[LLMClient.chat_stream]]
    vr --> tts[[TTSPlayer]]
    vr -.cancel scope.-> rr[TurnRouter]

    tr --> asm
    tr --> llm
    tr --> send[[BotHandle.send_text]]
    tr -.cancel scope.-> rr
```

## Components

- **CLI** — `familiar-connect run --familiar <id>` (argparse, subcommand dispatch).
- **Configuration** — TOML with deep-merge over `data/familiars/_default/character.toml`. Per-channel overrides live under `[channels.<id>]`. See [Configuration model](configuration-model.md).
- **Event bus** — in-process, topic-keyed fan-out. `InProcessEventBus` implements the `EventBus` Protocol. Per-topic `BackpressurePolicy` (`BLOCK`, `DROP_OLDEST`, `DROP_NEWEST`, `UNBOUNDED`). Lifecycle: `starting → running → draining → stopped`.
- **Turn router** — `TurnRouter.begin_turn(session_id, turn_id)` cancels any prior `TurnScope` in the same session before registering the new one; different sessions are independent.
- **Stream sources** — publish onto the bus.
  - `DiscordTextSource` — called from `on_message`; publishes `discord.text`.
  - `TwitchSource` — drains the `TwitchWatcher` queue; publishes `twitch.event`.
  - `VoiceSource` — drains the Deepgram transcription queue; publishes `voice.activity.start`, `voice.transcript.partial`, `voice.transcript.final`, `voice.activity.end`. All events in one utterance share `turn_id`.
- **Context assembly** — `Assembler` composes a layered system prompt. Phase-2 layers: `CoreInstructionsLayer` (file: `data/familiars/_default/core_instructions.md`), `CharacterCardLayer` (file: `data/familiars/<id>/character.md`), `OperatingModeLayer` (voice-terse vs text-verbose), `RecentHistoryLayer` (reads from `HistoryStore.recent`). Phase-3 adds rolling summary, cross-channel summary, and FTS-backed RAG.
- **LLM** — `LLMClient` exposes `chat()` (blocking) and `chat_stream()` (async-iterator of content deltas). The streaming variant releases the process-wide rate-limit semaphore as soon as the request is accepted so barge-in cancellation isn't starved.
- **TTSPlayer** — `speak(text, scope=...)` returns when playback finishes or the turn scope is cancelled. Production default is `DiscordVoicePlayer`, which synthesizes via the configured TTS client and pushes the resulting PCM through `voice_client.play(...)`. When no TTS client is configured the loop falls back to `LoggingTTSPlayer`, which only logs intended speech. `MockTTSPlayer` is used in tests.
- **BotHandle** — adapter exposed to the lifecycle wiring so bus-only processors can post back to Discord without taking a direct `discord.Bot` reference. Carries `send_text(channel_id, content)` and a `voice_runtime: dict[int, VoiceRuntime]` map populated by `/subscribe-voice`.
- **Processors** — subscribe to topics.
  - `DebugLoggerProcessor` — one log line per event on every subscribed topic.
  - `TextResponder` — consumes `discord.text` (appends the user turn directly, seeds the RAG cue, assembles prompt with `viewer_mode="text"`, streams LLM, posts via `BotHandle.send_text`, appends the assistant turn). Owning the user-turn write keeps read-after-write consistency for `RecentHistoryLayer` in the same task. A `SilentDetector` watches stream deltas; on a `<silent>` sentinel reply the post and assistant-turn append are skipped (the user turn is still recorded). See [Multi-party addressivity](context-pipeline.md#multi-party-addressivity).
  - `VoiceResponder` — consumes `voice.activity.start` (cancels prior scope via the router; fires `TTSPlayer.stop`) and `voice.transcript.final` (appends user turn, assembles prompt, streams LLM, speaks). Stale finals (mismatched `turn_id`) are dropped. Silent-sentinel handling mirrors `TextResponder`: on `<silent>`, TTS is not invoked.
- **Diagnostics** — `@span(name)` decorator in `familiar_connect.diagnostics.spans` emits timing logs (`span=<name> ms=<n> status=<ok|error>`). Logs-first aggregation; a metrics collector + `/diagnostics` slash command come in Phase 5.
- **Discord text** — `on_message` event handler + `subscribe-text` / `unsubscribe-text` slash commands. Built on py-cord.
- **Discord voice** — `subscribe-voice` / `unsubscribe-voice` slash commands join a voice channel with `DaveVoiceClient` (DAVE E2E encryption). On subscribe the bot attaches a `RecordingSink` and runs a `VoiceSource` task draining transcripts onto the bus. The audio pump dispatches per Discord user_id: the first audio chunk from a new SSRC lazily clones the configured Deepgram transcriber and opens a fresh WebSocket for that speaker, so two people talking concurrently get independent endpointing and don't slice each other's sentences. A per-user fan-in tags every result with the originating user_id before forwarding to the shared result queue. On unsubscribe the pump, source, and every per-user fan-in are cancelled, recording is stopped, and every per-user transcriber is closed.
- **Transcription** — Deepgram streaming client. The instance loaded at startup acts as a *template*: `clone()` is called once per Discord user that speaks. Diarization stays off — Discord delivers per-SSRC audio, so attribution is exact and not AI-inferred. Defaults bias toward fewer mid-sentence cuts: `endpointing_ms=500`, `utterance_end_ms=1500`, `smart_format=true`, `punctuate=true`. Optional `DEEPGRAM_KEYTERMS` biases nova-3 toward project jargon and member display names.
- **TTS synthesis** — Azure / Cartesia / Gemini clients behind a uniform `TTSResult` shape. `DiscordVoicePlayer` calls `synthesize(text)` and pushes the mono PCM (after stereo conversion) through pycord's voice client. When no TTS client is configured `LoggingTTSPlayer` is used.
- **OpenRouter LLM client** — one `LLMClient` per call-site slot.
- **SQLite history store** — `data/familiars/<id>/history.db`. Raw `turns` table is the source of truth; `summaries`, `cross_context_summaries`, and `people_dossiers` are watermarked side-indices.
- **Subscription registry** — `data/familiars/<id>/subscriptions.toml`, written by the subscribe/unsubscribe slash commands.
- **Twitch EventSub** — client code present; its queue is drained by `TwitchSource` onto the bus.

## Topics

Topic strings live in `familiar_connect.bus.topics`:

| Topic | Payload | Backpressure default |
|---|---|---|
| `discord.text` | channel, guild, `Author`, content | unbounded |
| `discord.voice.state` | member, channel | unbounded |
| `voice.audio.raw` | PCM chunk + speaker | drop-oldest |
| `voice.transcript.partial` | text + turn_id + user_id | block |
| `voice.transcript.final` | text + turn_id + user_id + speaker | block |
| `voice.activity.start` / `.end` | turn_id | block |
| `twitch.event` | `TwitchEvent` | unbounded |
| `llm.response.chunk` / `.final` | text delta / message | block |
| `tts.audio.chunk` / `.final` | audio bytes + word timestamps | block |

## Voice reply loop

```
voice.activity.start  → TurnRouter.begin_turn(session, turn_id)
                         → prior scope.cancel()
                         → TTSPlayer.stop()  (flush in-flight audio)

voice.transcript.final → if scope.turn_id == event.turn_id:
                           history.append(user turn)
                           Assembler.assemble(ctx)
                           LLMClient.chat_stream(messages)
                             (bail if scope.is_cancelled())
                           TTSPlayer.speak(reply, scope=scope)
                           history.append(assistant turn)
                           router.end_turn(scope)
```

`voice.transcript.final` is spawned as a per-(session, user) `asyncio.Task`, so the
bus dispatcher returns to the subscription loop immediately. A subsequent
`voice.activity.start` runs `prior.cancel()` while the prior turn is still
parked at an LLM or TTS await point — without the spawn, the dispatcher
would sit inside the prior `handle()` and the cancel signal would arrive
only after the old reply had played in full.

Scope keys are per `(channel_id, user_id)`. Discord delivers per-SSRC audio
so every speaker fires their own `activity.start`; channel-level scoping
would let any speaker barge any other speaker's in-flight reply, which is
not desired. Same-speaker self-barge still works as expected — the player's
poll loop catches `scope.is_cancelled()` and stops `vc.play()` within one
poll tick. A global `TTSPlayer.stop()` from `_on_activity_start` would also
cut a *different* user's in-flight reply (Discord exposes one shared voice
client per channel), so cancellation only flows through the scope.

Voice user turns are appended to history with the speaker's `Author`
resolved through `BotHandle.resolve_member(channel_id, user_id)`. The
resolver consults a voice-member side cache populated by two sources:
`on_voice_state_update` events for state changes (join/mute/move) and
a background `guild.fetch_member()` triggered when the audio pump sees
a new user_id for the first time. The side cache works around the
absence of the privileged `members` intent — without it
`guild.get_member()` only knows users seen through other events
(messages, voice state changes) and silently returns `None` for
voice-only joiners. A cache miss records the turn anonymously rather
than blocking the audio path on a Discord fetch.

Barge-in latency budget: 200 ms from a new `voice.activity.start` to TTS
playback halted. Verified end-to-end (bus subscribe pattern) by
`tests/test_voice_responder.py::TestDispatchLoop` and
`::TestBargeIn::test_barge_in_during_speech_cuts_playback_fast`.

## Per-channel latency knobs

`[channels.<id>]` in `character.toml` overrides three knobs:

- `history_window_size` — how many recent turns the `RecentHistoryLayer`
  pulls for this channel (default: `[providers.history].window_size`).
- `prompt_layers` — ordered list of layer names (currently parsed but
  applied only via the default ordering; per-channel reordering lands
  with Phase 3's richer layer stack).
- `message_rendering` — `"prefixed"` (always include
  `[HH:MM display_name]` content prefix; UTC) or `"name_only"`
  (rely on the OpenAI `name` field alone — save tokens in DMs).
