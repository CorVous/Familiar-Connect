# Architecture overview

A Discord bot shell with two-way plumbing for text and voice, plus a
Twitch EventSub client. Incoming events flow through an in-process
**event bus** to subscribed **processors**.

Phase 2 adds the voice reply loop with sub-200 ms barge-in: a new
utterance cancels the previous reply's `TurnScope`, stopping the
LLM stream and flushing the TTS buffer.

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
- **Discord voice** — `subscribe-voice` / `unsubscribe-voice` slash commands join a voice channel with `DaveVoiceClient` (DAVE E2E encryption). On subscribe the bot attaches a `RecordingSink`, starts the Deepgram transcriber against a fresh result queue, and runs a `VoiceSource` task draining transcripts onto the bus. On unsubscribe the per-channel pump + source tasks are cancelled, recording is stopped, and the transcriber is closed.
- **Transcription** — Deepgram streaming client. Instantiated on startup; wired to the bus via `VoiceSource` once a voice channel is active. v1 supports one voice channel at a time (single transcriber per familiar).
- **TTS synthesis** — Azure / Cartesia / Gemini clients behind a uniform `TTSResult` shape. `DiscordVoicePlayer` calls `synthesize(text)` and pushes the mono PCM (after stereo conversion) through pycord's voice client. When no TTS client is configured `LoggingTTSPlayer` is used.
- **OpenRouter LLM client** — one `LLMClient` per call-site slot.
- **SQLite history store** — `data/familiars/<id>/history.db`. Raw `turns` table is the source of truth; `summaries` and `cross_context_summaries` are watermarked side-indices.
- **Subscription registry** — `data/familiars/<id>/subscriptions.toml`, written by the subscribe/unsubscribe slash commands.
- **Twitch EventSub** — client code present; its queue is drained by `TwitchSource` onto the bus.

## Topics

Topic strings live in `familiar_connect.bus.topics`:

| Topic | Payload | Backpressure default |
|---|---|---|
| `discord.text` | channel, guild, `Author`, content | unbounded |
| `discord.voice.state` | member, channel | unbounded |
| `voice.audio.raw` | PCM chunk + speaker | drop-oldest |
| `voice.transcript.partial` | text + turn_id | block |
| `voice.transcript.final` | text + turn_id + speaker | block |
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

Barge-in latency budget: 200 ms from a new `voice.activity.start` to TTS
playback halted. Verified by
`tests/test_voice_responder.py::TestBargeIn::test_barge_in_during_speech_cuts_playback_fast`.

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
