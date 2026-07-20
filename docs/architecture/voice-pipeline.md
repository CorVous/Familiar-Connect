# Voice pipeline

How a spoken utterance becomes audible bot speech, and where to swap
stages. [Overview](overview.md) covers Discord plumbing;
[Streaming bus](streaming-bus.md) the in-process event bus.

## Cascaded vs full-duplex

- **Cascaded (STT → LLM → TTS).** Three swappable stages. Persona
  via prompt, brain via OpenRouter, tools available. Production stacks
  (Pipecat, LiveKit, RealtimeVoiceChat) target 700–900 ms voice-to-voice.
- **Full-duplex / S2S (Moshi, Sesame CSM).** One model, audio in and
  out, ~200 ms theoretical. LLM brain bundled — no OpenRouter, no
  tool-calling, degraded prompt knobs.

**Familiar-Connect is cascaded by design.** Persona + OpenRouter is
the central knob. Two-stage turn detection and sentence-streaming TTS
close most of the latency gap without forking the architecture. See
[Decisions — full-duplex S2S](decisions.md#full-duplex-speech-to-speech-pipelines-moshi-sesame-csm).
S2S is a research branch on the
[roadmap](roadmap.md#v5-full-duplex-s2s-as-a-research-branch).

## Stages

```
Discord Opus  →  RecordingSink  →  per-user PCM
                                        │
                                        ▼
                                 [VAD / Turn detection]
                                        │
                                        ▼
                                  [STT transcriber]
                                        │
                                        ▼
                                  voice.transcript.final
                                        │
                                        ▼
                            VoiceResponder  →  Assembler
                                        │            │
                                        ▼            ▼
                          LLMClient.chat_stream   prompt
                                        │
                                        ▼
                                [SentenceStreamer]
                                        │
                                        ▼
                                  [TTS client]
                                        │
                                        ▼
                                DiscordVoicePlayer
                                        │
                                        ▼
                                  Discord Opus out
```

Bracketed stages are pluggable. Unbracketed (recording sink, bus,
responder, assembler) are project glue.

## Turn detection

**Today:** Deepgram's hosted endpointer. One WebSocket per speaker,
biased by `endpointing_ms` and `utterance_end_ms`. See
[Tuning — STT](tuning.md#stt-deepgram).

**Field consensus:** two-stage detection is default everywhere
(Pipecat, LiveKit, TEN, Agora). Pure-VAD endpointing is an anti-pattern.

- *Stage 1* — local VAD (TEN-VAD, native lib + bundled ONNX,
  Apache 2.0). Fast, in-process. Beats remote by 150–200 ms.
- *Stage 2* — semantic turn classifier over buffered audio. Pipecat's
  Smart Turn v3 (BSD-2, ~12 ms, 360 MB) is the leanest open option;
  trained on filler words STT drops, so it beats transcription-based
  endpointing.

**Status:** V1 phase 2 — local endpointer behind a feature flag.
Three types plus a factory under
`familiar_connect::voice::turn_detection`:

- `TenVAD(sample_rate=16000, hop_size=256)` — Agora's TEN-VAD via the
  `ten_vad` package. Stateful native handle: feed 16 ms (256-sample)
  or 10 ms (160-sample) chunks of 16 kHz mono int16 PCM, get back a
  probability + flag pair; `reset()` between utterances rebuilds the
  C handle. Returns `is_speech(chunk)` for threshold use.
- `SmartTurnDetector(model_path, max_duration_s=16.0)` — Pipecat's
  Smart Turn v3. Stateless: feed the buffered utterance after VAD
  silence. Handles both 2-class softmax and single sigmoid output
  shapes (Pipecat's exports vary). Returns `is_complete(audio)`.
- `UtteranceEndpointer(vad, smart_turn, on_turn_complete, …)` —
  per-user state machine driving both above over a 48 kHz mono PCM
  stream. Feeds 16 ms VAD windows after 3:1 boxcar-decimation
  resample, tracks `IDLE → SPEAKING → silence-after-speech → classify`,
  and awaits `on_turn_complete(audio)` on a `complete` verdict. An
  `incomplete` verdict holds the callback until a fresh speech burst
  followed by a fresh silence streak.
- `LocalTurnDetector` (factory) + `create_local_turn_detector_from_env()`
  — bundles model paths and thresholds. Builds a fresh
  `UtteranceEndpointer` per Discord user (TenVAD's native handle is
  stateful; Smart Turn is shared).

Both runtimes are behind the `local-turn` feature; build it in:

```bash
cargo build --features local-turn
```

TEN-VAD ships its model + native shared library, bound through a
`ten-vad-sys` crate. Smart Turn's ONNX weights are pulled from
[`pipecat-ai/smart-turn-v3`](https://huggingface.co/pipecat-ai/smart-turn-v3)
on first use via the `hf-hub` crate — the Hub cache
(`~/.cache/huggingface`) covers offline reruns. Default filename is
the CPU export (`smart-turn-v3.2-cpu.onnx`); override via
`[providers.turn_detection.local].smart_turn_filename` when the ONNX
runtime (`ort`) is built with a GPU execution provider.

### How the audio path forks

When `[providers.turn_detection].strategy = "ten+smart_turn"` and
model files exist, `start_voice_intake` builds a per-user
endpointer alongside the per-user Deepgram clone. The shared sink-side
pump demuxes audio onto a per-user queue; one drain task per user_id
feeds every PCM chunk into both clone and endpointer. Per-user drain
tasks isolate slow speakers (network blip, slow VAD, GC pause) so one
stalled `send_audio`/`feed_audio` can't head-of-line-block the call:

```
Discord Opus → RecordingSink → per-user PCM
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
                   Deepgram clone       UtteranceEndpointer
                   (endpointing_ms=10,    (TEN-VAD + Smart Turn)
                    Finalize-driven)              │
                                                  │ on_turn_complete
                                                  ▼
                                        clone.finalize() ──► Deepgram flush
```

`clone.endpointing_ms` drops to `10` when local detection is active
so Deepgram won't endpoint on its own — it relies on `Finalize`
messages from the local chain. Strategy + tuning live in
[`[providers.turn_detection]`](tuning.md#local-turn-detection-v1).

Default is **off** (`strategy = "deepgram"`): the bot uses Deepgram's
hosted endpointer.

### Idle-finalize fallback

Both endpointing strategies share a failure mode: Discord's client VAD
halts RTP during silence, so neither Deepgram's hosted endpointer nor
the local TEN-VAD chain ever *sees* the trailing silence that ends a
turn. Without a backstop the buffered final sits until the speaker's
**next** utterance — the "transcript doesn't come through until the next
sound" symptom.

The per-user audio pump (`user_pump`) is that backstop. After each
chunk it arms an idle timer; when no audio arrives for the flush window
it forces the turn to end:

- **Plain Deepgram** (`detector is None`) — sends `Finalize`
  (`DeepgramTranscriber.finalize`) after `DEFAULT_IDLE_FINALIZE_S`
  (0.5 s), flushing whatever Deepgram has buffered.
- **Local turn detection** — calls
  `UtteranceEndpointer.force_complete_if_pending()` after
  `[providers.turn_detection.local].idle_fallback_s` (1.5 s default,
  longer so a natural pause doesn't defeat Smart Turn's
  hold-through-pause). It drains a turn stranded in
  `SPEAKING` (burst stopped before the silence streak classified) or
  `POST_INCOMPLETE` (a Smart Turn `incomplete` misfire), firing
  `on_turn_complete` on state rather than buffered bytes.

The timer only arms while the buffer is dirty, so a long silence blocks
on the queue rather than re-finalizing every window.

See [Roadmap V1](roadmap.md#v1-local-vad-semantic-turn-detection).

#### Test coverage

Two layers pin the state machine:

- `familiar-connect/tests/voice_endpointer.rs` — unit tests with canned
  VAD/SmartTurn return values. Drives every state-machine edge.
- `familiar-connect/tests/voice_endpointer_fixtures.rs` — audio-fixture integration
  tests. Synthesises 48 kHz mono int16 PCM (silence + 220 Hz sine
  bursts), feeds it through the real resampler + framer, and validates
  the three patterns field consensus calls out: **complete-sentence**
  (one classify, one callback), **mid-thought** (in-utterance pause
  below `silence_ms` must not trip classification), and **filler**
  (incomplete verdict holds the callback; resumed speech with a
  complete verdict fires it). VAD is energy-thresholded over actual
  frame bytes so the fixture drives transitions; SmartTurn is a
  verdict stub (no ONNX dependency in CI).

## STT (transcription)

**Today:** `DeepgramTranscriber` in `familiar_connect::stt::deepgram`.
Per-speaker clone-from-template; one stream per Discord user,
lazy-opened, closed after `idle_close_s`.

**Pluggability:** V3 phase 1 lifted the clone-template shape into a
`Transcriber` trait (`familiar_connect::stt::protocol`). The voice
pipeline (`bot.rs`, `sources/voice.rs`, `familiar.rs`) types against
the trait; backend selection lives in `stt::factory`, dispatched on
`[providers.stt].backend`.

V3 phase 2 added `ParakeetTranscriber` (NeMo Parakeet-TDT 0.6B v3,
local, no API key); phase 3 added `FasterWhisperTranscriber`
(`faster-whisper` over CTranslate2). Both use buffer-and-finalize:
48 kHz Discord PCM is resampled to 16 kHz mono and accumulated;
`finalize()` runs the model and emits one `is_final=True` result.
Neither has an internal endpointer, so both must pair with
`[providers.turn_detection].strategy = "ten+smart_turn"` — the local
endpointer drives `finalize()` on turn-complete.

Build with `cargo build --features local-turn,local-stt`. Parakeet
pulls torch + ~600 MB of weights; FasterWhisper is lighter (~150 MB
for `small`, no torch).

**Partial vs final transcripts.** Modal's benchmark: partials are a
UX feature, not a latency feature. The LLM can't start until the
final, so final-time gates everything. Local-VAD + final-only Parakeet
can beat streaming-Whisper end-to-end. Measure before optimising for
partials.

## LLM

`LLMClient.chat_stream` over OpenRouter. Already streaming, cancellable
via `TurnScope`. Stays. Lesson: don't waste streaming — feed the next
stage incrementally.

## Sentence streaming

`VoiceResponder` feeds each LLM delta through a `SentenceStreamer`
(`familiar_connect::sentence_streamer`) and calls `TTSPlayer.speak`
once per completed sentence. Time-to-first-audio drops from "after
the LLM finishes" to "after the first sentence" — the same 1–3 s
perceived-latency win Pipecat's `SentenceAggregator` ships.

Splitter is abbreviation-aware: `Mr.` / `Dr.` / `etc.` /
single-letter initials (`J. K. Rowling`) don't trip a boundary. A
trailing partial without a terminator (model omits the final period)
is drained on stream end via `flush()` and spoken last.

**Silent sentinel + leak guard.** `StreamGate` (Rust `silence.rs`) runs
ahead of the splitter on every delta. Sentences finalised before the
gate decides are buffered; on `silent` they're dropped and TTS is never
invoked; on `speak` they flush and the streamer feeds TTS as new
sentences arrive. Beyond `<silent>`, the gate also recognises a
tool-call block the model occasionally leaks as plain text (`<invoke …`,
`silent(…)`, `read_channel(…)`, `<tool_call …`), staying pending while
the token is still split across delta boundaries and latching `suppress`
(or `silent`, for a leaked `silent` call) so the raw XML never reaches
TTS or the persisted turn (issue #109). The confirmed-leak
classification is shared with the agentic loop's return-time strip guard
(`classify_leading_leak`), the single source of truth. The text path,
which streams no content mid-turn, keeps the simpler `SilentDetector`.

**Cancellation.** Each `TTSPlayer::speak(sentence, scope)` call is
awaited serially. Barge-in cancels the current `TurnScope`;
`DiscordVoicePlayer`'s poll loop cuts the in-flight sentence within
~20 ms and the responder bails before queueing the next. The
assistant turn records only if the full reply played uncancelled.

## TTS

Three clients behind `synthesize(text) → TTSResult`: `AzureTTSClient`,
`CartesiaTTSClient`, `GeminiTTSClient`. `DiscordVoicePlayer`
synthesises, mono→stereo, pushes through songbird. Without a configured
client, `LoggingTTSPlayer` logs the intended speech.

Already a trait seam. Adding a backend is one new type.

### Byte-level streaming (Cartesia)

`CartesiaTTSClient` exposes a second method,
`synthesize_stream(text) → AsyncIterator[bytes]`, yielding raw mono
`pcm_s16le` chunks as the WebSocket delivers them. When the configured
TTS client implements this, `DiscordVoicePlayer` takes the streaming
path:

1. Open Cartesia stream (~140 ms TTFB).
2. Pre-buffer the first chunk into a `StreamingPCMSource` (a
   thread-safe songbird audio source with `feed` / `close_input`).
3. `vc.play(source)` — songbird's audio thread drains 20 ms frames.
4. A producer task feeds the rest into the source as chunks arrive.
   `close_input()` on stream end lets the reader return end-of-stream and
   songbird stop the player cleanly.

That cuts `voice.tts_to_playback` from full-sentence synthesis time
(1.5–3 s for a long sentence on `cartesia-sonic-3` at ~270 ms/word)
down to ~one TTFB. Cancellation: `scope.is_cancelled()` flips
`vc.stop()` within a poll tick; the producer drops out of its loop on
the next `feed` and `close_input` releases any blocked reader.

Azure and Gemini stay on the buffered `synthesize` path (their SDKs
return one big result), so `DiscordVoicePlayer.speak` falls through
to the prior synthesize-then-play behaviour.

**Mimi-codec lineage.** Mimi (Kyutai, 12.5 Hz frames) is becoming the
open audio-token standard — Sesame CSM, Hibiki, Moshi all use it.
Sesame CSM-1B accepts conversational context for prosody continuity;
voice stability needs fine-tuning. V4 tracks adding a Sesame or Piper
backend once upstream stabilises.

## Latency budget

Cascaded with cloud STT/TTS, April 2026:

| Stage | Range |
|---|---|
| VAD detects end-of-speech | 50–150 ms |
| Semantic turn confirmation | 30–100 ms |
| STT final transcript | 200–400 ms cloud / 300–500 ms Faster-Whisper |
| LLM time-to-first-token | 200–500 ms |
| LLM first-sentence completion | +100–400 ms |
| TTS time-to-first-audio | 100–300 ms |
| Discord / Opus encoding + jitter | 60–120 ms |
| **Floor** | **~700 ms** |
| **Comfortable** | **1.0–1.2 s** |
| **Feels broken above** | **2 s** |

Biggest remaining wins: local VAD (150–200 ms) and semantic turn
detection (skip the silence timeout). Sentence-level TTS streaming and
byte-level Cartesia streaming both shipped — see
[Sentence streaming](#sentence-streaming) and
[Byte-level streaming](#byte-level-streaming-cartesia).

## Per-turn budget telemetry

`familiar_connect::diagnostics::voice_budget::VoiceBudgetRecorder` (a
process singleton like `SpanCollector`) stamps four phase markers
keyed by `turn_id` and emits one span per adjacent gap into the shared
collector, so `/diagnostics` shows the breakdown in its summary table.

| Phase | Stamp site |
|---|---|
| `vad_end` | the turn-complete callback calls `VoiceSource::record_vad_end`; `VoiceSource::handle` drains on the next transcript event for the same `user_id` |
| `stt_final` | `VoiceSource::handle` (just before publishing `voice.transcript.final`) |
| `llm_first_token` | `VoiceResponder::stream_and_speak` on first delta |
| `tts_first_audio` | `VoiceResponder::speak` (deduped — first sentence wins) |
| `playback_start` | `DiscordVoicePlayer.speak` after `vc.play(source)` |

| Span | Gap |
|---|---|
| `voice.vad_to_stt` | `vad_end` → `stt_final` (Deepgram finalize round-trip after local turn complete) |
| `voice.stt_to_ttft` | `stt_final` → `llm_first_token` (LLM TTFT, includes assembler) |
| `voice.ttft_to_tts` | `llm_first_token` → `tts_first_audio` (first-sentence completion) |
| `voice.tts_to_playback` | `tts_first_audio` → `playback_start` (TTS synthesis + voice-client lock) |
| `voice.total` | `stt_final` → `playback_start` (user-perceived latency) |

`vad_end` only stamps when local turn detection (TEN-VAD + Smart Turn)
is wired in. With Deepgram-only endpointing, VAD-end and final fuse
into one `is_final` result and the funnel starts at `stt_final`.
`voice.total` keeps its `stt_final` start so historical numbers stay
comparable.

Recorder is best-effort: the voice path never blocks on it, and
exceptions inside `record(...)` are swallowed so instrumentation can't
take the bot down.

### Prompt cache friendliness

OpenAI's prompt caching matches the longest stable prefix (1024-token
minimum, 128-token granularity). A change to any mid-prompt layer
invalidates everything after it, so `default_assembler` builds layers
in **stability descending** order:

| Position | Layer | Refresh trigger |
|---|---|---|
| 1 | `CharacterCardLayer` | file content change |
| 2 | `OperatingModeLayer` | `viewer_mode` flip (constant per mode) |
| 3 | `ConversationSummaryLayer` | `SummaryWorker` writes (every N turns) |
| 4 | `PeopleDossierLayer` | `PeopleDossierWorker` watermark advances |
| 5 | `RagContextLayer` | per-turn cue (always changes) |
| — | `RecentHistoryLayer` | per-turn (contributes user/assistant messages, not system text) |

`RagContextLayer` therefore sits at the tail of the system message,
so its inevitable per-turn churn invalidates *only* itself — the
prefix from `CharacterCardLayer` through `PeopleDossierLayer` stays
cached when its constituent layers haven't moved.

The `default_assembler` layer-order test in
`familiar-connect/src/commands/run.rs` pins this ordering so a refactor
doesn't silently drop into "everything goes cold" mode. Prompt-cache hit count surfaces as `cached=N` on the
`[LLM call]` log line below — if it drops to 0, suspect a mid-prompt
layer that just started churning between turns.

### LLM call signals

Every `LLMClient.chat_stream` call adds three spans + one structured
`[LLM call]` log line. The breakdown separates prompt-bloat from
OpenRouter routing-tax at a glance.

| Span | Phase |
|---|---|
| `llm.ttfb.<slot>` | request initiation → first response byte |
| `llm.ttft.<slot>` | request initiation → first content delta |
| `llm.total.<slot>` | request initiation → stream end |

The log line carries `slot`, `model`, `chars` (input payload size),
`ttfb_ms` / `ttft_ms` / `total_ms`, and — when upstream returns them
via OpenRouter's `usage: { include: true }` flag — `provider`,
`in_tokens`, `out_tokens`, and `cached` (prompt-cache hit count,
surfaced when the underlying provider supports it). `voice.stt_to_ttft`
covers the full STT-to-LLM-first-token gap; `llm.ttft.<slot>` is the
LLM-only slice plus headers. Comparing the two isolates assembler /
network from raw model latency.

## Barge-in

Already implemented. New `voice.activity.start` cancels prior
`TurnScope`:

1. Cancels in-flight LLM stream (semaphore released on accept so
   cancel isn't starved).
2. Calls `TTSPlayer.stop()` to flush in-flight audio.

Verified sub-200 ms by the barge-in tests in
`familiar-connect/tests/responders_voice.rs`.
See [Voice reply loop](overview.md#voice-reply-loop).

Every voice turn emits exactly one decision line for observability:

- `[💤 Voice] decision=silent` — `<silent>` sentinel latched.
- `[Voice] decision=respond` — gate opened on real content.
- `[Voice] decision=preempted` — barge-in cancelled the turn before
  the gate latched. Without this line a continuously-speaking user
  produced a chain of `[LLM call] status=cancelled` with no way to
  tell which transcript was dropped.

After `vc.stop()`, `DiscordVoicePlayer` polls `vc.is_playing()` for up
to 200 ms before releasing the play lock. Songbird's audio thread checks
the stop flag once per 20 ms tick, so the actual wait is one or two
polls; the upper bound is a safety net for a wedged thread. Without
that drain, a barge-in followed by an immediate next-speaker turn
would race: the next `speak()` acquires the lock the instant the prior
call returns, but songbird still has `is_playing() == true` for one tick
— and `vc.play()` returns `PlayError::AlreadyPlaying` ("Already playing audio.").
Pinned by `cancel_then_immediate_speak_does_not_collide` in
`familiar-connect/src/tts_player/discord_player/tests.rs`.

## Cross-speaker reply gate

Turn scopes are keyed per `(channel, user_id)`, so barge-in only ever
cancels within one speaker — a deliberate choice, since the shared
voice client means a global `TTSPlayer.stop()` would cut a *different*
user's in-flight reply. The side effect: when two people talk in one
window, each utterance spawns an independent reply pipeline, and the
two never cancel each other. Without serialization both assemble
before either commits an assistant turn, so both answer the same
moment — the back-to-back near-duplicate replies seen in production
("Fair enough. I'll reserve judgment…" / "Fair enough. I'll form my
opinion later…").

A per-channel `tokio::sync::Mutex` (`VoiceResponder::gate_for`) serializes
reply *generation*: `set_rag_cue` → assemble → stream → assistant-turn
commit run under the lock. The waiting pipeline therefore assembles
only after the prior reply lands in history, sees it in context, and
can resolve `<silent>` instead of duplicating. Two further points:

- **No perceived latency.** Playback is already serial on the shared
  voice client, so the second reply can't be *heard* until the first
  finishes anyway. Gating generation behind the same order spends time
  that was already going to be spent.
- **The user turn stays outside the lock.** Observation is never gated
  by a busy channel — every speaker's turn is recorded even while the
  bot replies to someone else. `set_rag_cue` moves *inside* the lock,
  which also closes a shared-state race where a concurrent pipeline
  could clobber the retrieval cue mid-assemble.

Barge-in composes cleanly: the lock releases on return or cancellation
(guard drop), and same-speaker self-barge still cancels via
the scope. Pinned by the cross-user reply-gate tests in
`familiar-connect/tests/responders_voice.rs`.

## Per-channel tuning

`[channels.<id>]` already covers voice-relevant knobs — trim
`history_window_size` on busy channels to shave LLM prompt + TTFT, drop
expensive layers via `prompt_layers`. See
[Tuning — per-channel overrides](tuning.md#per-channel-overrides). V1
adds strategy-level per-channel overrides once A1 lands.
