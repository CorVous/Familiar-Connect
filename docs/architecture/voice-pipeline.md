# Voice pipeline

How a spoken utterance becomes audible bot speech, and where to swap
stages. [Overview](overview.md) covers the Discord plumbing;
[Streaming bus](streaming-bus.md) covers the in-process event bus.

## Cascaded vs full-duplex

- **Cascaded (STT → LLM → TTS).** Three swappable stages. Persona
  via prompt, brain via OpenRouter, tools available. Production
  stacks (Pipecat, LiveKit, RealtimeVoiceChat) target 700–900 ms
  voice-to-voice.
- **Full-duplex / S2S (Moshi, Sesame CSM).** One model, audio in
  and out, ~200 ms theoretical. LLM brain bundled — no OpenRouter,
  no tool-calling, prompt knobs degraded.

**Familiar-Connect is cascaded by design.** Persona + OpenRouter is
the central knob. Two-stage turn detection and sentence-streaming
TTS close most of the latency gap without forking the architecture.
See [Decisions — full-duplex S2S](decisions.md#full-duplex-speech-to-speech-pipelines-moshi-sesame-csm).
S2S sits on the [roadmap](roadmap.md#v5-full-duplex-s2s-as-a-research-branch)
as a research branch.

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

**Field consensus:** two-stage detection is now the default
everywhere (Pipecat, LiveKit, TEN, Agora). Pure-VAD endpointing is
an anti-pattern.

- *Stage 1* — local VAD (Silero, ONNX, MIT). Fast, in-process.
  Local VAD beats remote by 150–200 ms.
- *Stage 2* — semantic turn classifier over buffered audio.
  Pipecat's Smart Turn v3 (BSD-2, ~12 ms, 360 MB) is the leanest
  open option; trained on filler words STT drops, which is why it
  beats transcription-based endpointing.

**Status:** V1 phase 2 — local endpointer wired behind a feature
flag. Three classes plus a factory live under
`familiar_connect.voice.turn_detection`:

- `SileroVAD(model_path, sample_rate=16000, chunk_size=512)` — Silero
  v5 ONNX. Stateful: feed 32 ms chunks, get speech probabilities;
  call `reset()` between utterances. Returns `is_speech(chunk)` for
  threshold use.
- `SmartTurnDetector(model_path, max_duration_s=16.0)` — Pipecat's
  Smart Turn v3. Stateless: feed the buffered utterance after VAD
  silence. Handles both 2-class softmax and single sigmoid output
  shapes (Pipecat's exports vary). Returns `is_complete(audio)`.
- `UtteranceEndpointer(vad, smart_turn, on_turn_complete, …)` — per-user
  state machine driving the two above over a 48 kHz mono PCM stream.
  Feeds 32 ms VAD windows after a 3:1 boxcar-decimation resample,
  tracks `IDLE → SPEAKING → silence-after-speech → classify`, and
  awaits `on_turn_complete(audio)` on a `complete` Smart Turn verdict.
  An `incomplete` verdict holds the callback until a fresh speech
  burst followed by a fresh silence streak.
- `LocalTurnDetector` (factory) + `create_local_turn_detector_from_env()`
  — bundle of model paths and thresholds. Builds a fresh
  `UtteranceEndpointer` per Discord user (SileroVAD is stateful; Smart
  Turn is shared).

Both ONNX runtimes lazy-import; install via the `local-turn` extra:

```bash
uv sync --extra local-turn
```

ONNX model files are not in the repo. Download separately:

- Silero v5: <https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx>
  (~2 MB)
- Smart Turn v3: <https://huggingface.co/pipecat-ai/smart-turn-v3.0>
  (~360 MB; pull the `.onnx` artifact)

Place under `data/models/` (gitignored) or override the path via env.

### How the audio path forks

When `LOCAL_TURN_DETECTION=1` and the model files exist,
`bot._start_voice_intake` builds a per-user endpointer alongside the
per-user Deepgram clone. The pump feeds every PCM chunk into both:

```
Discord Opus → RecordingSink → per-user PCM
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
                   Deepgram clone       UtteranceEndpointer
                   (endpointing_ms=10,    (Silero VAD + Smart Turn)
                    Finalize-driven)              │
                                                  │ on_turn_complete
                                                  ▼
                                        clone.finalize() ──► Deepgram flush
```

`clone.endpointing_ms` is dropped to `10` for the Deepgram instance
when local detection is active so Deepgram won't endpoint on its own —
it relies on `Finalize` messages driven by the local chain. Selector
TOML lands with [A1](roadmap.md#a1-strategy-swap-configuration-spine);
today the toggle is the env knobs in [Tuning — local turn detection](tuning.md#local-turn-detection-v1).

The default is **off**: with no model files (or `LOCAL_TURN_DETECTION`
unset), the bot keeps using Deepgram's hosted endpointer.

See [Roadmap V1](roadmap.md#v1-local-vad-semantic-turn-detection).

## STT (transcription)

**Today:** `DeepgramTranscriber`. Per-speaker clone-from-template
pattern; one stream per Discord user, lazy-opened, closed after
`idle_close_s`.

**Pluggability:** the clone-template shape is a Protocol seam in
spirit. V3 formalises it as `Transcriber` Protocol;
`FasterWhisperTranscriber` (CTranslate2) and `ParakeetTranscriber`
(Parakeet-TDT 0.6B v3) drop in behind it.

**Partial vs final transcripts.** Modal's benchmark: partials are a
UX feature, not a latency feature. The LLM can't start until the
final, so final-time gates everything. Local-VAD + final-only
Parakeet can beat streaming-Whisper end-to-end. Measure before
optimising for partials.

## LLM

`LLMClient.chat_stream` over OpenRouter. Already streaming, already
cancellable via `TurnScope`. Stays. Lesson: don't waste the streaming
property — feed the next stage incrementally.

## Sentence streaming

`VoiceResponder` feeds each LLM delta through a `SentenceStreamer`
(`familiar_connect.sentence_streamer`) and calls `TTSPlayer.speak`
once per completed sentence. Time-to-first-audio drops from "after
the LLM finishes" to "after the first sentence" — the same 1–3 s
perceived-latency win Pipecat's `SentenceAggregator` ships.

Splitter is abbreviation-aware: `Mr.` / `Dr.` / `etc.` /
single-letter initials (`J. K. Rowling`) don't trip a boundary. A
trailing partial that never reaches a terminator (model omits the
final period) is drained on stream end via `flush()` and spoken as
the last chunk.

**Silent sentinel.** `SilentDetector` runs ahead of the splitter
on every delta. Sentences finalised before the gate decides are
buffered; on `True` they're dropped and TTS is never invoked, on
`False` they flush and the streamer takes over feeding TTS as new
sentences arrive.

**Cancellation.** Each `await self._tts.speak(sentence, scope=...)`
is awaited serially. A barge-in cancels the current `TurnScope`;
`DiscordVoicePlayer`'s poll loop cuts the in-flight sentence within
~20 ms and the responder bails before queueing the next sentence.
The assistant turn is recorded only if the full reply played
without cancellation.

## TTS

Three clients behind `synthesize(text) → TTSResult`:
`AzureTTSClient`, `CartesiaTTSClient`, `GeminiTTSClient`.
`DiscordVoicePlayer` synthesises, mono→stereo, pushes through
pycord. Without a configured client, `LoggingTTSPlayer` logs the
intended speech.

Already a Protocol seam. Adding a backend is one new class.

### Byte-level streaming (Cartesia)

`CartesiaTTSClient` exposes a second method,
`synthesize_stream(text) → AsyncIterator[bytes]`, that yields raw
mono `pcm_s16le` chunks as the WebSocket delivers them. When the
configured TTS client implements this method, `DiscordVoicePlayer`
takes the streaming path:

1. Open Cartesia stream (~140 ms TTFB).
2. Pre-buffer the first chunk into a `StreamingPCMSource` (a
   thread-safe `discord.AudioSource` with `feed` / `close_input`).
3. `vc.play(source)` — pycord's audio thread drains 20 ms frames.
4. A producer task feeds the rest of the stream into the source as
   chunks arrive. `close_input()` on stream end lets the reader
   return `b""` and pycord stop the player cleanly.

That cuts `voice.tts_to_playback` from full-sentence synthesis time
(1.5–3 s for a long sentence on `cartesia-sonic-3` at ~270 ms/word)
down to ~one TTFB. Cancellation: `scope.is_cancelled()` flips
`vc.stop()` within a poll tick; the producer drops out of its loop
on the next `feed` and `close_input` releases any blocked reader.

Azure and Gemini stay on the buffered `synthesize` path (their SDKs
return one big result), so `DiscordVoicePlayer.speak` falls through
to the prior synthesize-then-play behaviour for those clients.

**Mimi-codec lineage.** Mimi (Kyutai, 12.5 Hz frames) is becoming
the open audio-token standard — Sesame CSM, Hibiki, Moshi all use
it. Sesame CSM-1B accepts conversational context for prosody
continuity; voice stability needs fine-tuning. V4 tracks adding a
Sesame or Piper backend when the upstream stabilises.

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
detection (skip the silence timeout). Sentence-level TTS streaming
and byte-level Cartesia streaming both shipped — see
[Sentence streaming](#sentence-streaming) and
[Byte-level streaming](#byte-level-streaming-cartesia).

## Per-turn budget telemetry

`familiar_connect.diagnostics.voice_budget.VoiceBudgetRecorder` (a
process singleton like `SpanCollector`) stamps four phase markers
keyed by `turn_id` and emits one span per adjacent gap into the
shared collector — so `/diagnostics` shows the breakdown in its
existing summary table.

| Phase | Stamp site |
|---|---|
| `stt_final` | `VoiceSource._handle` (just before publishing `voice.transcript.final`) |
| `llm_first_token` | `VoiceResponder._stream_and_speak` on first delta |
| `tts_first_audio` | `VoiceResponder._speak` (deduped — first sentence wins) |
| `playback_start` | `DiscordVoicePlayer.speak` after `vc.play(source)` |

| Span | Gap |
|---|---|
| `voice.stt_to_ttft` | `stt_final` → `llm_first_token` (LLM TTFT, includes assembler) |
| `voice.ttft_to_tts` | `llm_first_token` → `tts_first_audio` (first-sentence completion) |
| `voice.tts_to_playback` | `tts_first_audio` → `playback_start` (TTS synthesis + voice-client lock) |
| `voice.total` | `stt_final` → `playback_start` (user-perceived latency) |

`vad_end` isn't stamped distinctly today: Deepgram's hosted
endpointer fuses VAD-end and final into one `is_final` result.
Roadmap V1 (local Silero VAD) introduces a separate signal; the
recorder is structured to add it without churn.

Recorder is best-effort: the voice path never blocks on it, and
exceptions inside `record(...)` are swallowed so instrumentation
can't take the bot down.

### Prompt cache friendliness

OpenAI's prompt caching matches the longest stable prefix of a
request (1024-token minimum, 128-token granularity). Any change to a
mid-prompt layer cache-invalidates everything after it, so the
`_default_assembler` builds layers in **stability descending** order:

| Position | Layer | Refresh trigger |
|---|---|---|
| 1 | `CoreInstructionsLayer` | file content change |
| 2 | `CharacterCardLayer` | file content change |
| 3 | `OperatingModeLayer` | `viewer_mode` flip (constant per mode) |
| 4 | `ConversationSummaryLayer` | `SummaryWorker` writes (every N turns) |
| 5 | `CrossChannelContextLayer` | any source channel's summary writes |
| 6 | `PeopleDossierLayer` | `PeopleDossierWorker` watermark advances |
| 7 | `RagContextLayer` | per-turn cue (always changes) |
| — | `RecentHistoryLayer` | per-turn (contributes user/assistant messages, not system text) |

`RagContextLayer` therefore sits at the tail of the system message,
so its inevitable per-turn churn invalidates *only* itself — the
prefix from `CoreInstructionsLayer` through `PeopleDossierLayer` can
remain cached when its constituent layers haven't moved.

`tests/test_run_cmd.py::TestDefaultAssemblerLayerOrder` pins this
ordering so a refactor doesn't silently drop into "everything goes
cold" mode. Prompt-cache hit count surfaces as `cached=N` on the
`[LLM call]` log line below — if the count drops to 0, suspect a
mid-prompt layer that just started churning between turns.

### LLM call signals

Every `LLMClient.chat_stream` call adds three spans + one
structured `[LLM call]` log line. The breakdown tells prompt-bloat
apart from OpenRouter routing-tax at a glance.

| Span | Phase |
|---|---|
| `llm.ttfb.<slot>` | request initiation → first response byte |
| `llm.ttft.<slot>` | request initiation → first content delta |
| `llm.total.<slot>` | request initiation → stream end |

The log line carries `slot`, `model`, `chars` (input payload size),
`ttfb_ms` / `ttft_ms` / `total_ms`, and — when the upstream returns
them via OpenRouter's `usage: { include: true }` flag —
`provider`, `in_tokens`, `out_tokens`, and `cached` (prompt-cache
hit count, surfaced when the underlying provider supports it).
`voice.stt_to_ttft` already covers the full STT-to-LLM-first-token
gap; `llm.ttft.<slot>` is the LLM-only slice plus headers. Comparing
the two isolates the assembler / network from raw model latency.

## Barge-in

Already implemented. New `voice.activity.start` cancels prior
`TurnScope`:

1. Cancels in-flight LLM stream (semaphore released on accept so
   cancel isn't starved).
2. Calls `TTSPlayer.stop()` to flush in-flight audio.

Verified sub-200 ms by
`tests/test_voice_responder.py::TestBargeIn`. See
[Voice reply loop](overview.md#voice-reply-loop).

After `vc.stop()`, `DiscordVoicePlayer` polls `vc.is_playing()` for
up to 200 ms before releasing the play lock. Pycord's audio thread
checks the stop flag once per 20 ms tick so the actual wait is one
or two polls; the upper bound is a safety net for a wedged thread.
Without that drain, a barge-in followed by an immediate next-speaker
turn would race: the next `speak()` acquires the lock the instant
the prior call returns, but pycord still has `is_playing() == True`
for one tick — and `vc.play()` raises `ClientException('Already
playing audio.')`. Reproduced (and pinned) in
`tests/test_discord_voice_player.py::TestConcurrentSpeak::test_cancel_then_immediate_speak_does_not_collide`.

## Per-channel tuning

`[channels.<id>]` already covers voice-relevant knobs:

- `history_window_size` — trim recent history on busy channels to
  shave LLM prompt + TTFT.
- `prompt_layers` — drop expensive layers on low-stakes channels.
- `message_rendering` — `name_only` saves DM tokens.

V1 will add strategy-level per-channel overrides once A1 lands.
