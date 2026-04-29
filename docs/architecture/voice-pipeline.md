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
                              [Sentence streamer]   ← roadmap V2
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

**Roadmap:** V1 wires Silero + Smart Turn v3 as default, Deepgram
endpointer as fallback. TOML-driven:

```toml
[providers.turn_detection]
strategy = "silero+smart_turn"   # | "deepgram"
```

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

**Today:** `VoiceResponder` buffers the full LLM reply, then calls
`TTSPlayer.speak`.

**Field consensus:** Pipecat's `SentenceAggregator` flushes to TTS
on sentence boundaries. 1–3 s perceived-latency win; time-to-first-
audio drops from "after the LLM finishes" to "after the first
sentence".

**Roadmap:** V2 introduces a `SentenceStreamer` between LLM and TTS.
Abbreviation-aware splitting. `<silent>` sentinel detection moves
to its first-sentence callback. Cancellation flushes in-flight
sentences. Toggle via:

```toml
[providers.voice_pipeline]
sentence_streaming = true
```

## TTS

Three clients behind `synthesize(text) → TTSResult`:
`AzureTTSClient`, `CartesiaTTSClient`, `GeminiTTSClient`.
`DiscordVoicePlayer` synthesises, mono→stereo, pushes through
pycord. Without a configured client, `LoggingTTSPlayer` logs the
intended speech.

Already a Protocol seam. Adding a backend is one new class.

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

Biggest wins: local VAD (150–200 ms), semantic turn detection
(skip the silence timeout), sentence-level TTS streaming. V1 + V2
hit all three.

## Barge-in

Already implemented. New `voice.activity.start` cancels prior
`TurnScope`:

1. Cancels in-flight LLM stream (semaphore released on accept so
   cancel isn't starved).
2. Calls `TTSPlayer.stop()` to flush in-flight audio.

Verified sub-200 ms by
`tests/test_voice_responder.py::TestBargeIn`. See
[Voice reply loop](overview.md#voice-reply-loop).

## Per-channel tuning

`[channels.<id>]` already covers voice-relevant knobs:

- `history_window_size` — trim recent history on busy channels to
  shave LLM prompt + TTFT.
- `prompt_layers` — drop expensive layers on low-stakes channels.
- `message_rendering` — `name_only` saves DM tokens.

V1 and V2 will add strategy-level per-channel overrides once A1
lands.
