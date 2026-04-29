# Voice pipeline

How a user's spoken utterance becomes audible bot speech, and where
to swap stages of the pipeline. This page covers the
*architecture-level* picture; the [overview](overview.md) covers the
specific Discord plumbing, and the
[streaming bus](streaming-bus.md) covers the in-process event bus
that connects the stages.

## The cascaded vs full-duplex divide

Voice agents in 2026 split into two architectures:

- **Cascaded (STT → LLM → TTS).** Three separate models in a
  pipeline. You can swap any stage, prompt-engineer the persona,
  call tools, and route the LLM through OpenRouter. Production
  stacks (Pipecat, LiveKit, RealtimeVoiceChat) target 700–900 ms
  voice-to-voice latency.
- **Full-duplex / speech-to-speech (Moshi, Sesame CSM).** One model
  that ingests audio and emits audio with a text "inner monologue"
  prefix. Latency is dramatically better (200 ms theoretical) but
  the LLM brain is bundled — you give up OpenRouter, tool-calling,
  and most of the prompt knobs.

**Familiar-Connect is cascaded by design.** The character's appeal is
the LLM persona, and the OpenRouter pipeline is non-negotiable. Going
S2S would fork the architecture for a latency win the cascaded path
can mostly close with two-stage turn detection and sentence-streaming
TTS. See [Decisions — full-duplex S2S](decisions.md#full-duplex-speech-to-speech-pipelines-moshi-sesame-csm).

The full-duplex path is on the [roadmap](roadmap.md#v5-full-duplex-s2s-as-a-future-research-branch)
as a research branch, not a near-term plan.

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

Bracketed stages are pluggable. The unbracketed ones (recording sink,
event bus, responder, assembler) are project-specific glue and stay.

## Turn detection

### What ships today

Deepgram's hosted endpointer. The Discord audio pump opens one
Deepgram WebSocket per speaker (cloned from a template); Deepgram
returns finals when its model decides the user is done, biased by
`endpointing_ms` (silence threshold) and `utterance_end_ms` (speech-end
grace window). See
[overview — discord voice](overview.md) and the env vars in
[Tuning — STT](tuning.md#stt-deepgram).

### What the field has converged on

Every production voice-agent stack — Pipecat, LiveKit, TEN, Agora —
has converged on **two-stage turn detection**: a fast local VAD plus
a semantic turn classifier. Pure-VAD endpointing with long silence
timeouts is now an anti-pattern; it either cuts the user off or feels
laggy.

- **Stage 1 — local VAD.** Silero-VAD (ONNX, MIT) detects raw
  speech-vs-silence frames. Fast, runs in-process. Local VAD beats
  remote VAD by 150–200 ms (Pipecat docs are explicit; the wire
  hop is the cost).
- **Stage 2 — semantic turn classifier.** When VAD goes silent, run
  a small classifier over the buffered audio to decide whether the
  user is *done* (vs paused mid-thought). Pipecat's open-source
  Smart Turn v3 (BSD-2, ~12 ms inference, 360 MB) is the leanest
  open option and is trained on filler words ("um", "hmm") that
  STT systems normally drop, which is why it outperforms
  transcription-based endpointing.

### Roadmap

V1 in the [roadmap](roadmap.md#v1-local-vad-semantic-turn-detection)
adds Silero + Smart Turn v3 as the default, with Deepgram's hosted
endpointer remaining as a fallback. The selection is TOML-driven:

```toml
[providers.turn_detection]
strategy = "silero+smart_turn"   # | "deepgram"
```

## STT (transcription)

### What ships today

`DeepgramTranscriber` — streaming WebSocket client, per-speaker clone
pattern (one stream per Discord user, started lazily on first audio
chunk, closed after `idle_close_s` of silence). The instance loaded at
startup is a *template*; `clone()` produces per-user copies so
concurrent speakers don't fight for one stream.

### Pluggability

The transcriber-as-template is already a Protocol seam in spirit; the
roadmap formalizes it. Promoting `DeepgramTranscriber.clone()` and the
`(start, send_audio, finalize, stop) → output_queue` shape into a
`Transcriber` Protocol is V3.

A `FasterWhisperTranscriber` (CTranslate2-backed Whisper, runs locally,
near-universal default for self-hosted STT) and a
`ParakeetTranscriber` (NVIDIA Parakeet-TDT 0.6B v3, beats streaming
Whisper on final-transcript time when paired with external VAD) are
both natural drop-ins behind the same Protocol.

### Partial vs final transcripts

A non-obvious finding from Modal's open-models benchmark: **for
voice-agent latency, partial transcripts are a UX feature, not a
latency feature.** The LLM can't start until the final transcript
arrives, so final-transcript time is the gate. A local-VAD +
final-only Parakeet pipeline can beat streaming-Whisper end-to-end
even though it has no partials. Worth measuring before optimizing
for partials.

## LLM

`LLMClient.chat_stream` over OpenRouter, per-call-site slot
configuration. Already streaming, already cancellable via `TurnScope`.
Stays. The lesson from the field is just to not waste the streaming
property — feed it into the next stage incrementally rather than
buffering until completion.

## Sentence streaming

### What ships today

`VoiceResponder` consumes the LLM stream into a single accumulated
reply, then calls `TTSPlayer.speak(full_reply, scope=...)`. TTS waits
for the full response.

### What the field has converged on

Pipecat's `SentenceAggregator` flushes streaming LLM tokens to TTS at
sentence boundaries. **This single optimization is usually 1–3 s of
perceived-latency win.** Time-to-first-audio drops from "after the
LLM finishes" to "after the LLM's first sentence finishes" — typically
200–400 ms instead of 2–5 s.

### Roadmap

V2 introduces a `SentenceStreamer` between the LLM and the TTS player.
Buffer deltas, emit on sentence boundaries with abbreviation-aware
splitting, feed each sentence to TTS as soon as it's ready.

The `<silent>` sentinel detection (see
[multi-party addressivity](context-pipeline.md#multi-party-addressivity))
moves to the streamer's first-sentence check. Cancellation semantics
stay scope-driven: a cancelled scope flushes all in-flight sentences.

A TOML toggle keeps the old behaviour available for A/B:

```toml
[providers.voice_pipeline]
sentence_streaming = true
```

## TTS

### What ships today

Three TTS clients behind a uniform `synthesize(text) -> TTSResult`
shape: `AzureTTSClient`, `CartesiaTTSClient`, `GeminiTTSClient`. The
`DiscordVoicePlayer` calls `synthesize()`, converts mono PCM to stereo,
and pushes through pycord's voice client. When no TTS client is
configured, `LoggingTTSPlayer` falls back to logging the intended
speech.

### Pluggability

Already a Protocol seam (`TTSPlayer` and the `synthesize()` shape).
Adding a backend is one new class.

### Mimi-codec lineage

Mimi (Kyutai's neural audio codec, 12.5 Hz frame rate) is becoming the
lingua franca of next-generation TTS. Sesame CSM, Hibiki, and Moshi
all use it. Sesame CSM-1B in particular is interesting because it
takes *conversational context* (prior turns of audio + text) and
maintains tone/prosody continuity across a dialogue — the limitation
is voice stability, which fine-tuning addresses.

V4 in the [roadmap](roadmap.md#v4-pluggable-tts-backend-mimi-codec-readiness)
tracks adding a Sesame or Piper backend behind the existing surface
when the upstream is stable. No code change needed today.

## Latency budget

A realistic budget for the cascaded pipeline with cloud STT/TTS, as
of April 2026:

| Stage | Range |
|---|---|
| VAD detects end-of-speech | 50–150 ms tail |
| Semantic turn confirmation (Smart Turn class) | 30–100 ms |
| STT final transcript | 200–400 ms (cloud) / 300–500 ms (Faster-Whisper local) |
| LLM time-to-first-token | 200–500 ms (varies by OpenRouter provider) |
| LLM first-sentence completion | +100–400 ms |
| TTS time-to-first-audio | 100–300 ms (Cartesia is at the fast end) |
| Discord / Opus encoding + jitter | 60–120 ms |
| **Total realistic floor** | **~700 ms** |
| **Comfortable target** | **1.0–1.2 s** |
| **Feels broken above** | **2 s** |

Most of the wins come from:

1. **Local VAD.** The 150–200 ms wire-hop savings dominate other
   tuning.
2. **Semantic turn detection** that doesn't wait on a long silence
   timeout. Today's `endpointing_ms=500` + `utterance_end_ms=1500` is
   conservative; Smart Turn-style classification can fire faster
   without false-cutting.
3. **Sentence-level TTS streaming.** Time-to-first-audio drops from
   "after the LLM finishes" to "after the LLM's first sentence
   finishes."

V1 + V2 in the roadmap target all three.

## Barge-in

Already implemented. A new `voice.activity.start` from any speaker in
the channel cancels the prior `TurnScope`, which:

1. Cancels the in-flight LLM stream (the streaming variant releases
   the rate-limit semaphore on accept, so the cancel isn't starved).
2. Calls `TTSPlayer.stop()` to flush in-flight audio.

Verified end-to-end at sub-200 ms by
`tests/test_voice_responder.py::TestBargeIn`. See
[overview — voice reply loop](overview.md#voice-reply-loop) for the
exact sequence.

The [streaming bus ADR](streaming-bus.md) explains why this is
expressed as a turn-scoped cancellation rather than threaded
cancellation tokens or a separate kill signal.

## Per-channel tuning

`[channels.<id>]` overrides in `character.toml` already apply to
voice-relevant knobs:

- `history_window_size` — trims the recent-history layer for
  high-traffic channels where deeper history would blow the prompt
  budget without helping latency.
- `prompt_layers` — drop expensive layers (e.g. the cross-channel
  context layer) on a low-stakes channel to shave LLM-TTFT.
- `message_rendering` — `name_only` saves tokens in DMs.

V1 (turn detection) and V2 (sentence streaming) will add
strategy-level per-channel overrides once the providers config spine
lands (A1).
