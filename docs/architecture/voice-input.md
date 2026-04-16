# Voice input

How incoming audio reaches the reply path, plus text input during a voice session.

## Overview

The context pipeline is modality-aware (`ContextRequest.modality` is `"voice"` or `"text"`), providers and processors branch on it, and the budget / channel-mode tables distinguish `imitate_voice` from the text modes. `/subscribe-my-voice` joins the caller's voice channel, keeps a PCM sink open for TTS replies, **and** feeds incoming audio into the same `ConversationMonitor` that text messages go through. Everything downstream (history, memory writes, conversation flow) is unified across modalities.

## STT → pipeline wiring

Wired end-to-end in `bot.py` around the `subscribe_my_voice` command (`bot.py:787–820`):

- Per-speaker PCM streams are pulled from a `discord.py` sink into an asyncio-tagged queue — see `src/familiar_connect/voice/recording_sink.py`.
- `start_pipeline()` in `src/familiar_connect/voice_pipeline.py` creates one `DeepgramTranscriber` per speaker (`src/familiar_connect/transcription.py`) and fans tagged audio to the right stream.
- In parallel, each tagged chunk is also fed into `TenVadDetector` (`src/familiar_connect/voice/ten_vad.py`) — a local [TEN VAD](https://github.com/TEN-framework/ten-vad) instance per speaker. TEN VAD resamples 48 kHz → 16 kHz, runs in 16 ms hops, and emits edge-triggered `on_speech_start` / `on_speech_end` callbacks that drive `VoiceLullMonitor`. Deepgram is used for transcription text only; its hosted VAD events are no longer consumed.
- Each finalised transcript is routed through `VoiceLullMonitor` (`src/familiar_connect/voice_lull.py`) for debouncing. The monitor uses wall-clock since the last speech event (TEN VAD edge or Deepgram final) so `voice_lull_timeout` (default 5 s) reflects actual channel-wide quiet, not Deepgram's per-fragment endpointing.
- On each merged utterance, the bot calls `familiar.monitor.on_message(channel_id=voice_channel_id, speaker=..., text=..., is_lull_endpoint=True)` — the same entry point text messages use (see [Conversation flow](conversation-flow.md)).
- On a YES from the side-model gate, `on_respond` dispatches through `_run_voice_response`, which generates the reply and fans out to TTS.
- Per-speaker streams stay separate so transcription attribution matches who actually spoke, not "the channel."

## Audio pump & Deepgram flush

`_audio_pump` (`src/familiar_connect/voice_pipeline.py`) drains a per-speaker `asyncio.Queue` of PCM chunks into the speaker's Deepgram WebSocket. It runs as a two-state loop:

| State | Behaviour |
|---|---|
| `drained` | Block on `audio_queue.get()` — no idle window armed. Entered at startup and after each `Finalize`. |
| `dirty` | `wait_for(get(), timeout=DEFAULT_IDLE_FINALIZE_S)`. Real audio refreshes the window; a timeout fires `Finalize` and transitions back to `drained`. Entered on every successful `send_audio`. |

Why it matters: Discord's client-side VAD stops sending RTP packets during silence, so Deepgram's streaming endpointer (`endpointing=300`) never sees the in-stream silence it needs. The buffered final would then sit on Deepgram's side until the next speech burst flushed it — observable in production as multi-second transcript delays. The pump fixes this by sending Deepgram `{"type":"Finalize"}` (`DeepgramTranscriber.finalize`, `src/familiar_connect/transcription.py`) once per gap, forcing immediate emission.

Knobs:

- **`DEFAULT_IDLE_FINALIZE_S`** (`0.5`) — how long to wait with no chunks before flushing. Shorter → faster transcripts; too short and the pump finalises mid-utterance during normal speech jitter. Module-level constant; not currently config-exposed.

Guarantees:

- At most one `Finalize` per silence gap (the `dirty → drained` transition gates re-arming).
- No `Finalize` is sent before the pump has ever seen real audio.
- A real chunk after a flush re-enters `dirty`, arming a fresh window.
- `send_audio` errors are logged and swallowed; the pump keeps draining the queue so it never backs up.

## Deepgram reconnect resilience

`DeepgramTranscriber` (`src/familiar_connect/transcription.py`) reconnects automatically on both clean server-initiated closes (code 1000 — e.g. Deepgram session limit) and abrupt drops (code 1006 — network loss). Auth and billing codes (1008, 4xxx) are treated as permanent and do not retry.

### Replay buffer

Every PCM chunk passed to `send_audio` is appended to a bounded sliding-window buffer. On a successful reconnect the buffer is drained to the new WebSocket before any new audio is sent. This means any outage shorter than `DEEPGRAM_REPLAY_BUFFER_S` results in **zero audio loss** — Deepgram on the new connection receives a seamless continuation of the audio stream.

Buffer budget: `replay_buffer_s × sample_rate × channels × 2` bytes. Oldest chunks are evicted when the budget is exceeded.

After the drain, a `Finalize` control message is sent on the new connection — but only after a short wait equal to the real-time duration of the replayed audio (`replay_bytes / (sample_rate × channels × 2)`) plus a small cushion (`_FINALIZE_POST_REPLAY_BUFFER_S`, default 0.25 s) for server-side jitter. The replay arrives much faster than real-time, and Deepgram's server processes audio at roughly real-time on its side. Firing Finalize immediately emits "what's been transcribed so far", which would be only the first few chunks the server had time to consume — producing a truncated transcript. The delay lets the server catch up before the flush. Without any Finalize at all, the replayed audio would sit in-flight until the user's next utterance, because the pump is VAD-gated and stops feeding frames during silence. If the user is still speaking after the reconnect, the post-reconnect audio flows through normally during the wait (the send lock is released after the drain), so the pre- and post-reconnect segments merge into one transcript when Finalize fires.

### Reconnect timing

- First reconnect attempt: **immediate** (no delay).
- Subsequent failures: exponential backoff — `1 s → 2 s → 4 s … `capped at `DEEPGRAM_RECONNECT_BACKOFF_CAP_S`. Backoff resets when the new connection delivers a real `Results` transcript.

### Environment variable knobs (all optional)

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_REPLAY_BUFFER_S` | `5.0` | Seconds of audio retained for replay on reconnect. |
| `DEEPGRAM_KEEPALIVE_INTERVAL_S` | `3.0` | KeepAlive frame interval; prevents Deepgram's ~10 s idle timeout. |
| `DEEPGRAM_RECONNECT_MAX_ATTEMPTS` | `5` | Max consecutive reconnect failures before giving up. |
| `DEEPGRAM_RECONNECT_BACKOFF_CAP_S` | `16.0` | Maximum reconnect backoff in seconds. |

A structured log line is emitted after each successful recovery: `close_code`, outage duration in seconds, and attempt count.

## Barge-in / interruption handling

Once the familiar is speaking, subsequent user speech is policed by the interruption state machine. See [Voice interruption](interruption.md) for the full design.

## Text input during a voice session

While the bot is active in a voice channel, the associated text channel also serves as an input source. Text messages sent there are ingested as conversation input alongside speech via the normal `on_message` path.

- Messages are attributed to the sender's username (and linked to their `people/<slug>.md` if one exists).
- Text and voice input are interleaved in conversation history so the LLM sees a unified stream.
- Messages posted by the bot itself (status embeds, thread entries) are ignored as input.
- All input — spoken or typed — is treated identically by the LLM once it reaches the pipeline. It is attributed to the sender's username and processed in arrival order; the input method is not surfaced in the conversation history.

## Non-goals

- **Multi-voice-channel support.** The Discord voice API permits one voice connection per gateway session; this constraint stays. Running multiple voice sessions simultaneously means running multiple bot processes (see [Configuration model § voice single-connection limitation](configuration-model.md#voice-channel-single-connection-limitation)).
- **Speaker diarisation beyond what Discord already gives us.** Discord's voice API attributes audio per-user; we rely on that rather than a separate diarisation model.
- **Fully local STT.** A hosted STT provider (Deepgram) is the shipped path. Local STT (faster-whisper) is a later optimisation if it ever becomes a priority.

## Future work

- **Image input during voice sessions.** Vision-capable models (Claude and several OpenRouter models) can take images attached to messages in the voice text channel. Not yet ingested on the voice path; would need a pass-through for attachments plus a fallback (log warning, describe by filename/type) when the configured model is not vision-capable. Persistence strategy (ephemeral vs. saved) also open.
- **STT provider abstraction.** Deepgram is the shipped provider. The provider boundary in `transcription.py` is small and easy to swap; adding an alternative (hosted or local) would slot in behind the same interface.
