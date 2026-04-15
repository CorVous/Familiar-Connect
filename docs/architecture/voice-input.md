# Voice input

How incoming audio reaches the reply path, plus text input during a voice session.

## Overview

The context pipeline is modality-aware (`ContextRequest.modality` is `"voice"` or `"text"`), providers and processors branch on it, and the budget / channel-mode tables distinguish `imitate_voice` from the text modes. `/subscribe-my-voice` joins the caller's voice channel, keeps a PCM sink open for TTS replies, **and** feeds incoming audio into the same `ConversationMonitor` that text messages go through. Everything downstream (history, memory writes, conversation flow) is unified across modalities.

## STT → pipeline wiring

Wired end-to-end in `bot.py` around the `subscribe_my_voice` command (`bot.py:787–812`):

- Per-speaker PCM streams are pulled from a `discord.py` sink into an asyncio-tagged queue — see `src/familiar_connect/voice/recording_sink.py`.
- `start_pipeline()` in `src/familiar_connect/voice_pipeline.py` creates one `DeepgramTranscriber` per speaker (`src/familiar_connect/transcription.py`) and fans tagged audio to the right stream.
- Each finalised transcript is routed through `VoiceLullMonitor` (`src/familiar_connect/voice_lull.py`) for debouncing. The monitor uses Deepgram-event wall-clock (each `Results(is_final)` or `SpeechStarted` re-arms it) as the silence signal so `voice_lull_timeout` (default 5 s) reflects actual channel-wide quiet, not Deepgram's per-fragment endpointing.
- Discord's client-side VAD stops RTP delivery during silence, so the audio pump watchdogs the per-user queue and sends Deepgram `{"type":"Finalize"}` after `DEFAULT_IDLE_FINALIZE_S` of no chunks (`src/familiar_connect/voice_pipeline.py`) — without this, the buffered final waits for the next speech burst to flush it.
- On each merged utterance, the bot calls `familiar.monitor.on_message(channel_id=voice_channel_id, speaker=..., text=..., is_lull_endpoint=True)` — the same entry point text messages use (see [Conversation flow](conversation-flow.md)).
- On a YES from the side-model gate, `on_respond` dispatches through `_run_voice_response`, which generates the reply and fans out to TTS.
- Per-speaker streams stay separate so transcription attribution matches who actually spoke, not "the channel."

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
