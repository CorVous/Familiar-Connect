# Voice input

Everything the bot needs to handle incoming audio on the reply path, plus a few adjacent concerns about text and image input that arrive *during* a voice session.

!!! info "Status: Design"
    `/subscribe-my-voice` ships today. It joins the caller's voice channel and keeps the PCM sink open for TTS replies, but **incoming audio is not yet wired into the context pipeline** — this is the last step-7 deferral from the [Context pipeline](../architecture/context-pipeline.md) branch.

## Motivation

The context pipeline is modality-aware (`ContextRequest.modality` is `"voice"` or `"text"`), providers and processors can branch on it, and the budget / channel-mode tables already distinguish `imitate_voice` from the text modes. What's missing is the actual plumbing from Deepgram (or whichever STT provider) back into `ContextPipeline.run()`. Once that lands, voice input is a first-class peer of text input, and everything downstream (history, memory writes, conversation flow) just works.

## Sketch

### STT → pipeline wiring

- The voice subscription already maintains a PCM audio stream per speaker in the channel. That stream becomes the input to an STT client (Deepgram websocket in the first pass).
- Each finalised transcript is built into a `ContextRequest(modality="voice", speaker=..., utterance=..., ...)` and handed to the same `ConversationMonitor` text messages go through (see [Conversation flow](conversation-flow.md)).
- The reply flows back through the pipeline, is sent to Discord as a text message in the voice channel's paired text channel, and fans out to TTS when a voice subscription exists in the same guild.
- Per-speaker streams are kept separate so the transcription attribution matches who actually spoke, not "the channel."

### Barge-in / interruption handling

When the familiar is mid-reply and someone starts talking, the bot needs a policy for whether to keep talking, stop immediately, or finish the current sentence. A separate, more detailed design proposal lives on the [Interruption flow](interruption-flow.md) page — it is scoped specifically to voice as written, and should be rescoped to cover both voice and text latency handling before it ships.

### Text and image input during a voice session

While the bot is active in a voice channel, the associated text channel should also serve as an input source. Text messages and images sent there are ingested as conversation input alongside speech.

- Messages are attributed to the sender's username (and linked to their `people/<slug>.md` if one exists).
- Text input and voice input are interleaved in the conversation history so the LLM sees a unified stream.
- Messages posted by the bot itself (status embeds, thread entries) are ignored as input.
- Images attached to messages in the voice text channel are passed to the LLM as vision input. The LLM should be a vision-capable model (Claude and several OpenRouter models support this). If the configured model cannot handle vision, log a warning and describe the attachment by filename/type only. Multiple images in one message are all passed in order. Images are ephemeral by default — used for the current turn only; persistence strategy to be decided later.
- All input — whether spoken or typed — is treated identically by the LLM once it reaches the pipeline. It is attributed to the sender's username and processed in arrival order. The input method is not surfaced in the conversation history.

## Non-goals

- **Multi-voice-channel support.** The Discord voice API permits one voice connection per gateway session; this constraint stays. Running multiple voice sessions simultaneously means running multiple bot processes (see [Configuration model § voice single-connection limitation](../architecture/configuration-model.md#voice-channel-single-connection-limitation)).
- **Speaker diarisation beyond what Discord already gives us.** Discord's voice API already attributes audio per-user; we rely on that rather than a separate diarisation model.
- **Fully local STT in the first pass.** A hosted STT provider (Deepgram or similar) is fine. Local STT is a later optimisation if it ever becomes a priority.

## Open questions

- **STT provider choice.** Deepgram is the default assumption from earlier planning. Worth confirming before coding; the provider boundary is a small one and easy to swap.
- **Turn segmentation.** When is "the user has stopped talking" fired? Silence threshold? Deepgram's own endpointing? Both? This affects the naturalness of turn-taking.
- **Barge-in / interruption handling.** See [Interruption flow](interruption-flow.md). Needs a scope review (voice-only vs. modality-agnostic) before anything else.
- **Vision fallback UX.** What's the observable behaviour when the user sends an image to a non-vision model? A warning in the same channel? A silent log line? Both?
