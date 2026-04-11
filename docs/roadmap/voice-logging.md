# Voice channel session logging

How to surface "what's happening in the voice channel right now" in the paired text channel without flooding it with new messages.

!!! info "Status: Design"
    Not yet shipped.

## Motivation

While the familiar is active in a voice channel, operators (and other participants) want to see what's being said: live transcripts, the bot's replies, session status. The naive approach — one Discord message per transcript line — drowns the text channel in noise within seconds. Discord rate-limits also kick in long before a busy conversation settles down. We need a layout that shows a rolling "what's happening right now" view and a scrollable transcript, without putting either on the main channel feed.

## Sketch

### Combined approach: live-edited embed + thread

1. **On voice join**, the bot posts one embed message in the paired text channel. That's the *only* top-level message the session creates.
2. **A thread is opened from that message**, named with the session timestamp (`Session — 2026-04-11 21:07`). All transcript entries and session events go into the thread — the main channel stays clean.
3. **The embed is live-edited** to show current session state: who's speaking right now, the last exchange, session duration, and message count. Edits are debounced (batch for ~2 seconds) to stay well under Discord's ~5 edits/sec rate limit.
4. **The thread receives append-only entries**: timestamped transcript lines, bot responses, notable events (join, leave, interruption, Twitch event acknowledgement).
5. **On voice leave**, the embed is updated to "Session ended — {duration}, {message_count} exchanges", and the thread is archived.

This keeps the main channel to one message per session while preserving full history in the thread. The [session logging](session-logging.md) writer pass reads the thread content at session end to generate its memory summary.

### Alternatives considered

- **Single pinned message, live-edited, no thread.** Simpler, but loses history the moment it's overwritten. Suitable only if persistence moves entirely to the memory directory and nobody needs to scroll back in Discord itself.
- **Thread-only, no top-level embed.** Skips the live-state view. Operators then have to open the thread to see if the bot is still active, which is a worse UX.
- **One message per transcript line, no embed, no thread.** Rejected — floods the channel and hits rate limits immediately.

## Implementation notes

- **Associated text channel.** py-cord's `VoiceChannel.text_channel` property gives the paired channel in stage / voice channels (a newer Discord feature). For regular voice channels without that property, the operator configures a text channel ID on the channel config sidecar.
- **Edit debouncing.** Buffer embed updates for ~2 seconds before flushing to avoid Discord's edit rate limits.
- **Thread message format.** Short, regular, timestamp + speaker + text. No distinction between voice and text input:
    - `[12:34] User: "can you tell me a story"`
    - `[12:34] Bot: "Once upon a time..."`
- **Inline image noting.** Images sent during the session are noted inline in the thread: `[12:34] User: "what do you think of this?" [image: screenshot.png]`. Vision input itself is handled at the pipeline level — see [Voice input § text and image input](voice-input.md#text-and-image-input-during-a-voice-session).
- **Privacy opt-out.** Sensitive content (private conversations) should respect a configurable opt-out so logging can be disabled per-familiar or per-channel. Likely lives on `ChannelConfig` as a `session_logging: bool` field.

## Non-goals

- **Logging outside Discord.** This page is only about surfacing the session inside the Discord text channel. Persistent on-disk memory of the session is a separate concern — see [Session logging](session-logging.md).
- **A "what happened while I was away" catch-up feature.** Re-opening the archived thread is the catch-up feature.

## Open questions

- **Thread auto-archive behaviour.** Discord auto-archives threads after inactivity (1h / 1d / 3d / 1w depending on server boost level). Do we explicitly archive on voice leave, or let Discord do it? Explicit is cleaner for the "session ended" marker.
- **What to show in the embed.** A minimal set: "currently speaking: X", "last utterance: ...", "last reply: ...", "session started: ...", "exchanges: N". More may clutter; less may feel stale. Measure in practice.
- **Multi-session overlap.** If the bot joins a different voice channel during an existing session (or is restarted mid-session), what happens to the old embed? Probably edit it to "interrupted / ended".
