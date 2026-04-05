# Text Channel Summoning

## Overview

The familiar can be summoned to a text channel to have a conversation entirely through text, without a voice channel. It can also be dismissed. At any given time, a familiar may only be awake in one place — either a voice channel or a text channel, never both simultaneously.

## Commands

- `/awaken` — summons the familiar to the text channel where the command is run
- `/sleep` — dismisses the familiar from wherever it is currently awake

These commands already exist for voice; they should work the same way in text channels.

## Behavior When Active in a Text Channel

- The familiar reads all messages sent in the channel while it is awake
- It responds in text (no TTS)
- Logging follows the same thread approach as voice channel logging:
  - One anchor message is posted when the familiar awakens, with a thread created from it
  - The thread receives the full conversation transcript
  - When dismissed, the anchor message is updated to "Session ended" and the thread is archived
- Images sent in the channel are ingested as vision input, same as in voice text channels

## Chat Frequency

The same chattiness system from the voice channel applies here. The familiar evaluates each message against the same decision pipeline:

1. **Direct address** (name mention, @mention): Always respond
2. **Direct question to nobody specific**: Roll against chattiness threshold
3. **Silence detection**: If nobody sends a message for N seconds (scaled by chattiness), the bot may interject
4. **Topic relevance**: If the conversation touches the familiar's domain knowledge, increase response probability

The chattiness slider (0–100) maps to the same behavior ranges as voice. Rate limiting also applies — minimum gap between unprompted responses, hard cap on unprompted responses per minute, and a raised threshold when multiple people are actively typing.

## One Location at a Time

A familiar may only be awake in one channel at a time — voice or text. If a `/awaken` command is issued while the familiar is already active elsewhere:

- Inform the user where the familiar is currently awake
- Require an explicit `/sleep` first, or ask for confirmation before moving it

This applies across channels and servers if the familiar is shared across multiple Discord servers.

## Session Handling

Text channel sessions are treated identically to voice sessions for lorebook purposes:

- A session summary is written at the end
- People entries and topic entries are updated as normal
- The only difference is the input modality — no speech transcription involved
