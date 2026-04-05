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

The same chattiness system described in `plan.md` (Message Processing & Chattiness) applies here. "Silence" is measured by message inactivity rather than audio silence.

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
