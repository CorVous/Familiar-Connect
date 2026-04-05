# Voice Channel Text & Media Input + Logging

## Overview

While the bot is active in a voice channel, the associated text channel serves two purposes:
1. **Input** — text messages and images sent there are ingested as conversation input alongside speech
2. **Output** — the bot logs session activity there without spamming new messages

## Text & Image Input

### Text Messages

While the bot is awake in a voice channel, it listens for messages posted in the associated text channel and feeds them into the same conversation pipeline as transcribed speech.

- Messages are attributed to the sender's username (linked to their people entry in the lorebook)
- Text input and voice input are interleaved in the conversation history so the LLM sees a unified stream
- The bot's response goes out over voice (TTS) as normal; the transcript thread also logs the text message as input
- Messages posted by the bot itself (status embeds, thread entries) are ignored as input

### Image Input

Images attached to messages in the voice text channel are passed to the LLM as vision input.

- Supported: images attached directly to a message or posted as Discord image links
- The image is downloaded and passed alongside any text in that message
- The LLM should be a vision-capable model (Claude supports this natively)
- If the model cannot handle vision, log a warning and describe the attachment by filename/type only
- Multiple images in one message are all passed in order
- Images are not stored persistently by default — they are used for the current turn only. If persistence is desired, the lorebook session summarizer should describe notable images in its summary.

### Input Priority & Attribution

When voice and text arrive close together, they are processed in arrival order. The conversation history entry should make the input type clear:

```
[12:34] 🎙 Username: "hey can you look at this"
[12:34] 💬 Username: "what do you think of this image?" [image: screenshot.png]
```

---

## Logging Approaches (Output)

### 1. Live-Edited Status Message (Recommended for session state)

Post a single message when the bot joins the voice channel, then **edit it in place** as the session progresses.

- Use `message.edit()` via py-cord to update the content
- Works well for a rolling "what's happening right now" display:
  - Who is speaking
  - Last transcribed line
  - Bot's last response
  - Session duration / message count
- Discord rate-limits edits to ~5/sec; batch updates to stay well under this
- The message can use an embed for structured layout

**Limitations:** Only shows current state — history is lost as it gets overwritten.

---

### 2. Thread off a Single Message (Recommended for transcript history)

Post one anchor message in the voice text channel, then open a **thread** from it. All transcript entries and session events go into the thread.

- The main channel stays clean — only one message per session is visible
- The thread acts as a scrollable transcript log
- Thread can be archived when the bot leaves the voice channel
- py-cord: `message.create_thread(name="Session - {date}")` then `thread.send()`

**Limitations:** Threads add a small UI step to view history; older threads are auto-archived by Discord after inactivity.

---

### 3. Combination Approach

Use both techniques together:

1. Bot joins voice → posts one embed message in the voice text channel
2. A thread is created from that message, named with the session timestamp
3. The embed is **live-edited** to show current session state (who's talking, last exchange)
4. The thread receives append-only entries: timestamped transcript lines, bot responses, notable events
5. Bot leaves voice → embed is updated to "Session ended", thread is archived

This keeps the main channel to one message per session while preserving full history in the thread.

---

### 4. Single Pinned Message (Simpler alternative)

Maintain one **pinned message** in the voice text channel that is always edited in place — no threads.

- Shows a rolling summary of recent activity (last N exchanges)
- No thread clutter
- History is not preserved — suitable if persistence is handled by the lorebook instead

---

## Recommendation

Use the **combination approach** (option 3):

- One embed per session in the voice text channel (clean, not spammy)
- Thread for full append-only transcript (browsable history)
- Lorebook session summarizer reads the thread content at session end to generate its summary entry

## Implementation Notes

- py-cord's `VoiceChannel.text_channel` property gives the associated text channel in a stage/voice channel (Discord feature); for regular voice channels, a configured text channel ID will be needed
- Edits should be debounced — buffer updates for ~2 seconds before editing to avoid rate limits
- Thread messages can be short: `[12:34] 🎙 User: "can you tell me a story"` / `[12:34] 💬 User: "what do you think?" [image]` / `[12:34] Bot: "Once upon a time..."`
- Text messages and images from the voice text channel are logged in the thread alongside voice transcript entries so the full session history is unified
- Sensitive content (e.g. private conversations) should respect a configurable opt-out so logging can be disabled per-familiar or per-server
