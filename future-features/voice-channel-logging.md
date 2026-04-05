# Voice Channel Conversation Logging

## Problem

Logging conversation activity (transcripts, LLM responses, session status) to a Discord voice text channel without creating a flood of new messages.

## Approaches

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
- Thread messages can be short: `[12:34] User: "can you tell me a story"` / `[12:34] Bot: "Once upon a time..."`
- Sensitive content (e.g. private conversations) should respect a configurable opt-out so logging can be disabled per-familiar or per-server
