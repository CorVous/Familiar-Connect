# Session logging & post-session writer pass

How the familiar captures what just happened when a conversation ends — both as a raw transcript on disk and as a distilled update to its long-term memory.

## Motivation

Two separate problems share the same trigger (a session ending) and deserve a single page:

1. **Keep the raw data forever.** Conversation transcripts and the original (unmodified) character card should be preserved, so that if context-management strategies change in the future no data is lost to a smarter-later-on version of the bot. This is a "don't throw things away" principle.
2. **Distil the session into memory.** Raw history isn't useful to the LLM in full — it's what [`HistoryProvider`](context-pipeline.md#6-historyprovider) compresses into a rolling summary for the system prompt. On top of that, a cheap side-model writer pass reads the just-ended session, produces a session summary file, and proposes targeted edits to the relevant `people/` and `topics/` files. This is the *only* code path that mutates `memory/` under normal operation (see [Memory § writing into memory](memory.md#writing-into-memory)).

Both answer different questions: raw history answers "what was actually said?"; the writer pass answers "what should the familiar remember about this going forward?"

## Raw history preservation

- **`HistoryStore` is the source of truth** for turns — partitioned by `(familiar_id, channel_id)` for the per-channel recent window and by `(familiar_id,)` for the global rolling summary.
- **The base character card is preserved** by the unpacker as `memory/self/.original.png`, alongside the unpacked per-field Markdown files.
- **No pruning policy.** History rows stay forever unless the operator deletes them manually. Disk is cheap; lost transcripts are expensive.

## Post-session writer pass

Implemented in `src/familiar_connect/memory/writer.py` (`MemoryWriter`) and scheduled by `src/familiar_connect/memory/scheduler.py` (`MemoryWriterScheduler`).

### Triggers

The scheduler runs the writer on any of:

- **Turn-count threshold.** After `turn_threshold` (default 50) new turns have accumulated past the last watermark, `notify_turn` fires a run.
- **Idle timeout.** After `idle_timeout` seconds (default 1800) of silence, a timer-driven run checks for unsummarized turns.
- **Explicit flush.** `flush()` is called on unsubscribe events so an ending session is written out promptly rather than waiting for the idle timer.

A per-familiar `asyncio.Lock` serialises runs; overlapping triggers collapse into a single write.

### Inputs

- The session's unsummarized turns from `HistoryStore`, bounded by the writer watermark.
- The familiar's current `memory/people/` and `memory/topics/` files for anyone who participated.
- The current character card. The rest of `memory/` is reachable via `MemoryStore` if the writer asks for it.

### Channel context header

When the writer is constructed with a `channel_context_lookup` callable (wired to `ConversationMonitor.format_channel_context` in the normal process), the prompt transcript is prefixed with a `## Context` block listing every distinct channel the turns came from — for example:

```
## Context
- #general
- #general › feature-brainstorm (thread)
- forum:announcements › hotfix-rollout (forum post)
```

The labels come from `ConversationMonitor`, which stores a `ChannelContext(name, kind, parent_name)` for each subscribed channel. Threads and forum posts therefore surface in session summaries with their human-readable location rather than as bare numeric IDs. Unknown channels (no context registered) are suppressed, keeping the block empty in test setups and during early boot.

### Model

A cheap side-model LLM (`LLMClient`) configured alongside the other side-model calls (rolling summary, side-model gate).

### Outputs

Three kinds of file mutation, all routed through `MemoryStore` so they hit the audit log under the `memory_writer` source:

1. **A new `sessions/<date>-<slot>.md` file.** One per session. The slot suffix disambiguates multiple sessions on one day.
2. **Edits (append or rewrite) to `people/<name>.md`** for each person who showed up. Updates impressions, adds notable moments, flags multi-username suspicions (see [Memory § multi-username handling](memory.md#peoplenamemd-someone-the-familiar-has-interacted-with)).
3. **Edits to `topics/<slug>.md`** for recurring subjects. Updates opinions, adds events, links back to the session file.

After a successful run the writer advances the watermark in `HistoryStore` so the same turns are never written twice.

## Session boundary heuristics

Exact boundaries:

| Modality | Start | End |
|---|---|---|
| Text | First message after a quiet period (or after `/subscribe-text`) | Explicit `/unsubscribe-text`, idle timeout, or turn-count threshold |
| Voice | `/subscribe-my-voice` joins the channel | `/unsubscribe-voice`, idle timeout, or turn-count threshold |

The *same* session can legitimately span both modalities if someone typed in a voice channel's paired text channel. The writer treats interleaved voice and text turns as a single session; see [Voice input § text input during a voice session](voice-input.md#text-input-during-a-voice-session).

## Non-goals

- **In-conversation memory writes.** Letting the main LLM write to `memory/` during a reply is strictly more powerful than the writer pass and strictly more dangerous (bot rewriting its memory in real time during latency-sensitive voice turns). Not in scope. If it comes later, it should be feature-flagged per character and audited via the `MemoryStore` audit log.
- **Automatic history pruning.** Disk space is cheap; lost transcripts aren't.
- **Cross-familiar writer passes.** Each familiar's writer pass runs only against its own memory directory.
- **Housekeeping passes** (duplicate detection, conflict reconciliation, stale-belief flagging). A related but distinct roadmap item — see [Memory § future add-ons](memory.md#future-add-ons).

## Future work

- **Human-in-the-loop review.** The writer currently mutates files directly. A dry-run mode that writes proposed diffs to `sessions/.pending/` for operator review would be safer for early iterations on a new familiar.
- **Partial-session recovery.** If the bot crashes between the last turn and the next scheduler tick, the raw turns are still in `HistoryStore` but no writer pass has run. The watermark covers startup catch-up automatically for the next scheduled run; a dedicated startup sweep would just tighten the worst-case latency.
- **Integration with voice channel logging.** Once [Voice logging](../roadmap/voice-logging.md) ships its live-edited transcript thread, the writer could read the thread as a human-readable cross-check alongside `HistoryStore`.
