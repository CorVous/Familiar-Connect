# Session logging & post-session writer pass

How the familiar captures what just happened when a conversation ends — both as a raw transcript on disk and as a distilled update to its long-term memory.

!!! info "Status: Design"
    `HistoryStore` ships today and captures every turn. The **writer pass** that reads recent history at session end, summarises it into `memory/sessions/<date>-<slot>.md`, and proposes updates to relevant `people/` and `topics/` files is still a roadmap item.

## Motivation

Two separate problems share the same trigger (a session ending) and deserve a single page:

1. **Keep the raw data forever.** Conversation transcripts and the original (unmodified) character card should be preserved, so that if context-management strategies change in the future no data is lost to a smarter-later-on version of the bot. This is a "don't throw things away" principle.
2. **Distil the session into memory.** Raw history isn't useful to the LLM in full — it's what [`HistoryProvider`](../architecture/context-pipeline.md#6-historyprovider) compresses into a rolling summary for the system prompt. On top of that, a cheap side-model writer pass should read the just-ended session, produce a session summary file, and propose targeted edits to the relevant `people/` and `topics/` files. This is the *only* code path that mutates `memory/` under normal operation (see [Memory § writing into memory](../architecture/memory.md#writing-into-memory)).

Both are needed because they answer different questions: raw history answers "what was actually said?", the writer pass answers "what should the familiar remember about this going forward?"

## Sketch

### Raw history preservation

- **`HistoryStore` is already the source of truth** for turns — it partitions by `(familiar_id, channel_id)` for the per-channel recent window and by `(familiar_id,)` for the global rolling summary. Nothing on this page changes that.
- **Base character card is already preserved** by the unpacker as `memory/self/.original.png`, alongside the unpacked per-field Markdown files. Nothing on this page changes that either.
- **No pruning policy.** History rows stay forever unless the operator deletes them manually. Disk is cheap; lost transcripts are expensive.

### Post-session writer pass

- **Trigger.** Fires on one of:
    - `/unsubscribe-text` or `/unsubscribe-voice` from an active channel (explicit end).
    - A text-session idle timeout (no messages in the subscribed channel for N minutes).
    - A voice-session idle window (everyone's been silent for M minutes and the bot has nothing in its buffer).
- **Input.** The session's raw turns from `HistoryStore`, the familiar's current `memory/people/` and `memory/topics/` files for anyone who participated, and the current character card. Everything else in memory is fair game to `read_file` on via `MemoryStore` as the writer pass works.
- **Model.** A cheap side model (same family `HistoryProvider` uses for rolling summaries). Deterministic-mode-injectable for tests, same harness pattern as `ContentSearchProvider`.
- **Outputs.** Three kinds of file mutation, all through `MemoryStore` so they hit the audit log:
    1. **A new `sessions/<date>-<slot>.md` file.** One file per session. The slot suffix (`evening`, `afternoon`, `short`) disambiguates multiple sessions on one day. Contents are freeform — a paragraph summary, a bulleted list of highlights, or whatever the writer pass produces.
    2. **Edits (append or rewrite) to `people/<name>.md`** for each person who showed up. The writer pass updates impressions, adds notable moments, flags multi-username suspicions (see [Memory § multi-username handling](../architecture/memory.md#peoplenamemd--someone-the-familiar-has-interacted-with)).
    3. **Edits to `topics/<slug>.md`** for recurring subjects that came up. Updates opinions, adds new events, links back to the session file.
- **Concurrency.** Only one writer pass runs at a time per familiar. Serialised by a lock on the `Familiar` bundle. Sessions ending back-to-back queue up.

### Session boundary heuristics

A session is whatever the writer pass decides to consume as "one chunk." Exact boundaries:

| Modality | Start | End |
|---|---|---|
| Text | First message after a quiet period (or after `/subscribe-text`) | Explicit `/unsubscribe-text`, or idle timeout |
| Voice | `/subscribe-my-voice` joins the channel | `/unsubscribe-voice`, or voice-session idle window |

The *same* session can legitimately span both modalities if someone typed in a voice channel's paired text channel. The writer pass treats interleaved voice and text turns as a single session; see [Voice input § text and image input](voice-input.md#text-and-image-input-during-a-voice-session).

### Integration with voice channel logging

[Voice channel logging](voice-logging.md) posts a live-edited embed and a per-session thread in the paired text channel. When the writer pass fires, it can read the thread contents as an input alongside `HistoryStore` — the thread is already a human-readable transcript, so it's a useful cross-check.

## Non-goals

- **In-conversation memory writes.** Letting the main LLM write to `memory/` during a reply is strictly more powerful than the writer pass and strictly more dangerous (bot rewriting its memory in real time during latency-sensitive voice turns). Not in the first cut. If it comes later, it should be feature-flagged per character and audited via the `MemoryStore` audit log.
- **Automatic history pruning.** Disk space is cheap; lost transcripts aren't.
- **Cross-familiar writer passes.** Each familiar's writer pass runs only against its own memory directory.
- **Housekeeping passes** (duplicate detection, conflict reconciliation, stale-belief flagging). A related but distinct roadmap item — mentioned on the [Memory](../architecture/memory.md#future-add-ons) page.

## Open questions

- **Idle timeouts.** What are the defaults? Voice is probably tighter than text (5 minutes vs 30?). Configurable per-familiar in `character.toml`.
- **What if the writer pass times out?** Retry on the next session boundary? Flag the session as `pending-writer` on disk so a catch-up job can pick it up? Simplest: log and skip, rely on the next session to pick up the slack.
- **Should the writer pass surface its proposed edits for human review?** Fully automatic is fastest; human-in-the-loop is safer for early iterations. Start automatic; add a dry-run mode that writes proposed diffs to a `sessions/.pending/` directory if we need it.
- **Partial-session recovery.** If the bot crashes mid-session, the raw turns are still in `HistoryStore` but no writer pass has run. The next startup could run a catch-up writer pass against any orphaned session ranges. Nice to have; not day-one.
