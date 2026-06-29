# Activities

The familiar can get up from the screen. Via the `start_activity`
tool she decides to go do something — a walk, an errand, whatever the
operator authors in her catalog — goes absent for a rolled duration,
and comes back with a generated experience. The mental model is a
person at a screen who gets up and comes back: while out she may miss
messages, a real @ping might pull her back early, and a long absence
means she returns with fresh eyes rather than perfect recall.

Absence is **global** (away from the screen), not per-channel: pings
from any subscribed channel count, and the return lands at the
focused text channel. Voice is out of scope in v1 — the engine
refuses to start an activity while a voice subscription is active.

The whole feature is gated on the per-familiar catalog sidecar
`data/familiars/<id>/activities.toml`. The `ActivityEngine` is
constructed only when the catalog is non-empty; with the file missing
or the catalog empty, nothing is wired and behavior is byte-for-byte
unchanged. All policy lives in the tool description and injected
state lines — zero growth to `character.md` or directives.

## Lifecycle

```
start_activity tool call   → ActivityEngine.defer_start(type_id, note)
                               → roll duration from catalog [lo, hi]
                               ↳ returns {ack, label, duration_minutes}

reply ships                → engine.end_turn()
                               → INSERT INTO activities (...)
                               → presence: idle (dnd if unreachable) + activity label
                               → remember departure turn id
                               → arm return timer

(absence — gate decides per inbound message)

timer fires / cut short    → generate experience (background slot)
                               → mechanical event-fact
                               → marked assistant turn in focused channel
                               → archive watermark (long absences)
                               → staged-turn promotion (reads the screen)
                               → missed-ping wake (only with cause)
                               → presence: online
```

### Starting

The model calls `start_activity(activity, note?)` — a text-registry
tool whose `activity` enum is built from the catalog at
registry-build time, so each familiar's sidecar shapes the schema.
The start is **deferred**: the tool stages it and the responder
applies it via `end_turn()` after the current reply ships (same
deferral pattern as `shift_focus`), so she says her goodbye in the
same message. Duration is rolled uniformly from the catalog entry's
`duration_minutes = [lo, hi]` at start. The optional `note` records
intent and seeds the experience later.

The engine also runs an idle-nudge loop (started in
`ActivityEngine.start()`, ~60s tick): when `should_nudge` is
eligible — the focused channel quiet for `idle_nudge_minutes`, no
activity active, at least `min_gap_minutes` since the last return,
local time inside `active_hours` (in the familiar's `display_tz`) —
it publishes a synthetic wake event into the focused text channel
whose content mentions the quiet and offers `start_activity`. The
quiet clock is fed by the responder (`note_traffic` on every handled
text event); firing marks the nudge pending so the debounce window
starts. The nudge only earns the model a turn — going out remains
its decision.

### While out — absence gating

The `TextResponder` consults `ActivityEngine.gate(payload)` at the
top of `handle()`, before any prompt assembly. Three outcomes:

- **Suppress** — non-ping messages are recorded as staged turns
  (history stays complete; at return they are promoted into the
  cross-channel window — see [Returning](#returning)) but produce no
  typing indicator, no LLM call, and no reply. She's away from the
  screen. Suppressed turns whose payload carried `pings_bot` are
  noted live (`note_missed_ping`) for the at-return wake — this is
  what catches cross-channel pings and reply-pings, since the gate
  runs for every subscribed channel.
- **Judgment** — a real @ping in the **focused** text channel while
  the active type is `reachable` earns one normal prose-tier turn
  with a state line appended to the trailing system message, this
  turn only: how long she's been out, who pinged, and that replying
  means heading back while `silent()` means staying out. A real reply
  triggers a cut-short return (`notify_reply_sent`); a silent verdict
  keeps her out. One judgment turn per author per absence: after an
  author's verdict, their further pings are suppressed and noted like
  any other; the latch clears when the next activity starts. Real
  pings from unfocused channels never earn judgment — they are
  suppressed and noted, and surface in the return wake.
- **Unreachable** — a real @ping on a non-`reachable` type is
  suppressed like any other message; it surfaces in the missed-ping
  wake at return.

Her own alarms pierce any absence: a synthetic wake payload carrying
`alarm: True` (the alarm waker's marker) always gates normal —
reachable or not — and replying to it does not cut the absence short.

"Real @ping" means the bot user appears in the message's mentions —
ingest computes a `pings_bot` payload flag covering both `<@id>`
mentions and reply-pings, while role/`@everyone` mentions and bare
name-mentions never count. Payloads without the flag (synthetic
events) fall back to a raw-content mention scan.

### Returning

The return timer (or a cut-short reply) drives one flow:

1. **Experience generation** — the background LLM slot writes a
   short first-person account from the catalog `seed`, the optional
   tool-call `note`, and a cut-short hint. The prompt carries a
   **provenance rail**: experiences are about places, things, and
   herself — never invented claims, conversations, or encounters
   involving real people. An LLM failure degrades to a stock
   one-liner; the return flow always finishes.
2. **Mechanical event-fact** — a direct fact row, no LLM:
   "Sapphire spent Jun 12 afternoon out for a creek walk". The *event*
   is durable memory even though the rich experience text is not.
3. **Marked return turn** — the experience persists as an assistant
   turn (her own narration, not system authority) in the channel
   focused at return, prefixed `[returned from <label>]` and
   tagged `mode = "activity_return"` (`ACTIVITY_RETURN_MODE`). It is
   visible to history and RAG but skipped by the fact extractor,
   keyed on the `mode` column — the prefix is display-only (see
   [Memory](#memory) below).
4. **Archive watermark** — an absence of at least
   `archive_after_minutes` sets the watermark for **every** channel
   (`channel_archive_watermark` table) at the global departure turn
   id: absence is from the whole screen, so one departure point
   breaks every channel's window. The prompt window resets there;
   scrollback does not (see
   [Context interaction](#interaction-with-the-context-pipeline)).
5. **Staged promotion** — staged turns stored since departure are
   promoted (marked consumed) across **all** channels: she reads the
   screen when she gets back, so the cross-channel window shows what
   she missed. Pre-absence staged turns in never-attended channels
   keep their attentional semantics.
6. **Missed-ping wake** — live-noted pings are merged (deduped) with
   a content scan of turns stored since departure in the focused
   channel. If any exist, a synthetic `discord.text` wake event fires
   with auto-injected context: a one-line list of
   missed pings, where the newest three get `turns_around` excerpts
   (~5 turns each), older pings collapse to one-liners, and pings
   already inside the visible last-few window skip excerpts. With no
   missed pings the return is silent — no turn, no announcement
   without cause.
7. **Presence** — status returns to online; while out it was idle
   (yellow) with the catalog `label` as activity text. Unreachable
   types (`reachable = false`) show do-not-disturb (red) instead of
   idle.

### Restart safety

The `activities` table is append-only; the active row has
`actual_return_at` NULL until finished (status `completed` or
`cut_short`). On boot the engine reloads the active row and re-arms
the return timer with a short floor delay (~20s) so bus consumers
and the Discord session exist first — a past-due return fires at
boot + floor rather than inline, which means the return flow (and
its missed-ping wake) lands on a live bus and survives a restart.
The departure turn id is recomputed from turn timestamps (same
precedent as `AlarmScheduler`).

Away presence survives restarts and reconnects via `on_ready`: the
engine starts before Discord login, so its own boot-time presence
call never reaches Discord. After the focus presence sync, `on_ready`
calls the engine's `resync_presence()`, which re-issues idle/dnd plus
label when an activity is still in flight (a no-op when idle or when
activities are disabled). Gateway reconnects re-fire `on_ready`, so a
presence reset mid-activity heals the same way.

## Configuration

Catalog sidecar `data/familiars/<id>/activities.toml` (same pattern
as `lorebook.toml`). The `_default` skeleton ships fully commented
out, i.e. disabled. Missing file or empty catalog disables the
feature; a present-but-invalid file fails loudly with `ConfigError`
so a typo never silently drops a knob.

The shipped skeleton (uncomment and adapt to enable):

```toml
--8<-- "data/familiars/_default/activities.toml"
```

Top-level knobs (all optional):

| Knob | Default | Purpose |
|---|---|---|
| `archive_after_minutes` | `45` | Absence at/above this sets the archive watermark for all channels at the departure turn. |
| `idle_nudge_minutes` | `20` | Focused-channel quiet time before an idle nudge may fire; also the nudge debounce window. |
| `min_gap_minutes` | `90` | Minimum gap after a return before the next nudge. Gates nudges only — never blocks a `start_activity` call. |
| `active_hours` | unset (always) | `"HH:MM-HH:MM"` in `display_tz`; may wrap midnight. Nudges fire only inside this window. Keep it disjoint from the sleep `window` — overlap lets an idle "do something" nudge and the bedtime "go to bed" nudge co-fire in the pre-grace stretch (once force-sleep fires, the active state suppresses the idle nudge). |

Catalog entry (`[[catalog]]`, one per activity type):

| Field | Required | Purpose |
|---|---|---|
| `id` | yes | Stable identifier; becomes a `start_activity` enum value. Must be unique. `sleep` is reserved (below). |
| `label` | yes | Discord presence text while out; also names the activity in turns and facts. |
| `duration_minutes` | yes* | `[lo, hi]` roll range in minutes, `0 < lo <= hi`. *Optional and ignored on the scheduled sleep entry — return is fixed at window end. |
| `reachable` | no (`true`) | A real @ping while out earns a judgment turn; `false` means nothing until return. |
| `content_source` | no (`"authored"`) | Where experience text comes from. Only `"authored"` is valid today; `"adapter"` is a reserved seam for future adapter-backed types (e.g. actually watching a video and reporting on it) and is rejected with an explicit message until implemented. |
| `seed` | yes | Authored prompt seed for experience generation (dream prose for the sleep entry). |

### The reserved `sleep` entry

The catalog id `sleep` is reserved for the [sleep cycle](sleep.md).
Its wall-clock schedule — `window = "HH:MM-HH:MM"` (in `display_tz`,
may wrap midnight) and `grace_minutes` (default 30) — lives in
`character.toml [sleep]`, not on the catalog entry; the entry only
marks which activity the schedule drives. While the entry is otherwise
an ordinary catalog row (the model can `start_activity` into it at the
bedtime nudge), the engine's tick loop owns its schedule: a
once-per-occurrence bedtime nudge at window start, a force-start past
`grace_minutes`, and a wake fixed at the window's end regardless of
start time. Because the wake is fixed,
`start_activity("sleep")` is refused more than an hour before the
window — a midday call would otherwise mean a ~20-hour absence. Sleep
departure fires the hygiene + dream passes in the background, and the
return turn carries the dream prose under `mode = "sleep_return"` — see
[sleep.md](sleep.md) for the full semantics.

The sleep entry in the shipped skeleton above shows the reserved row:
`reachable = false`, an authored dream `seed`, and a comment pointing at
`character.toml [sleep]` for the schedule keys.

## Interaction with the context pipeline

- **`RecentHistoryLayer`** — fetches with
  `recent_cross_channel(respect_archive=True)`, which drops turns
  at/below their channel's archive watermark inside the window query,
  strictly per-channel, before voice coalescing and the silence-gap
  fold so merged turns never smuggle archived text. The window may
  shrink rather than backfill — the archive marks a break. The
  silence-gap fold is independent: a
  long absence is also a wall-clock gap, so on channels with the fold
  enabled the window may already reset at the return boundary even
  below the archive threshold.
- **`read_channel` scrollback** — deliberately ignores the watermark:
  fresh eyes are not a memory hole, and she can scroll back through
  what she missed. The tool grew two paging parameters: `before_id`
  (turns with `id < before_id`) and `around_id` (a `turns_around`
  window centred on a turn id); the two are mutually exclusive.
- **Store** — `recent(..., before_id=)` extends the existing query;
  `recent_cross_channel(..., respect_archive=)` applies the watermark
  filter as a single-SQL correlated subquery outside the latest-N
  window (shrink, not backfill); `turns_around(...)` backs both
  `around_id` paging and the missed-ping wake excerpts.

## Memory

The return turn is visible to history and RAG like any assistant
turn, but `FactExtractor` filters it out of every extraction batch, keyed
on the turns `mode` column (`ACTIVITY_RETURN_MODE`,
`"activity_return"`); the `RETURN_TURN_MARKER_PREFIX` content prefix
(`[returned from `) stays as a display marker only. Experience text
is self-generated fiction — the same claim/fiction discipline that
keeps in-character narration out of the fact store applies, and the
mechanical event-fact already records that the activity happened.
The extractor watermark still advances over skipped turns.

The sleep return turn is the exception: tagged `mode = "sleep_return"`
(`SLEEP_RETURN_MODE`), it **is** processed — with a code-enforced rail
that dream-grounded facts land dream-framed under the `ego:` subject
only. See [sleep.md](sleep.md#dream-aware-extraction).

## Deliberately v1

- **Experiences never enter the fact store.** Only the mechanical
  event-fact is durable; consolidating experience texts into memory
  is deferred to future sleep-cycle work.
- **Voice excluded.** `defer_start` refuses while a voice
  subscription is active; there is no voice-side gating.
- **Live ping capture is in-memory.** While the process runs, the
  gate notes every suppressed `pings_bot` turn (covers cross-channel
  and reply-pings). A restart loses that list; the fallback is the
  content scan of the focused channel, which only finds `<@id>`
  mention strings — a reply-ping landing across a restart can be
  missed. Known gap, accepted for v1.
- **`content_source = "adapter"` is a reserved seam**, not a
  feature: real-content activity types plug in there later without a
  schema change.
