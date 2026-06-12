# Sleep

The sleep cycle is the familiar's nightly consolidation. The full
design has four parts — memory hygiene, a dream pass that forms her
opinions and produces a dream artifact, a behavioral silent window,
and conversation-log archiving — driven as a catalog **activity** (the
authored sleep description seeds the dream prose) with the backend
passes coupled to that activity's lifecycle.

All four parts are now built. **Memory hygiene** (function 1) and the
**dream / opinion-formation** pass (function 2) remain manual and
dry-run-first via `familiar-connect sleep`; they also run
automatically, coupled to the [sleep activity's](#sleep-as-an-activity)
departure. The behavioral silent window is the sleep activity itself
(`reachable = false` suppresses pings; her alarms still pierce), and
log archiving comes free from the activity return's archive watermark
— a full night is far past `archive_after_minutes`.

One sleep runs **hygiene then dream** (`--stage` isolates either for
dev). Hygiene cleans the fact base; dream forms opinions on the cleaned
base, and hygiene's retirements feed the dream prompt as a known-bits
deny-list. The two passes own separate watermark axes (below).

## Memory hygiene

Batch-of-10 fact extraction can't see day-level patterns — a claim
asserted nine times by one speaker and denied by its subject every
time reads, turn by turn, like nine facts. The hygiene pass shows the
LLM the **whole window at once** so it can catch the bit, the
near-duplicate pile-up, and the claim misfiled under the wrong person.

The pass proposes; **code decides**. Every LLM proposal is validated
against safety rails before anything is accepted, and even an accepted
plan is only applied with `--apply`.

### Two action verbs

Both are supersede-only — facts are never deleted, only retired (the
row survives for audit).

- **retire** — drop a fact with no replacement: noise, a duplicate of
  one that's kept, or a bit/claim recorded as a fact.
- **rewrite** — merge near-duplicates or re-attribute a misfiled claim
  into **one** consolidated fact that supersedes the listed old facts.
  Provenance (`source_turn_ids`) is the union of the merged facts.

These cover the hygiene functions from the design: supersede junk
(retire), merge duplicates (rewrite N→1), re-attribute claims (rewrite
with corrected subject). Run-on dossiers heal on their own — the
people-dossier worker rebuilds a subject's dossier from scratch
whenever one of its facts is retired or superseded.

### Rails (enforced in code)

| Rail | Meaning |
|------|---------|
| `unknown_id` | Every referenced id must be a current fact in the window. |
| `self_subject` | A fact under the familiar's own `self:` key (an opinion) is never touched. **Hygiene does not adjudicate feelings** — an opinion changes only through the dream pass (grounded in experience), never the janitor (textual redundancy / contradiction). |
| `duplicate_target` | A fact may be targeted by at most one action. |
| `subject_introduced` | A rewrite may not introduce a person-subject absent from its source facts — the only exception is the familiar's own `self:` key. This carries the extractor's claim-discipline into consolidation: a bit about a person can be re-attributed **to the familiar**, never minted as a new fact **about** the person. |
| `subject_lost` | A rewrite of facts that *had* subjects may not drop them all — that would orphan the fact (NULL subjects) and rebuild the sources' dossiers without its content. Empty keys are rejected when any source had subjects. |
| `noop` | A single-source rewrite that restates the fact unchanged is rejected. |
| `empty_text` | A rewrite must carry non-empty new text. |
| `cap` | Cumulative facts mutated per run ≤ `--cap` (default 50). Excess actions are deferred and recorded. |

The rails gate **subject attribution**, not free text. `new_text` is
unvalidated prose, so a rewrite can *mention* a new person — they get
no subject key, so dossiers and per-subject queries stay clean (the
contamination surface), but the text still reaches FTS/semantic recall.
This is the deliberate boundary: hygiene polices what facts are *about*,
not every word they contain.

`subject_introduced` and the self-key exception are the contamination
guard that the [KaillaDame disengagement] work and the
[self-dossier](memory-strategies.md) substrate established: opinions
and bits about people must inherit the same discipline that real-time
extraction does.

### Authored facts

There is no "untouchable" flag. Authored lore (her tastes, description,
nature, seeded from `seed_turns.toml`) is protected the same way every
other retirement is: the operator reviews the dry-run audit before any
`--apply`. A binary pin proved too blunt — it couldn't tell canonical
lore from a redundant restatement of it, and review handles that nuance.
(If hygiene is ever wired to a clock, it will need a real safety story,
since nothing auto-protects lore.)

## The pass

```
gather_window   → current facts (cap facts_max, newest) + turns since
                  the sleep watermark (cap turns_max) + high-water marks
build_prompt    → whole window rendered, facts grouped with subjects
LLM (background)→ proposes {retire, rewrite}
parse_actions   → permissive JSON parse (garbage → empty plan)
validate        → rails filter proposals → HygienePlan (accepted + rejected)
                  (read-only to here — no DB writes)
apply_hygiene   → only with --apply: execute, advance sleep watermark
hygiene_audit   → JSON artifact written every run (dry or applied)
```

### Sleep watermark (two axes)

`sleep_watermark` (one row per familiar) has two independently-owned
axes: hygiene owns `last_fact_id` (how far the fact base is
consolidated), dream owns `last_turn_id` (how far the log is formed
into opinions). `advance_sleep_watermark` is partial-update — each pass
touches only its own axis, so neither can clobber the other's progress
even within one `sleep` run. Each defines the window its pass covers; a
missed night just widens it, so one run covers the whole gap. A
watermark advances only on `--apply`; a dry run leaves it untouched.

### Concurrency and idempotence

The live bot shares the DB, and the plan→apply gap spans an LLM call
over the whole window. If a writer supersedes a planned fact in that
gap, apply **skips** that action (per-action `ValueError` handling)
rather than crashing mid-loop — recorded in the report and audit. A
rewrite pre-flights its sources and skips the whole action if any is no
longer current, so it never strands a half-merged fact. This also makes
apply idempotent: re-running a partially-applied plan skips what already
happened instead of erroring. The audit is written in a `finally`, so
the run that mutated rows is never the run with no artifact.

### Audit artifact

Every run writes `data/familiars/<id>/sleep_audits/<id>-<UTC>.json`
(override with `--audit-dir`). It records `applied`, the watermark, the
window's considered/truncated counts (truncation is surfaced so a cap
never reads as full coverage), every accepted action with its reason,
every rejected proposal with the rail that caught it and the raw
payload, any actions skipped at apply time, and `notes` (e.g. an
unparseable LLM reply, so a zeroed plan is never silent). This is what
the operator audits before trusting `--apply`, and it is itself
evaluable.

## Dream / opinion formation

The dream pass forms the familiar's **opinions** — her own stances —
from the conversation log, so she stays consistent with how she acts.
An opinion is just a fact routed to her `self:` subject (always-injected
via the self-dossier), **grounded in her log** through `source_turn_ids`
pointing at the turns that demonstrate it. The model proposes; **code
decides**, same as hygiene.

### Two-pass, grounded by construction

The window is the whole log up to the dream watermark, bucketed by
calendar day in `display_tz`. Steady-state (one day's delta) is a single
small pass; the first run is a catch-up over the whole backlog, so:

1. **Per-day candidates** — each day, the model surfaces stance-moments,
   each carrying that day's cited turn-ids. Code keeps only ids that
   really belong to the day (contamination stays local, no cross-day
   drift).
2. **One synthesis** over all candidates — merges restatements, sees the
   whole arc at once. Code enforces every synthesized opinion's
   `source_turn_ids` ⊆ the union of its input candidates' ids, so the
   synthesis **cannot invent grounding**.

### Rails

| Rail | Meaning |
|------|---------|
| `ungrounded` | Source ids must be non-empty and ⊆ the candidate union. An opinion citing nothing real is rejected. |
| `empty_text` | Opinion text must be non-empty. |
| `duplicate` | Normalized-text dedup across the plan — catch-up restates the same stance on different days; only the first is kept. |
| `cap` | Total opinions ≤ `--opinion-cap` (default 60). |

Plus a **flag** (not a rejection): an opinion grounded in no turn the
familiar *herself* authored is surfaced in the audit as
`no_self_authored` — she's a bot, so everything she did is in her own
turns; a stance grounded only in others' messages is the room's, not
hers. `valid_from` is set to the earliest grounding day, so backlog
opinions carry an honest stance timeline even though they're all written
tonight — and a change of mind lands as two facts with ordered
`valid_from`, no supersession needed.

Each minted opinion also carries an LLM-assigned **importance** (1–10),
rated by how durable/central the stance is to who she is (not how
strongly worded): 7–9 a durable core stance, 4–6 characteristic but
situational, 2–3 momentary texture. Out-of-range or missing values
clamp/default to a neutral 5 rather than rejecting the opinion. The
importance feeds the same rank-aware self-dossier as hygiene's facts.

Contamination still in the log (bits, impersonation) is handled without
scrubbing it: hygiene's retirements from the same run feed the dream
prompt as a **known-bits deny-list** — she may have a *take* on a bit
(finding it tedious is real characterization) but never treats it as a
true event.

### Dream audit

The dream audit (`<id>-dream-<UTC>.json`) renders each proposed
opinion's **cited-turn excerpts inline**, so the dry-run review is a skim
— opinion plus the actual turns it stands on, no DB lookups — alongside
the `no_self_authored` flags and the rejected proposals.

## Sleep as an activity

Sleep is a reserved [catalog activity](activities.md) — the entry with
id `sleep`. It reuses the whole activity machinery (departure, return,
presence, archive watermark, staged-turn promotion, missed-ping wake)
and adds a behavioral schedule the engine owns; no separate scheduler
process exists.

### The window

The sleep entry carries two keys no other entry may: `window =
"HH:MM-HH:MM"` (in `display_tz`, may wrap midnight) and `grace_minutes`
(default 30). With a `window`, `duration_minutes` is optional and
ignored — the wake is **fixed at the window's end** (alarm-style),
regardless of when sleep started. The engine's existing ~60s tick
drives the schedule:

- **Bedtime nudge** — when local time enters the window and she is not
  already out, one synthetic bedtime nudge (debounced per window
  occurrence) offers `start_activity` with sleep framing. Going to bed
  willingly stays the model's call.
- **Grace backstop** — past window start + `grace_minutes`, if no
  sleep has *started* this occurrence, the engine force-starts the
  sleep activity directly — no LLM choice. This also covers booting
  mid-window: the first tick force-sleeps. "Already slept this
  occurrence" is judged by the newest sleep row's start time.
- **While asleep** — `reachable = false` suppresses pings like any
  unreachable activity; her own alarms still pierce. A restart
  mid-sleep re-arms the fixed wake from the persisted row.

Keep the catalog's `active_hours` disjoint from the sleep `window`: they
are independent checks on the same tick, so an overlap lets an idle
nudge and the bedtime nudge co-fire in the pre-grace stretch. Once
force-sleep fires, the active state suppresses the idle nudge.

### Lifecycle-coupled passes

Sleep departure fires hygiene then dream (`apply=True`) as a
background task, audits in the usual `sleep_audits/`. The watermarks
define each pass's window, so a missed night just means a wider window
next sleep — one dream, no catch-up worker. Hygiene's retirements feed
the dream's deny-list exactly like a CLI `all` run. A pass failure is
logged and degrades the wake to seed-only prose; it never blocks the
return or crashes the engine.

Right after the passes finish, the engine **produces and persists the
dream prose** — it generates the narration from the seed + freshly
minted opinions, mints the dream-journal `self:` fact, then writes the
prose onto the sleep row's `experience_text`. That column's presence is
the single "dream fully produced" signal. So the dream is durable
within minutes of bedtime and **survives a mid-sleep restart**: the
reloaded active row already carries the prose, and the return reuses it
verbatim — no regeneration, no double journal fact. Prose-gen runs
under its own guard; a failure degrades to the wake fallback rather
than killing the passes. (Residual window: a crash after the journal
append but before the `experience_text` write re-journals seed-only
prose at the next return — a rare possible duplicate; hygiene
reconciles. No two-phase machinery.)

### The dream artifact

At wake, the return turn *is* the dream: `[returned from asleep]
<dream prose>`. Normally the prose was already produced + persisted at
pass-completion (above), so the return reads `experience_text` straight
off the row. If the passes failed or crashed before persisting, the
return falls back to generating prose at wake: the background LLM,
prompted with the sleep entry's authored `seed` (the seed **retunes
dream prose** — that is its design point) plus the night's minted
opinions when available. A one-shot hand-authored first dream can be
placed at `data/familiars/<id>/seed_dream.md`: used **verbatim**
instead of generation, then renamed `seed_dream.consumed.md`
(idempotent, seed_turns spirit).

The prose is also minted as a durable, dream-framed `self:` fact — the
dream-journal **stopgap**; a real `dreams` table is deferred. It is
journaled once, at pass-completion on the normal path (or at wake on
the fallback path). The mechanical event-fact ("spent … night asleep")
is written as for any activity.

### Dream-aware extraction

The sleep return turn carries `mode = "sleep_return"`
(`SLEEP_RETURN_MODE`), not `activity_return`. `FactExtractor` keeps
skipping `activity_return` but **processes** `sleep_return` turns with
dream framing, and the claim-discipline rail is enforced in code, not
just prompt: any fact grounded in a dream turn is forced to the `self:`
subject only and dream-framed ("she dreamed that …") — dream content
never lands under a person's key. Unsourced facts fall back to the
batch *minus* dream turns (return-turn precedent), so real facts about
people stay person-attributable.

## CLI

```
familiar-connect sleep --familiar <id>                  # dry-run (default): hygiene + dream, audits only
familiar-connect sleep --familiar <id> --apply          # execute against the live store
familiar-connect sleep --familiar <id> --stage dream    # one stage only (dev)
```

Flags: `--stage {all,hygiene,dream}`, `--apply`, `--cap N` (hygiene
mutations), `--opinion-cap N` (dream opinions), `--facts-max N`,
`--turns-max N`, `--audit-dir PATH`. Requires `OPENROUTER_API_KEY`; both
passes run on the `background` LLM slot. This remains the manual
operator path alongside the lifecycle-coupled runs.

## Not yet built

A real `dreams` table (the journal currently rides the fact store as
dream-framed `self:` facts), and retiring the rolling summary in favor
of facts + RAG as the continuity bridge. See the roadmap.

[KaillaDame disengagement]: memory-strategies.md
