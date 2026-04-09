# Memory — freeform text, per familiar

## Overview

A familiar's long-term memory is a directory of plain-text files on disk, one tree per familiar, rooted under `data/familiars/<familiar_id>/memory/`. Multiple character folders can coexist on one install; the bot process picks one via `FAMILIAR_ID` at startup. This document covers *what goes in that directory* — the content conventions. The *pipeline that reads and writes it* lives in `context-management.md`, the architectural principles live in `plan.md` § Context Management, and the configuration model lives in `future-features/configuration-levels.md`.

This document replaces an earlier "lorebook" spec that described structured, tagged entries with triggered retrieval. The new design deliberately drops structured formats in favour of freeform Markdown that a cheap tool-using model can `grep` / `glob` / `read_file` at reply time. Old rationale is preserved in VCS history; see the commit that replaced `lorebook.md` with this file if you need the context.

## Why freeform text

- **Authoring cost is near zero.** The familiar (or a human) writes in Markdown. No schemas to satisfy, no trigger words to author, no tags to maintain.
- **Debuggability is free.** You can `grep -r "Alice"` in your terminal and see the exact same thing the model sees.
- **One tool surface for everything.** Lore, session summaries, per-person notes, per-topic notes, and scratchpad jottings all live in the same directory and are searched by the same tools. No schema-translation layer between different "kinds of memory."
- **Cache invalidation is easy.** Because the directory is the source of truth, any derived artefact (future vector indices, tag caches, graph indexes) is a regenerable cache. If it's out of date, throw it away.
- **SillyTavern interop is an importer, not a runtime.** Lorebook/World-Info JSON is unpacked into Markdown files in the directory at import time. After that it's just text.

## Directory layout

Default path:

```
data/familiars/<familiar_id>/memory/
```

Conventional (not enforced) subdirectories:

```
memory/
    self/
        description.md
        personality.md
        scenario.md
        first_mes.md
        mes_example.md
        system_prompt.md
        post_history_instructions.md
        .original.png              # original character card bytes
    people/
        alice.md
        bob.md
    topics/
        elden-ring.md
        last-tuesdays-argument-about-ska.md
    sessions/
        2026-04-07-evening.md
        2026-04-08-afternoon.md
    lore/
        house-rules.md
        backstory.md
        imported/                  # SillyTavern lorebook import target
            the-old-citadel.md
            the-sisters-oath.md
    notes.md                       # free-form scratchpad
```

Subdirectories are a human-ergonomic suggestion. The search agent treats the whole tree as a single pile of text and will follow Markdown links that point elsewhere in the tree.

## File conventions

- **Plain Markdown.** No YAML frontmatter requirement. If you *want* to put frontmatter on a file that's fine — it'll just be text the model reads along with everything else.
- **One H1 per file, near the top**, describing what the file is about. This helps the search agent find relevant files from filename + first line alone, without needing to read the whole file.
- **Markdown cross-links are encouraged.** If `topics/last-tuesdays-argument-about-ska.md` mentions Alice, it should link to her: `[Alice](../people/alice.md)`. Today this is just a convenience for humans; later it gives us free graph-style traversal without changing the storage format.
- **No required fields.** A person file doesn't have to have "feelings" and "known usernames" as sections; it can have whatever sections the familiar (or a human) felt like writing. The search agent is flexible about structure.
- **Short is fine.** A file can be three sentences. A file can be a single bullet list. There is no minimum.
- **Size caps.** The `MemoryStore` enforces a per-file size cap (default 256 KB) to keep `grep` fast and to prevent one runaway file from eating everything. Hitting the cap is a soft error — the file is still readable, but writes to it fail until something is trimmed.

## Content conventions by subdirectory

These are *conventions*, not requirements. A familiar that doesn't use them is fine; the search agent will still work.

### `self/` — the familiar's self-description

- Written on familiar creation from a Character Card V3 (see `character-card-unpacker` in `context-management.md`).
- The familiar does not normally edit these files. A human can edit them to tune the familiar's persona without re-importing a card.
- `self/.original.png` preserves the original card bytes so a future unpacker revision can re-run against the source.

### `people/<name>.md` — someone the familiar has interacted with

Suggested (not required) sections:

- Who they are, in a sentence or two.
- Usernames the familiar has seen them under, and the contexts each was used in. Notes on whether these are suspected-same-person or confirmed-same-person.
- The familiar's feelings about them — impressions, trust level, emotional history, memorable moments. Updated over time.
- Links to topic files where this person shows up often.

**On multi-username handling.** When a person appears under a new name, the familiar should:

1. Look for clues it may be the same person (they mention a previous conversation, same writing style, explicit self-disclosure, etc.).
2. If it's a *guess*, write a new person file with a note linking to the suspected match, rather than silently merging.
3. If it's *confirmed* (by the person, or by a server admin), the old and new files are merged by the post-session writer pass (or by a human editor).

### `topics/<slug>.md` — recurring subjects

Suggested sections:

- What the topic is — a sentence or two for context.
- The familiar's opinions and feelings — what it thinks, likes, dislikes, or finds interesting. Updated as views develop.
- Notable things said or events relating to this topic.
- Links to the people files of participants.

### `sessions/<date>-<slot>.md` — per-session summaries

- Written by the post-session writer pass (cheap-model call at session end or timeout).
- One file per session. The slot suffix (`evening`, `afternoon`, `short`) disambiguates multiple sessions in one day.
- Contents are freeform: a paragraph summary, a bulleted list of highlights, or whatever the writer pass produces.
- The writer pass is also responsible for *updating relevant people/ and topics/ files* based on what happened in the session.

### `lore/` — persistent background

- House rules, backstory, world-building, anything that's meant to be stable.
- SillyTavern lorebook imports land in `lore/imported/` by default, one Markdown file per ST entry. The importer preserves each entry's original trigger keywords as a bulleted list at the top of the file for human reference, but nothing in the runtime reads them.

### `notes.md` — scratchpad

- A single file for freeform jottings that don't yet deserve their own file.
- The housekeeping pass (future) can propose promoting notes to their own files.

## Writing into memory

Two patterns, in descending order of preference:

1. **Post-session writer pass (preferred).** After a session ends (via `/sleep`, text-session timeout, or an idle window in voice), a cheap side-model reads the session transcript, produces a session summary file, and proposes updates to relevant `people/` and `topics/` files. The writer pass is the *only* code path that mutates memory under normal operation.
2. **In-conversation writer tool (not first pass).** In principle, the main LLM or the content search agent could be given a `write_file` tool during a reply and edit memory live. This is strictly more powerful than the writer pass but also strictly more dangerous (the bot rewriting its own memory in real time, during latency-sensitive voice turns). We do not build this in the first cut. If we build it later, it should be heavily audited via the `MemoryStore`'s audit log and probably feature-flagged per character.

## Reading from memory

Reads happen through `ContentSearchProvider` (see `context-management.md`). The provider gives a cheap tool-using model a deadline-bounded loop with these tools scoped to the familiar's `MemoryStore`:

- `list_dir(path)`
- `glob(pattern)`
- `grep(pattern, path="")`
- `read_file(path)`

The model's job is to find and return the snippets of memory that are actually relevant to the current turn. The provider's job is to run that loop within a deadline, log the tool calls for debuggability, and emit the result as a single `Contribution` that the budgeter can place in the system prompt.

## Future add-ons (not first pass)

All of these are optional enhancements on top of the "just text" baseline. None of them change the storage format. They are listed here so they're considered in later design decisions, not as work items for the current branch.

- **Graph traversal via Markdown links.** A tool (`follow_links(path)`) that the search agent can use to walk from a file to everything it links to, without having to `grep` for the link.
- **Vector index as a second search tool.** If and only if measurements show `grep` getting too slow on large directories, add a `semantic_search(query, k)` tool backed by a local embedding model (e.g. `sentence-transformers`) and a simple on-disk or SQLite vector table. The `ContentSearchProvider` agent chooses when to use it. Grep remains the default.
- **Housekeeping passes.** Cheap-model jobs that scan the directory for duplicate entries, conflicting information, stale beliefs, or notes ready to be promoted to their own files. They propose edits; a human reviews.
- **Per-user personas as first-class people files.** A convention (not a code change) where a Discord user id maps to a `people/<user_id>.md` file. The search agent already handles this, but we could standardise the filename so the writer pass can find existing per-user files without searching for the user's name first.
- **Markdown-link-aware duplicate detection.** The housekeeping pass notices two files that claim to be about the same person but aren't linked to each other, and flags them for merging.
- **Export to SillyTavern lorebook JSON.** The reverse of the importer. Useful if a user wants to move a familiar's memory out to another tool.

## Goals carried forward from the old `lorebook.md`

For future-me grepping for these: the goals from the earlier lorebook spec — per-person tracking, per-topic tracking, per-session summaries, careful multi-username handling, an LLM-driven manager that keeps the main model's context lean — are all preserved. What changed is *how*: the "structured entries with keyword triggers surfaced by a manager LLM" implementation becomes "freeform Markdown files surfaced by a cheap tool-using agent." The manager LLM still exists; it's just called `ContentSearchProvider` now and it has `grep` instead of a tag index.
