# Memory

A familiar's long-term memory is a directory of plain-text files on disk, one tree per familiar, rooted under `data/familiars/<familiar_id>/memory/`. Multiple character folders can coexist on one install; the bot process picks one via `FAMILIAR_ID` at startup.

This page covers *what goes in that directory* — the content conventions. The *pipeline that reads and writes it* lives in [Context pipeline](context-pipeline.md).

!!! success "Status: Implemented"
    `MemoryStore`, the character-card unpacker, the lorebook importer, `CharacterProvider`, `ContentSearchProvider`, and the **post-session memory writer** are all in place. The writer pass automatically updates `sessions/`, `people/`, and `topics/` files from conversation history.

## Why freeform text

- **Near-zero authoring cost.** Write Markdown. No schemas, no trigger words, no tags.
- **Free debuggability.** `grep -r "Alice"` in your terminal shows the exact same thing the model sees.
- **One tool surface for everything.** Lore, session summaries, per-person notes, per-topic notes, and scratchpad jottings all live in the same directory, searched by the same tools. No schema-translation layer.
- **Easy cache invalidation.** The directory is the source of truth; any derived artefact (vector indices, tag caches, graph indexes) is a regenerable cache.
- **SillyTavern interop is an importer, not a runtime.** Lorebook/World-Info JSON is unpacked into Markdown at import time. After that it's just text.

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
        discord-123456789.md
        twitch-U99.md
        _aliases.json             # display-name → canonical-slug index
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

- **Plain Markdown.** No YAML frontmatter requirement. Frontmatter is fine if you want it — it's just text the model reads along with everything else.
- **One H1 per file, near the top.** Helps the search agent find relevant files from filename + first line alone, without reading the whole file.
- **Markdown cross-links encouraged.** If `topics/last-tuesdays-argument-about-ska.md` mentions Alice, link to her: `[Alice](../people/alice.md)`. Today this is a convenience for humans; later it enables graph-style traversal without changing the storage format.
- **No required fields.** A person file can have whatever sections the familiar (or a human) felt like writing.
- **Short is fine.** Three sentences, a single bullet list — there is no minimum.
- **Size caps.** `MemoryStore` enforces a per-file cap (default 256 KB) to keep `grep` fast. Hitting the cap is a soft error — the file is still readable, but writes fail until something is trimmed.

## Content conventions by subdirectory

These are *conventions*, not requirements. A familiar that doesn't use them is fine; the search agent will still work.

### `self/` — the familiar's self-description

- Written on familiar creation from a Character Card V3 (see [Bootstrapping](../guides/bootstrapping.md)).
- The familiar does not normally edit these files. A human can edit them to tune the familiar's persona without re-importing a card.
- `self/.original.png` preserves the original card bytes so a future unpacker revision can re-run against the source.

### `people/<platform>-<user_id>.md` — someone the familiar has interacted with

Files are keyed on the platform-scoped immutable user id (`Author.slug`), so a Discord display-name change keeps the same file and a repeat Twitch viewer resolves to the same notes across event types. Discord gives `people/discord-<id>.md`; Twitch gives `people/twitch-<id>.md`.

Suggested (not required) sections:

- Who they are, in a sentence or two.
- `## Aliases` — display names, usernames, and nicknames the familiar has seen. The writer pass harvests this list back into `people/_aliases.json` so recall finds the file by any known name.
- The familiar's feelings about them — impressions, trust level, emotional history, memorable moments. Updated over time.
- Links to topic files where this person shows up often.

**On multi-username handling.** The canonical slug is immutable — a display-name change does not create a new file. When a new name is observed, the writer appends it under `## Aliases` in the existing file and rebuilds `_aliases.json`. Cross-platform identity (the same human on Discord and Twitch) is *not* automatically merged: each platform gets its own file. A human editor (or a future housekeeping pass) can link or merge them.

**Legacy history migration.** When upgrading from a pre-`Author` install, `HistoryStore` migrates any `speaker` column into `author_display_name` with a synthesised `legacy-discord` platform tag so historical turns keep their attribution. Pre-existing `people/<name>.md` files are left in place; the memory-writer rewrites them under canonical slugs the next time that person speaks.

!!! note "Cleanup debt from the Author refactor"
    The following is scaffolding that exists only to carry live installs across the speaker → `Author` schema change. Delete once every install has been upgraded:

    - `_migrate_if_needed` branch guarded by `if "speaker" in columns:` in `src/familiar_connect/history/store.py` (copies the old column into `author_display_name` and drops it).
    - The corresponding `test_migration_adds_mode_column_to_existing_db` legacy-schema case in `tests/test_history_store.py`.
    - This note and the "Legacy history migration" paragraph above.

    No `people/<display-name>.md` renamer ships — the writer regenerates under canonical slugs organically, so the old files can be deleted by hand after the cutover.

### `topics/<slug>.md` — recurring subjects

Suggested sections:

- What the topic is — a sentence or two for context.
- The familiar's opinions and feelings — what it thinks, likes, dislikes, or finds interesting. Updated as views develop.
- Notable things said or events relating to this topic.
- Links to the people files of participants.

### `sessions/<date>-<slot>.md` — per-session summaries

- Written by the `MemoryWriter` when a turn-count or idle-timeout threshold is reached.
- One file per date and time-of-day slot (`morning`, `afternoon`, `evening`, `night`). Subsequent writer passes in the same slot re-read the existing summary, merge in the newer turns, and overwrite the file — they do **not** fork a `-2`/`-3` sibling. A long conversation that straddles multiple writer invocations lands in one file.
- Contents are freeform: a paragraph summary, a bulleted list of highlights, or whatever the writer pass produces.
- The writer pass also *creates and updates relevant people/ and topics/ files* based on what happened in the session.

### `lore/` — persistent background

- House rules, backstory, world-building, anything that's meant to be stable.
- SillyTavern lorebook imports land in `lore/imported/` by default, one Markdown file per ST entry. The importer preserves each entry's original trigger keywords as a bulleted list at the top of the file for human reference, but nothing in the runtime reads them.

### `notes.md` — scratchpad

- A single file for freeform jottings that don't yet deserve their own file.
- The housekeeping pass (future) can propose promoting notes to their own files.

## Writing into memory

Two patterns, in descending order of preference:

1. **Post-session writer pass (implemented).** The `MemoryWriter` and `MemoryWriterScheduler` (in `familiar_connect.memory.writer` and `familiar_connect.memory.scheduler`) automatically summarize conversation history into long-term memory files. The writer is triggered by two conditions — whichever fires first:

    - **Turn-count threshold** — after N turns (default 50, configurable via `[memory_writer].turn_threshold` in `character.toml`).
    - **Idle timeout** — after M seconds of silence (default 1800s / 30 min, configurable via `[memory_writer].idle_timeout`).
    - **Flush on unsubscribe** — `/unsubscribe-text` and `/unsubscribe-voice` also trigger an immediate write if there are unsummarized turns.

    The writer calls the `memory_writer` LLM slot (a cheap side-model) with the unsummarized transcript and any existing relevant memory files. The model produces structured output that the writer parses into:

    - A **session summary** → `sessions/<date>-<slot>.md`
    - **People file** creates/updates → `people/<name>.md`
    - **Topic file** creates/updates → `topics/<slug>.md`

    A watermark in the `HistoryStore` tracks which turns have been summarized, so the writer never re-processes old turns and advances only on success. All writes are tagged with `source="memory_writer"` in the `MemoryStore` audit log.

2. **In-conversation writer tool (not first pass).** In principle, the main LLM or the content search agent could be given a `write_file` tool during a reply and edit memory live. This is strictly more powerful than the writer pass but also strictly more dangerous (the bot rewriting its own memory in real time, during latency-sensitive voice turns). We do not build this in the first cut. If we build it later, it should be heavily audited via the `MemoryStore`'s audit log and probably feature-flagged per character.

## Reading from memory

Reads happen through `ContentSearchProvider` (see [Context pipeline](context-pipeline.md#8-contentsearchprovider)), which stacks three tiers:

1. **Deterministic people lookup** — always runs first, no LLM. See [People lookup guarantee](#people-lookup-guarantee) below.
2. **Embedding retrieval** — local fastembed/ONNX against the SQLite cache under `.index/embeddings.sqlite`. Feeds the top-K chunks to tier 3.
3. **Single-shot filter** — one cheap-LLM call (with optional single grep escalation, 2 calls max) selecting what to forward to the main model.

The provider's job is to run all three tiers within a deadline, log decisions for debuggability, and emit the collected `Contribution`s for the budgeter to place in the system prompt.

### People lookup guarantee

Before tier 2/3 run, `ContentSearchProvider` always loads a priority-ordered set of `people/*.md` files into `Layer.content`:

1. **Speaker's file.** `people/<request.author.slug>.md` — an exact lookup by canonical slug (e.g. `discord-123456789`). No fuzzy matching, because the slug comes from the immutable platform id carried on `Author`.
2. **Pending-turn batch.** Every distinct author in `request.pending_turns` — the buffered user messages being responded to. Ensures everyone in the current exchange is on the page.
3. **Recent channel participants.** Up to 5 most-recently-seen distinct users in the channel, via `HistoryStore.recent_distinct_authors`. Keeps the familiar warm on regulars across turns. Limit tunable via `ContentSearchProvider(recent_author_limit=...)`.
4. **Mentioned-name files.** Resolved via `people/_aliases.json`, a `{ "lowercased-name": "author-slug" }` index rebuilt by the writer. The utterance is tokenized (single words, capitalized mid-sentence words, hyphenated phrases); each candidate is looked up in the alias map and the resolved canonical file is loaded.

Each loaded file is truncated to ~800 tokens before emission. The tier emits at priority 85 (between `CharacterProvider` at 100 and the filter tier at 70), with source `content_search.people`. On total-content overflow, tail files are dropped in the same 1→4 order — the speaker always survives, then the batch, then recent participants, with utterance-only mentions dropped first.

This tier runs regardless of tier-2/3 behaviour. Even if the filter emits an empty `ANSWER:`, the retriever returns nothing, or either raises, people files are still surfaced — the familiar does not forget someone it has notes on.

## `MemoryStore` API

`MemoryStore` (in `familiar_connect.memory.store`) owns a per-familiar directory, scoped by `familiar_id`. Default path: `data/familiars/<familiar_id>/memory/`.

API (all synchronous file I/O — these are small text files on local disk):

- `list_dir(rel_path: str = "") -> list[MemoryEntry]`
- `read_file(rel_path: str) -> str`
- `write_file(rel_path: str, content: str)` — writes via temp-file + rename.
- `append_file(rel_path: str, content: str)`
- `grep(pattern: str, rel_path: str = "", case_insensitive: bool = True) -> list[GrepHit]` — uses Python's `re` over the file tree. No shell-out.
- `glob(pattern: str) -> list[str]`

**Path-traversal safety** — every operation resolves against the store's root with `Path.resolve()` and rejects anything outside it. No `..`, no absolute paths, no symlinks out. See [Security](security.md).

**Sanity limits** — per-file size cap (configurable, default 256 KB), per-operation result cap, per-directory file count cap. Exceeding a cap raises a typed exception the search agent can observe.

**Audit log** — every write is logged (file, length, source) so "when did the bot's beliefs about Alice change" has a reproducible answer.

## Derived indices

Markdown is the source of truth. Any index built from it is a regenerable cache — deletable at any time, rebuilt from scratch on the next startup. Indices live under `memory/.index/` and that directory is automatically added to `memory/.gitignore` on first write.

Currently materialised:

- **`memory/.index/embeddings.sqlite`** — SQLite database populated by `ContentSearchProvider`'s embedding retriever (tier 2). Schema: one row per chunk holding the `heading_path`, `char_start`/`char_end`, `token_count`, `text`, and a 384-float `vector` column (little-endian `float32` BLOB, L2-normalised at write time so query is a plain dot product). A separate `documents` table keyed by `rel_path` tracks `mtime` so rebuilds only re-embed changed files.

### Chunking convention

When authoring memory files, write with H2-scoped sections so chunking cleanly follows topic boundaries:

- Files under ~300 words are kept as a single whole-file chunk — fine for short people notes or a two-paragraph topic entry.
- Longer files are sliced per H2 section, with the file's H1 + the H2 heading prepended to the embedding input. A preamble before the first H2 becomes its own chunk tagged with just the H1.
- H2 sections over ~1200 tokens are further split into 800-token sliding windows with 200-token overlap, so a sentence near a boundary is never orphaned.

### Embedding model

The production model is **`BAAI/bge-small-en-v1.5`** (384-dim, ~68 MB INT8-quantised ONNX), loaded via [`fastembed`](https://github.com/qdrant/fastembed) on CPU. Total on-disk footprint including `fastembed` + its transitive deps (`onnxruntime`, `numpy`, `tokenizers`, `huggingface_hub`, `sympy`, `pillow`) is ~370 MB.

**No third-party API calls.** On first run the model weights are fetched from HuggingFace and cached under `~/.cache/fastembed/` (override via `FASTEMBED_CACHE_PATH`). Every subsequent run is fully offline. Air-gapped deployments can pre-seed the cache directory.

Model swap path: to upgrade to `BAAI/bge-base-en-v1.5` (768-dim, ~215 MB, +1.4 MTEB points), change the config key and delete `memory/.index/embeddings.sqlite`. The next startup rebuilds against the new model.

### Rebuild freshness

- **On startup** — `EmbeddingIndex.build_if_stale(store, model)` compares each `(rel_path, mtime)` against the `documents` table. Unchanged files are skipped; changed files are re-embedded transactionally (delete old chunks, insert new) in batches of 32 so ONNX session overhead is amortised. Vanished files are purged.
- **On write** — the memory store is wrapped in an `IndexingMemoryStore` decorator that emits a `rel_path` event on every `write_file` / `append_file`; a background async worker drains the queue and re-embeds each dirty file. The plain `MemoryStore` itself stays index-agnostic.

Because the deterministic people-lookup tier is the correctness floor, retrieval tolerates a missing or partially-built index — the retriever returns `[]`, the filter LLM falls back on just the speaker/mentioned-name files and its own tools.

## Future add-ons

All optional enhancements on top of the "just text" baseline. None of them change the storage format.

- **Graph traversal via Markdown links.** A tool (`follow_links(path)`) that the search agent can use to walk from a file to everything it links to, without having to `grep` for the link.
- **Housekeeping passes.** Cheap-model jobs that scan the directory for duplicate entries, conflicting information, stale beliefs, or notes ready to be promoted to their own files. They propose edits; a human reviews.
- **Markdown-link-aware duplicate detection.** The housekeeping pass notices two files that claim to be about the same person but aren't linked to each other, and flags them for merging.
- **Export to SillyTavern lorebook JSON.** The reverse of the importer.
