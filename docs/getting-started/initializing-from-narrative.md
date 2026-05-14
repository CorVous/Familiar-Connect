# Initializing a familiar from narrative

`familiar-connect init` bootstraps a new familiar from a folder of
narrative-style markdown. It drives the `background` LLM slot through
generating three authored-canon files:

- `character.md` — first-person persona description, read verbatim by
  `CharacterCardLayer`.
- `character.toml` — minimal overlay (e.g. `display_tz`, `aliases`).
  Deep-merges over `data/familiars/_default/character.toml`, so omitted
  keys keep their defaults.
- `lorebook.toml` — keyword-activated authored canon: places,
  factions, recurring objects, world rules.

## What it does *not* touch

By design, `init` writes only authored canon. It does **not** create
or modify:

- `history.db`
- `facts`
- `people_dossiers`
- `reflections`
- any `accounts` rows

This keeps the trust boundary documented in
[Memory strategies — Why authored canon stays separate from
experiential memory](../architecture/memory-strategies.md#why-authored-canon-stays-separate-from-experiential-memory)
intact. A freshly-initialized familiar starts with rich world canon
and an empty experiential memory; facts and dossiers accumulate from
real conversations.

This is also why `canonical_key`s for narrative-only characters (an
"Aria" who exists in the source markdown but hasn't joined Discord
yet) are not pre-seeded. Lorebook entries are the right home for
named NPCs at init time — once a real user shows up, the experiential
memory layer takes over.

## Usage

```bash
export OPENROUTER_API_KEY=...   # same key as `run` uses
uv run familiar-connect init aria --from path/to/narrative-dir
```

Flags:

- `--from DIR` — directory of `*.md` files describing the character.
  Walked recursively; files are concatenated with `## <relpath>`
  headers in sorted order.
- `--force` — allow overwriting an existing
  `data/familiars/<id>/` folder.
- `--dry-run` — print planned writes; do not touch disk.

The `_default` folder name is reserved and cannot be the target.

## How it talks to the LLM

`init` reuses the `background` slot defined in
`data/familiars/_default/character.toml` ([llm.background]). No new
provider plumbing, no new env vars — `OPENROUTER_API_KEY` is the only
required secret, same as `run`.

Three sequential `chat()` calls run per init:

1. **`character.md`** — persona description, 200-600 words, markdown.
2. **`character.toml`** — JSON object of supported overlay keys,
   serialised back to TOML.
3. **`lorebook.toml`** — JSON object with an `entries` array;
   validated, normalised, written as TOML.

Each call sees the prior output as context, so the persona voice
informs the lorebook tone, and config aliases align with the character
description.

## After init

Run the bot pointed at the new familiar:

```bash
uv run familiar-connect run --familiar aria
```

Edit any of the three files by hand — they're plain markdown / TOML
and the runtime picks up the changes on the next assemble (file
layers content-hash on every assemble, so no restart required for
`character.md` / `lorebook.toml`; `character.toml` is read at startup).
