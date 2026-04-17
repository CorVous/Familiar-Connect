# On-disk layout

The bot loads exactly one character per process, picked by `FAMILIAR_ID`
(or `--familiar <id>`), from `data/familiars/<id>/`. The smallest
layout that boots is:

```
data/familiars/aria/
└── character.toml          # optional — defaults apply if missing
```

Everything else (`memory/`, `history.db`, `subscriptions.toml`,
`channels/`, `modes/`) is created on first launch. Multiple character
folders can sit side-by-side under `data/familiars/`; only the one you
point `FAMILIAR_ID` at is loaded per process.

## Giving the familiar a persona

Drop Markdown files into `data/familiars/<id>/memory/self/` (e.g.
`description.md`, `personality.md`, `scenario.md`) — the
`CharacterProvider` concatenates whatever is present. If you want to
seed a familiar from a V3 character-card PNG or a SillyTavern lorebook
export, see the [Bootstrapping guide](../guides/bootstrapping.md) for
the operator recipes.

## Per-mode instructions

Each channel mode can carry its own author-note that gets injected into
`Layer.author_note` on every turn. The lookup order is:

1. **Per-channel backdrop** — set via `/channel-backdrop` (see below).
2. **Familiar's own mode file** — `data/familiars/<id>/modes/<mode>.md`.
3. **Repo default** — `data/familiars/_default/modes/<mode>.md` (ships with
   the repo; covers all three modes out of the box).

Drop a file into `data/familiars/<id>/modes/` to override the default for
that familiar. The filename must match the mode value:
`text_conversation_rp.md`, `full_rp.md`, or `imitate_voice.md`. Empty or
missing file at a tier → the next tier is tried.

```
data/familiars/aria/modes/
├── text_conversation_rp.md    # familiar-specific override
├── full_rp.md
└── imitate_voice.md           # if absent, _default/modes/imitate_voice.md is used
```

## Per-channel backdrop

A **backdrop** is a custom author-note for a single channel that replaces the
mode instruction for that channel. Set it with `/channel-backdrop` — a modal
opens with a multi-line text field; submit to save, submit blank to clear.

The backdrop is stored in `data/familiars/<id>/channels/<channel_id>.toml`:

```toml
channel_name = "general"   # informational; written by the slash command
mode = "full_rp"

backdrop = """
Reply as a stern tavern keeper. Call the user "traveler."
Keep it to two sentences.
"""
```

Switching modes with `/channel-full-rp` and siblings now preserves any
`backdrop` (and `[typing_simulation]` block) that was already in the sidecar.
A malformed sidecar (bad hand-edit, torn write) is discarded with a warning
log on the next `/channel-<mode>` or `/channel-backdrop` invocation, which
rewrites it with valid TOML — so recovery never requires shell access.

Threads and forum posts each get their own sidecar keyed by the thread id. The
`channel_name` field is written as `#general -> brainstorm` so the
file can be found by name when browsing `channels/` directly.

## Example `character.toml`

```toml
default_mode = "text_conversation_rp"   # full_rp | text_conversation_rp | imitate_voice

[providers.history]
window_size = 20

[layers.depth_inject]
position = 0   # SillyTavern @D 0 — immediately before the final user turn
role = "system"
```

See the [Configuration model](../architecture/configuration-model.md)
for the full layout and the rationale behind the
one-active-familiar-per-process rule.
