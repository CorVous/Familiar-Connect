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

To tune *how* the familiar writes in a given channel mode (e.g. "keep
it short, reply like a chat-room message"), drop a Markdown file into
`data/familiars/<id>/modes/<mode>.md`. The filename must match the
mode value: `text_conversation_rp.md`, `full_rp.md`, or
`imitate_voice.md`. Missing file = no per-mode instruction; empty
file = no-op. The text lands in `Layer.author_note` of the system
prompt on every turn whose channel is in that mode.

```
data/familiars/aria/modes/
├── text_conversation_rp.md    # "Reply as if in an internet chat room. A few lines, max."
├── full_rp.md                 # "Prose style. Describe actions in italics. Stay in character."
└── imitate_voice.md           # "Speak naturally. One or two sentences."
```

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
