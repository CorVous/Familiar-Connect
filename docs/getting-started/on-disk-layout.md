# On-disk layout

The bot loads one character per process, picked by `FAMILIAR_ID` (or
`--familiar <id>`), from the **familiars root** `<root>/<id>/`. Smallest
layout that boots:

```
<root>/aria/
└── character.toml          # optional — defaults apply if missing
```

`history.db` (Turso, SQLite-compatible) and `subscriptions.toml` are
created on first launch. `fts/turns/` and `fts/facts/` tantivy index
dirs are created lazily alongside the DB. Multiple character folders
can sit side-by-side under the root; only the one `FAMILIAR_ID` points
at is loaded per process.

## Where the familiars root lives

Per-user familiars are stored under the platform per-user data directory
so a `git clean -fdx` in a repo checkout can no longer wipe live state
(issue #201):

| Platform | Default familiars root |
|---|---|
| Linux   | `~/.local/share/familiar-connect/familiars` (honours `XDG_DATA_HOME`) |
| macOS   | `~/Library/Application Support/familiar-connect/familiars` |
| Windows | `%APPDATA%\familiar-connect\familiars` |

Set `FAMILIARS_ROOT` to override the root entirely (it takes top
precedence — useful for tests or a custom data location).

On startup the bot performs a one-shot, best-effort migration: any legacy
familiar folder under the CWD-relative `data/familiars/<id>` (other than
`_default`) is moved into the resolved root. The move is idempotent and
never clobbers a familiar already present at the destination; a familiar
that cannot be moved (e.g. a cross-device rename) is left in place with a
log hint.

The shipped `_default` profile is a **tracked repo resource**, not
per-user state, so it never migrates. It is resolved from the
CWD-relative `data/familiars/_default` (override with
`FAMILIAR_DEFAULTS_ROOT` — e.g. to point a `cargo install`ed binary at
its bundled copy).

An optional `activities.toml` carries the activities catalog —
see [Activities](../architecture/activities.md#configuration).

## Example `character.toml`

```toml
display_tz = "UTC"
aliases    = []

[providers.history]
voice_window_size = 20
text_window_size  = 30

[llm.fast]
model       = "anthropic/claude-haiku-4.5"
temperature = 0.7
reasoning   = "off"

[llm.prose]
model       = "z-ai/glm-5.2"
temperature = 0.7
reasoning   = "medium"

[llm.background]
model        = "z-ai/glm-5.2"
temperature  = 0.7
reasoning    = "medium"
tool_calling = true

[tts]
provider    = "azure"
azure_voice = "en-US-AmberNeural"
```

See the [Configuration model](../architecture/configuration-model.md)
for the full surface.
