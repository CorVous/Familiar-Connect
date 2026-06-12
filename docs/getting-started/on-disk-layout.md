# On-disk layout

The bot loads one character per process, picked by `FAMILIAR_ID` (or
`--familiar <id>`), from `data/familiars/<id>/`. Smallest layout that
boots:

```
data/familiars/aria/
└── character.toml          # optional — defaults apply if missing
```

`history.db` (Turso, SQLite-compatible) and `subscriptions.toml` are
created on first launch. `fts/turns/` and `fts/facts/` tantivy index
dirs are created lazily alongside the DB. Multiple character folders
can sit side-by-side under `data/familiars/`; only the one
`FAMILIAR_ID` points at is loaded per process.

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
model       = "z-ai/glm-5.1"
temperature = 0.7
reasoning   = "medium"

[llm.background]
model        = "z-ai/glm-5.1"
temperature  = 0.7
reasoning    = "medium"
tool_calling = true

[tts]
provider    = "azure"
azure_voice = "en-US-AmberNeural"
```

See the [Configuration model](../architecture/configuration-model.md)
for the full surface.
