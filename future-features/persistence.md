# Persistence & Context Management — Proposal

## Guiding Principle

Store all raw history permanently so no data is lost. Context management strategies (sliding window size, summarization prompts, RAG retrieval) can evolve over time without re-collecting data. The base character card is stored unmodified alongside any runtime overrides.

---

## Goals

1. **Survive restarts** — conversation history, per-guild Twitch settings, and user facts persist across bot restarts
2. **Bounded context window** — the LLM sees at most ~20 recent exchanges plus a rolling summary, not unbounded history
3. **Full audit trail** — every message is stored permanently for future RAG retrieval and analytics
4. **Per-guild isolation** — each Discord server has its own history, settings, and session state
5. **Foundation for RAG** — the schema supports adding `sqlite-vec` embeddings later without migration headaches

---

## Why SQLite

- Bundled with Python — zero new infrastructure
- Single-file database matches the "single runtime" design goal
- `sqlite-vec` extension adds vector search on the same DB later
- Fast enough for this workload — a few writes per second at most
- Trio-compatible via `trio.to_thread.run_sync` (SQLite I/O is fast, thread offload is fine)

---

## Schema

### `messages` — Full conversation archive

```sql
CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    user_id     INTEGER,              -- NULL for assistant messages
    username    TEXT,                  -- display name at time of message
    role        TEXT NOT NULL,         -- 'user', 'assistant', 'system'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    source      TEXT NOT NULL DEFAULT 'discord'  -- 'discord', 'twitch', 'voice'
);

CREATE INDEX idx_messages_guild_channel ON messages(guild_id, channel_id, created_at);
CREATE INDEX idx_messages_guild_time    ON messages(guild_id, created_at);
```

Every message goes here — user input, assistant replies, Twitch events, transcribed voice. This table is append-only; nothing is ever deleted from it by the context management system.

### `summaries` — Rolling conversation summaries

```sql
CREATE TABLE summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id        INTEGER NOT NULL,
    channel_id      INTEGER NOT NULL,
    summary_text    TEXT NOT NULL,
    messages_from   INTEGER NOT NULL,  -- first message.id covered
    messages_to     INTEGER NOT NULL,  -- last message.id covered
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_summaries_guild_channel ON summaries(guild_id, channel_id, created_at DESC);
```

When messages age out of the sliding window, they get compressed into a summary row. The `messages_from`/`messages_to` columns record which messages were summarized so the process is idempotent and auditable.

### `guild_settings` — Per-guild configuration

```sql
CREATE TABLE guild_settings (
    guild_id    INTEGER PRIMARY KEY,
    data        TEXT NOT NULL,        -- JSON blob
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
```

Stores Twitch config, chattiness level, familiar name, selected providers, etc. as a JSON blob. This keeps the schema stable as settings evolve — no ALTER TABLE needed for new config fields.

### `user_facts` — Per-user persona notes

```sql
CREATE TABLE user_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    fact        TEXT NOT NULL,
    source      TEXT,                 -- where the fact was learned
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_user_facts_guild_user ON user_facts(guild_id, user_id);
```

The familiar remembers things about each user ("Alice is allergic to peanuts", "Bob's favorite game is Elden Ring"). Facts are extracted by the LLM during conversation and stored here. Retrieved during prompt assembly for the RAG context layer.

### `character_cards` — Unmodified base cards

```sql
CREATE TABLE character_cards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    name        TEXT NOT NULL,
    card_json   TEXT NOT NULL,         -- full TavernAI V2/V3 JSON, unmodified
    source_file TEXT,                  -- original filename if loaded from PNG
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
```

Stores the raw character card JSON as-is. If the context management strategy changes in the future, the original card is always available to reprocess.

### Future: `embeddings` (added with RAG)

```sql
-- Added when sqlite-vec is integrated
CREATE VIRTUAL TABLE embeddings USING vec0(
    message_id  INTEGER,
    embedding   FLOAT[1536]           -- text-embedding-3-small dimension
);
```

This is out of scope for the initial persistence work but the schema is designed so it slots in cleanly — `embeddings.message_id` references `messages.id`.

---

## Context Management Strategy

### Sliding Window

- Keep the **last ~20 exchanges** (user + assistant pairs) in the LLM context verbatim
- Count is configurable per guild via `guild_settings`
- Messages are loaded from `messages` table on each LLM call, not kept in memory

### Summarization

When messages age out of the window:

1. Collect all messages between the end of the last summary and the start of the current window
2. Send them to the LLM with a summarization prompt: *"Summarize the following conversation, preserving key facts, decisions, and emotional tone. Be concise (~10:1 compression ratio)."*
3. Store the result in `summaries`
4. The most recent summary is included in the system prompt's "conversation summary" layer

**Trigger:** Summarization runs when the gap between the latest summary's `messages_to` and the window start exceeds a threshold (e.g., 10+ unsummarized messages).

### Prompt Assembly (Updated)

The existing `SystemPromptLayers` in `llm.py` already has the right shape:

```
1. core_instructions    — safety rails, base behavior
2. character_card       — from character_cards table (or loaded PNG)
3. rag_context          — user facts + future embedding retrieval
4. conversation_summary — latest row from summaries table
5. recent_history       — last ~20 messages from messages table
```

No structural changes needed — just populate layers 3–5 from the database instead of from the in-memory `TextSession.history` list.

---

## Module Design

### `db.py` — Database access layer

```python
class Database:
    """SQLite database for conversation persistence."""

    def __init__(self, path: str | Path) -> None: ...

    # -- Setup --
    async def initialize(self) -> None:
        """Create tables if they don't exist."""

    # -- Messages --
    async def store_message(self, guild_id, channel_id, user_id, username, role, content, source) -> int: ...
    async def get_recent_messages(self, guild_id, channel_id, limit=40) -> list[Message]: ...
    async def get_messages_range(self, guild_id, channel_id, after_id, before_id) -> list[Message]: ...

    # -- Summaries --
    async def store_summary(self, guild_id, channel_id, summary_text, messages_from, messages_to) -> int: ...
    async def get_latest_summary(self, guild_id, channel_id) -> Summary | None: ...

    # -- Guild settings --
    async def get_guild_settings(self, guild_id) -> dict: ...
    async def set_guild_settings(self, guild_id, data: dict) -> None: ...

    # -- User facts --
    async def store_user_fact(self, guild_id, user_id, fact, source=None) -> int: ...
    async def get_user_facts(self, guild_id, user_id) -> list[str]: ...

    # -- Character cards --
    async def store_character_card(self, guild_id, name, card_json, source_file=None) -> int: ...
    async def get_character_card(self, guild_id, name) -> str | None: ...
```

All `async` methods use `trio.to_thread.run_sync` internally to run SQLite operations off the event loop.

### Changes to existing modules

| Module | Change |
|---|---|
| `text_session.py` | Replace in-memory `history` list with DB-backed message retrieval. Make registry per-guild. |
| `bot.py` / `on_message` | Store each user message and assistant reply via `Database.store_message()` |
| `llm.py` | No changes — it already accepts `SystemPromptLayers` and `list[Message]` |
| `commands/run.py` | Initialize `Database`, pass it into `create_bot` |

---

## Migration Path

### Phase 1: Message Storage (MVP)
- Create `db.py` with `messages` table only
- Wire `store_message` into the message handler
- Load recent history from DB instead of in-memory list
- Per-guild session isolation

### Phase 2: Summarization
- Add `summaries` table
- Implement summarization trigger and LLM call
- Populate `conversation_summary` layer from DB

### Phase 3: Guild Settings
- Add `guild_settings` table
- Migrate Twitch config persistence (feeds into Twitch slash commands feature)
- Store chattiness, provider selection, etc.

### Phase 4: User Facts
- Add `user_facts` table
- Implement fact extraction prompt (run periodically or after each conversation turn)
- Include user facts in RAG context layer

### Phase 5: RAG Embeddings
- Add `sqlite-vec` dependency
- Create `embeddings` virtual table
- Embed messages on insert
- Retrieve relevant context via cosine similarity during prompt assembly

---

## Open Questions

1. **DB file location:** Default to `~/.familiar-connect/familiar.db`? Or next to the bot's working directory? Should be configurable via env var (`FAMILIAR_DB_PATH`).
2. **Summarization model:** Use the same OpenRouter model as chat, or a cheaper/faster model for summarization?
3. **Fact extraction:** Should user facts be extracted automatically by the LLM, or only when explicitly told ("remember that...")?
4. **Backup strategy:** Since it's a single SQLite file, periodic `.backup()` calls or WAL mode checkpointing may be sufficient. Worth documenting.
5. **Schema migrations:** Use a simple version table + migration scripts, or a lightweight tool like `yoyo-migrations`? For a single-developer project, hand-written migrations with a version check are probably fine.
