# Security

Familiar-Connect handles user-provided API keys and tokens (Discord bot token, Twitch OAuth, Deepgram, Cartesia, Azure, OpenRouter). Treat all credentials as secrets.

The overall trust model is single-operator: the admin who runs the bot has full access to every character's memory directly on disk. There is no per-user sandboxing. See [Configuration model § Trust model](configuration-model.md#trust-model) for the rationale. The defensive lines below are the ones that *do* exist.

## Credential storage

- **Never hardcode tokens or API keys** in source code, config files checked into git, or log output.
- Store secrets in environment variables or a `.env` file that is **gitignored**.
- If persisting user-configured keys, encrypt at rest using a machine-local key (e.g. via `cryptography.Fernet` with a key derived from a master secret in an env var).
- SQLite database files containing user data should not be committed to the repo.

## Transport & network

- All external API calls (Deepgram, Cartesia, Azure, OpenRouter, Twitch) must use TLS (HTTPS / WSS) — never downgrade to plaintext.
- The monitoring dashboard (when built — see Roadmap) should bind to `127.0.0.1` by default, not `0.0.0.0`, to avoid exposing it to the network.
- If the dashboard is exposed externally, require authentication (even a simple shared secret or token header).

## Logging & error handling

- **Never log secrets** — sanitize tokens, API keys, and auth headers from log output and error messages.
- Avoid logging full request/response bodies from API calls that may contain keys.
- Use structured logging so sensitive fields can be filtered consistently.

## Input validation

- Sanitize user input from Discord commands and Twitch events before passing to the LLM or storing in the database.
- Treat all text from external sources (transcription output, Twitch chat, Discord messages) as untrusted.
- Apply length limits on user-provided configuration values (personality prompts, familiar names) to prevent abuse.

## `MemoryStore` path safety

The per-familiar memory directory is the one code path that accepts *model-provided* file paths (via `ContentSearchProvider`'s tool-using loop). `MemoryStore` enforces:

- Every operation resolves against the store's root with `Path.resolve()` and rejects anything outside it. No `..`, no absolute paths, no symlinks out.
- Per-file size cap (default 256 KB), per-operation result cap, and per-directory file count cap — exceeding a cap raises a typed exception the search agent can observe and back off from.
- Every write is recorded in an audit log (file, length, source) so "when did the bot's beliefs about Alice change" has a reproducible answer.

Tests for these live in `tests/test_memory_store.py`.

## Dependency hygiene

- Dependency versions are pinned in `pyproject.toml` / `uv.lock` to avoid supply-chain surprises.
- Review new dependencies before adding them — prefer well-maintained packages with active security response.
- Keep dependencies updated for security patches.

## Principle of least privilege

- The Discord bot should only request the permissions it actually needs (voice connect, send messages, use slash commands).
- Twitch token scopes should be minimal — only what EventSub subscriptions require.
- API keys for third-party services should use the most restrictive tier/role available.
