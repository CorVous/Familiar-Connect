# Security

Familiar-Connect handles user-provided API keys and tokens (Discord bot token, Deepgram, Cartesia, Azure, OpenRouter). Treat all credentials as secrets.

The overall trust model is single-operator: the admin who runs the bot has full access to every character's data directly on disk. There is no per-user sandboxing.

## Credential storage

- **Never hardcode tokens or API keys** in source code, config files checked into git, or log output.
- Store secrets in environment variables or a `.env` file that is **gitignored**.
- SQLite database files containing user data should not be committed to the repo.

## Transport & network

- All external API calls (Deepgram, Cartesia, Azure, OpenRouter, Twitch) must use TLS (HTTPS / WSS) — never downgrade to plaintext.

## Logging & error handling

- **Never log secrets** — sanitize tokens, API keys, and auth headers from log output and error messages.
- Avoid logging full request/response bodies from API calls that may contain keys.

## Input validation

- Sanitize user input from Discord commands and Twitch events before passing to the LLM or storing in the database.
- Treat all text from external sources (transcription output, Twitch chat, Discord messages) as untrusted.

## Dependency hygiene

- Dependency versions are pinned in `pyproject.toml` / `uv.lock` to avoid supply-chain surprises.
- Review new dependencies before adding them — prefer well-maintained packages with active security response.

## Principle of least privilege

- The Discord bot should only request the permissions it actually needs (voice connect, send messages, use slash commands).
- API keys for third-party services should use the most restrictive tier/role available.
