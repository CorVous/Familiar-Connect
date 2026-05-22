# Security

Familiar-Connect handles user-provided API keys and tokens (Discord bot, Deepgram, Cartesia, Azure, OpenRouter). Treat all credentials as secrets.

Trust model is single-operator: the admin running the bot has full access to every character's on-disk data. No per-user sandboxing.

## Credential storage

- **Never hardcode tokens or API keys** in source, git-tracked config, or log output.
- Store secrets in environment variables or a **gitignored** `.env`.
- SQLite database files carrying user data stay out of the repo.

## Transport & network

- All external API calls (Deepgram, Cartesia, Azure, OpenRouter, Twitch) use TLS (HTTPS / WSS). Never downgrade to plaintext.

## Logging & error handling

- **Never log secrets.** Sanitize tokens, API keys, and auth headers from log output and error messages.
- Avoid logging full request/response bodies from API calls that may carry keys.

## Input validation

- Sanitize user input from Discord commands and Twitch events before passing to the LLM or storing it.
- Treat all text from external sources (transcripts, Twitch chat, Discord messages) as untrusted.

## Dependency hygiene

- Dependency versions pinned in `pyproject.toml` / `uv.lock` to avoid supply-chain surprises.
- Review new dependencies before adding — prefer well-maintained packages with active security response.

## Principle of least privilege

- Discord bot requests only the permissions it actually needs (voice connect, send messages, use slash commands).
- Third-party API keys use the most restrictive tier/role available.
