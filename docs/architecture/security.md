# Security

Familiar-Connect handles user-provided API keys and tokens (Discord bot, Deepgram, Cartesia, OpenRouter). Treat all credentials as secrets.

Trust model is single-operator: the admin running the bot has full access to every character's on-disk data. No per-user sandboxing.

## Credential storage

- **Never hardcode tokens or API keys** in source, git-tracked config, or log output.
- Store secrets in environment variables or a **gitignored** `.env`.
- SQLite database files carrying user data stay out of the repo.

## Transport & network

- All external API calls (Deepgram, Cartesia, OpenRouter, Twitch) use TLS (HTTPS / WSS). Never downgrade to plaintext.

## Logging & error handling

- **Never log secrets.** Sanitize tokens, API keys, and auth headers from log output and error messages.
- Avoid logging full request/response bodies from API calls that may carry keys.

## Input validation

- Sanitize user input from Discord commands and Twitch events before passing to the LLM or storing it.
- Treat all text from external sources (transcripts, Twitch chat, Discord messages) as untrusted.

## Outbound image fetches (`view_image`)

The model never passes a URL. It passes an `image_id` (`img_0`, `img_1`, …)
that the bot minted while reading the message, and `collect_images`
(`src/bot.rs`) builds that map from three sources: attachments, embeds
(preferring Discord's re-hosted `proxy_url`), and **a regex scrape of image
URLs out of the message text**. The third source is attacker-controlled — anyone
in the channel can paste `https://attacker.example/x.png` and it becomes a
fetchable image id. A fetch discloses the operator's IP to that host and
reaches whatever the host is, including addresses only the operator's machine
can route to.

The gate lives at the fetch boundary (`tools::image_policy::UrlGuard`), not at
collection, so all three sources — and any fourth added later — pass through
one check. Two rules apply before a socket opens:

1. **Unconditional.** Scheme must be `http` or `https`, and every address the
   host resolves to must be public unicast. Loopback, RFC1918, link-local
   (including the `169.254.169.254` cloud-metadata address), CGNAT,
   documentation, benchmarking and reserved ranges are refused, as are their
   IPv6 equivalents (ULA, link-local, v4-mapped, v4-compatible, NAT64-embedded).
   The check runs on the *resolved* addresses, so a public name pointing at
   `127.0.0.1` is refused too. **No configuration disables this.**
2. **Host allowlist.** `[tools].trusted_image_hosts` is default-deny; a host
   outside it is refused before any network call, logged once, and the tool
   returns the refusal to the model. `[tools].allow_untrusted_image_urls =
   true` drops this rule — rule 1 still applies.

Redirects are followed by hand (max 5 hops) rather than by `reqwest`, because a
permitted host can `302` to a private address and `reqwest`'s redirect hook is
synchronous — it cannot re-resolve. Every hop re-enters the same guard. The
connection is then pinned to the addresses the guard validated
(`resolve_to_addrs`), so a second lookup cannot answer differently (DNS
rebinding). Residual gap: an attacker-controlled authoritative server can still
return a public address that is *also* reachable internally; nothing here helps
against that.

Fetches remain capped at 15s, 20 MiB, and an image content-type allow-list.

## Dependency hygiene

- Dependency versions pinned in `Cargo.toml` / `Cargo.lock` to avoid supply-chain surprises.
- Review new dependencies before adding — prefer well-maintained packages with active security response.

## Principle of least privilege

- Discord bot requests only the permissions it actually needs (voice connect, send messages, use slash commands).
- Third-party API keys use the most restrictive tier/role available.
