# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

---

## Goals

- **Single runtime**: The entire backend runs as one Python process using `trio`. No separate worker scripts, no external message broker.
- **Unified entry point**: One `python main.py` starts everything — Discord gateway, voice capture, transcription, LLM, TTS, and Twitch listener all run as concurrent tasks in a `trio.Nursery`.

---

## Target Architecture

All components run as coroutines within a single `trio` event loop:

```
Discord Voice → audio capture → trio.MemoryChannel
                                      ↓
                            Transcription (Deepgram streaming)
                                      ↓
                  trio.MemoryChannel (text) ← Twitch Events
                                      ↓
                          Message Processor + Chattiness
                                      ↓
                           Claude LLM + Conversation History
                                      ↓
                              TTS (Cartesia / Azure) → Audio
                                      ↓
                    trio.MemoryChannel → Discord Voice Playback
```

Use red/green A/B TDD.

---

## Core Features

### Discord Bot
Built with **py-cord**. Voice send/receive uses **davey** to handle Discord's DAVE (Audio/Video E2E Encryption) protocol.

- **`/awaken`** — Joins your voice channel, captures audio in real-time from all speakers
- **`/sleep`** — Gracefully disconnects
- **`/setup`** — Configuration wizard (UI-driven via Discord modals) to set:
  - Familiar name, personality prompt, chattiness preset (Quiet / Reserved / Moderate / Talkative / Very Talkative)
  - Which transcription/LLM/TTS provider to use
  - Model selection, temperature

Bot token in `.env` as `DISCORD_BOT`.

### Transcription

**Primary: Deepgram (Nova-2, streaming)**
- Native WebSocket streaming API, ~300ms latency, strong accuracy
- ~$0.0043/min pay-as-you-go, $200 free credit on signup
- Handles raw PCM streams directly — maps well to Discord's audio pipeline
- Good Python SDK (`deepgram-sdk`)

**Fallback: faster-whisper (local)**
- Zero cost, no rate limits, no external dependency
- Requires a GPU for real-time performance (large-v3 model, ~1-2s per 5s chunk on RTX 3060+)
- Good offline / privacy-preserving option

Pipeline: Discord 48kHz Opus → decode to PCM → resample to 16kHz → stream to Deepgram WebSocket (or feed chunks to faster-whisper).

### Message Processing & Chattiness

The bot evaluates each incoming event (transcription, text message, Twitch event) against a decision pipeline to determine whether to respond.

> **Future consideration:** Using a smaller/cheaper model to make response-worthiness decisions may be a better strategy than hardcoded heuristics, but is deferred for now due to implementation complexity.

**Response decision heuristics (evaluated in priority order):**
1. **Direct address** (name mention, @mention, "hey familiar-name"): Always respond.
2. **Direct question to nobody specific** ("does anyone know..."): Roll against chattiness threshold.
3. **Silence detection**: If nobody speaks for N seconds (scaled by chattiness), the bot may interject.
4. **Topic relevance**: If the conversation touches the familiar's domain knowledge, increase response probability by ~20-30%.
5. **Twitch events**: Always acknowledge subs/bits/raids. Follows only at Moderate or above.

**Chattiness presets:**

| Preset | Behavior |
|--------|----------|
| **Quiet** | Only respond when directly addressed by name |
| **Reserved** | + Respond to direct questions aimed at nobody specific |
| **Moderate** | + Interject after prolonged silence. React to Twitch events. |
| **Talkative** | + Probabilistic response to general conversation |
| **Very Talkative** | + Comment on most topics, shorter silence threshold, react to almost everything |

**Turn-taking rules:**
- Wait 1.5–2s after the last speaker finishes (VAD silence) before starting a response
- If someone starts speaking while the bot is generating but hasn't started outputting audio, abort
- If already speaking, finish the current sentence then yield
- If interrupted twice in 60s, double the silence threshold temporarily

**Rate limiting:**
- Minimum 15–30s between unprompted responses (shorter for chattier presets)
- Hard cap: max 3 unprompted responses per minute
- If 3+ humans are actively talking (multiple speakers in last 10s), raise the response threshold — talk less in fast-moving conversations, not more

### AI Response (Claude LLM)

**Context management:** Hybrid sliding-window + summarization.
- Keep the last ~20 exchanges verbatim in context
- When messages age out, compress them into a rolling conversation summary (~10:1 ratio)
- Store all raw messages in SQLite with timestamps for RAG retrieval

**RAG with sqlite-vec:**
- Use `sqlite-vec` for cosine similarity search over embeddings
- Embed message chunks and persona-relevant facts using `text-embedding-3-small`
- Store embeddings alongside message metadata (timestamp, user, channel) for filtered retrieval
- Lightweight, no external vector DB needed for single-server use

**Character system (borrowing from SillyTavern):**
- Adopt the TavernAI Character Card V2 JSON schema as config format
- Separate fields: description, personality summary, example dialogues, scenario
- Per-user "persona" notes so the familiar remembers facts about each user
- No direct SillyTavern integration (it's a UI-focused Node.js app, not a library)

**System prompt layering:**
1. Core instructions and safety rails
2. Character card (personality, speech patterns, backstory)
3. Retrieved RAG context
4. Conversation summary
5. Recent message history

**Token budget (~30k per call for Sonnet, ~8k for Haiku):**
- System prompt + character card: ~2k tokens
- RAG retrieved chunks: ~3k tokens
- Conversation summary: ~1k tokens
- Recent history: ~20k tokens (Sonnet) / ~4k tokens (Haiku)
- Response headroom: ~4k tokens (capped at 200 tokens for voice output)

Configurable model, temperature. Max 200 output tokens for voice responses to keep latency low.

### Text-to-Speech

**Primary: Cartesia Sonic**
- Purpose-built for real-time conversational AI
- Sub-100ms time-to-first-byte — best-in-class latency
- Native WebSocket streaming, quality rivaling ElevenLabs at lower cost
- Outputs 44.1kHz PCM natively
- Voice cloning support

**Secondary: Azure Speech (Neural)**
- Keep the 9 original Azure voices for nostalgia
- ~$16/M chars, ~100-200ms latency, rock-solid reliability
- Mature Python SDK, good fallback if Cartesia has downtime

**Budget fallback: Fish Audio**
- Generous free tier for development/testing
- Community voice models for variety

Pipeline: LLM text → stream to Cartesia/Azure WebSocket → receive PCM audio → resample to 48kHz Opus → feed to Discord voice playback.

### Twitch Integration
- Connects to Twitch EventSub WebSocket as a task in the root nursery
- Feeds channel events directly into the internal text queue:
  - Channel point redemptions, subscriptions, gift subs, cheers (bits), follows, ad breaks

### Monitoring Dashboard

**Starlette + Hypercorn** (trio-native web dashboard):
- Hypercorn natively supports trio via `hypercorn.trio.serve` — runs as a nursery task alongside the bot
- Starlette is ASGI-compatible, lightweight
- Routes:
  - `/health` — JSON status of each service (Discord, Twitch, transcription, TTS, LLM)
  - `/events` — Recent event log via SSE or WebSocket
  - Simple HTML page consuming these endpoints
- Controls can be added as POST endpoints
- Dependencies: `hypercorn`, `starlette`

---

## Security Guidelines

This project handles user-provided API keys and tokens (Discord bot token, Twitch OAuth, Deepgram, Cartesia, Azure, OpenAI). Treat all credentials as secrets.

### Credential Storage
- **Never hardcode tokens or API keys** in source code, config files checked into git, or log output
- Store secrets in environment variables or a `.env` file that is **gitignored**
- If persisting user-configured keys (e.g. from `/setup`), encrypt at rest using a machine-local key (e.g. via `cryptography.Fernet` with a key derived from a master secret in an env var)
- SQLite database files containing user data should not be committed to the repo

### Transport & Network
- All external API calls (Deepgram, Cartesia, Azure, Claude, Twitch) must use TLS (HTTPS / WSS) — never downgrade to plaintext
- The monitoring dashboard should bind to `127.0.0.1` by default, not `0.0.0.0`, to avoid exposing it to the network
- If the dashboard is exposed externally, require authentication (even a simple shared secret or token header)

### Logging & Error Handling
- **Never log secrets** — sanitize tokens, API keys, and auth headers from log output and error messages
- Avoid logging full request/response bodies from API calls that may contain keys
- Use structured logging so sensitive fields can be filtered consistently

### Input Validation
- Sanitize user input from Discord commands and Twitch events before passing to the LLM or storing in the database
- Treat all text from external sources (transcription output, Twitch chat, Discord messages) as untrusted
- Apply length limits on user-provided configuration values (personality prompts, familiar names) to prevent abuse

### Dependency Hygiene
- Pin dependency versions in `requirements.txt` to avoid supply-chain surprises
- Review new dependencies before adding them — prefer well-maintained packages with active security response
- Keep dependencies updated for security patches

### Principle of Least Privilege
- Discord bot should only request the permissions it actually needs (voice connect, send messages, use slash commands)
- Twitch token scopes should be minimal — only what EventSub subscriptions require
- API keys for third-party services should use the most restrictive tier/role available

---

## Getting Started

### Prerequisites
- Python 3.11+
- SQLite (bundled with Python)
- GPU recommended for local Whisper fallback
