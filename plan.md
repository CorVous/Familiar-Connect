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
                            Transcription
                                      ↓
                  trio.MemoryChannel (text) ← Twitch Events
                                      ↓
                          Message Processor + Chattiness
                                      ↓
                           Claude LLM + Conversation History
                                      ↓
                              Azure TTS → Audio
                                      ↓
                    trio.MemoryChannel → Discord Voice Playback
```

---

## Core Features

### Discord Bot
Built with **py-cord**. Voice send/receive uses **davey** to handle Discord's DAVE (Audio/Video E2E Encryption) protocol.

- **`/awaken`** — Joins your voice channel, captures audio in real-time from all speakers
- **`/sleep`** — Gracefully disconnects
- **`/setup`** — Configuration wizard (UI-driven via Discord modals) to set:
  - Familiar name, personality prompt, chattiness level (0–100)
  - Which transcription/LLM/TTS provider to use
  - API keys per provider, model selection, temperature

### Transcription
- Pick a reliable cheap provider. Maybe avoid relying on free services to avoid getting rate-limited or blocked?
- Converts Discord's 48kHz Opus audio → 16kHz WAV for transcription (or whatever the API needs)

### Message Processing
- Research the best way to figure out when the AI should speak in the conversation. Inputs will involve voice call data, but also text messages and twitch events and chat.
- Use SQLite or equivalent for vectorization, storage, etc. as needed.

### AI Response
- Research the best strategy for the conversational chatbot
- Leverage lessons learned by the SillyTavern community- terms like "connection profile", etc. might be ok to use, or maybe we could plug into a running silly-tavern instance somehow?
- Configurable model, temperature, max 200 tokens
- Full conversation history with timestamps
- System prompt from the familiar's configured personality
- Combine retrieval-augmented generation (RAG) with some budgeted trigger or vectorized activations to fill out the context.

### Text-to-Speech
- Research popular, reliable, cheap voice providers
- Also keep the 9 selectable Azure voices for nostalgia
- Resamples audio to 96kHz as neeed for Discord playback

### Twitch Integration
- Connects to Twitch EventSub WebSocket as a task in the root nursery
- Feeds channel events directly into the internal text queue:
  - Channel point redemptions, subscriptions, gift subs, cheers (bits), follows, ad breaks

---

## Getting Started

### Prerequisites
- Python 3.11+
- SQLite (bundled with Python)

### Run
```bash
pip install -r requirements.txt
python main.py
```
