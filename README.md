# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

---

## Goals

- **Single runtime**: The entire backend runs as one Python process using `trio`. No separate worker scripts, no external message broker.
- **No RabbitMQ**: Internal async channels (`trio.MemoryChannel`) replace RabbitMQ for all inter-component communication.
- **Unified entry point**: One `python main.py` starts everything — Discord gateway, voice capture, transcription, LLM, TTS, and Twitch listener all run as concurrent tasks in a `trio.Nursery`.

---

## Target Architecture

All components run as coroutines within a single `asyncio` event loop:

```
Discord Voice → audio capture → trio.MemoryChannel
                                      ↓
                            Transcription (Deepgram/Whisper)
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
- **`/awaken`** — Joins your voice channel, captures audio in real-time from all speakers
- **`/sleep`** — Gracefully disconnects
- **`/setup`** — Configuration wizard (UI-driven via Discord modals) to set:
  - Familiar name, personality prompt, chattiness level (0–100)
  - Which transcription/LLM/TTS provider to use
  - API keys per provider, model selection, temperature

### Transcription
- **API-based**: Deepgram ("nova-2") or OpenAI Whisper API
- **Local**: Whisper running via `trio.to_thread.run_sync` to avoid blocking the event loop, with GPU (CUDA) or CPU fallback
- Converts Discord's 48kHz Opus audio → 16kHz WAV for transcription

### Message Processing
- **Chattiness meter**: 0–100 engagement system — keywords trigger full attention, normal messages only get processed based on the meter level
- **Speaking detection**: Delays processing while users are actively talking (2s buffer)
- Pulls conversation history from SQLite and sends it to the LLM for context

### AI Response (Anthropic Claude)
- Configurable model (Opus/Sonnet/Haiku), temperature, max 200 tokens
- Full conversation history context with timestamps
- System prompt from the familiar's configured personality

### Text-to-Speech (Azure)
- 9 selectable Azure neural voices
- Resamples audio from 16kHz → 96kHz for Discord playback

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

---

## Service Integrations

| Category       | Provider(s)                             |
|----------------|-----------------------------------------|
| Transcription  | Deepgram, OpenAI Whisper (API or local) |
| Language Model | Anthropic Claude (Opus, Sonnet, Haiku)  |
| Text-to-Speech | Azure Cognitive Services (9 voices)     |
| Streaming      | Twitch EventSub WebSocket               |
| Database       | SQLite                                  |
