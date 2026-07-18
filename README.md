# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens, understands speech, and talks back with real AI voices. Written in Rust.

**Full documentation lives at [`docs/`](./docs/index.md).** The architecture pages describe the system as designed; the implementation is the Rust workspace in this repo (ported from the original Python prototype in July 2026 — see [`docs/rust-port/`](./docs/rust-port/DESIGN.md) for the port design, per-subsystem specs, and the review log).

## Quickstart

```bash
cp .env.example .env                    # fill in your tokens
cargo run --release --features discord -- run --familiar <id> -v
```

Feature flags select the integration surface (defaults cover storage, HTTP, and images):

```bash
# Text-only Discord bot
cargo build --release --features discord

# Voice (DAVE E2EE via songbird) + Deepgram streaming STT
cargo build --release --features discord,discord-voice,stt-deepgram

# Local ML extras (ONNX turn detection, local embeddings)
cargo build --release --features local-turn,local-embed
```

See [`docs/rust-port/DAVE-RUNBOOK.md`](./docs/rust-port/DAVE-RUNBOOK.md) for prerequisites (Discord bot token + intents, OpenRouter key, optional TTS/STT keys), the on-disk `<familiars-root>/<id>/` shape, platform notes (Windows voice builds need CMake), and a staged smoke ladder for voice.

## Where things are

- **[Architecture](./docs/architecture/overview.md)** — the bot shell, event bus, memory strategies, voice pipeline.
- **[Port history](./docs/rust-port/DESIGN.md)** — module map, decision log, conventions; [`specs/`](./docs/rust-port/specs/) holds the per-subsystem behavioral contracts and the [review log](./docs/rust-port/specs/review-log.md).
- **[Contributing](./CONTRIBUTING.md)** — dev workflow and conventions.

## Development commands

```bash
cargo build                                    # compile (default features)
cargo test                                     # run the test suite (~2,000 tests)
cargo clippy --all-targets -- -D warnings      # lint (pedantic; warnings are errors)
cargo fmt                                      # format
cargo run -- diagnose <logfile>                # span/latency report from a captured log
```

The pre-commit hook at [`.githooks/pre-commit`](./.githooks/pre-commit) runs the same gates (`git config core.hooksPath .githooks` to enable).
