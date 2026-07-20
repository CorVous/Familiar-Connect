# Contributing

This file is the working contributor reference — hot commands, the
post-change checklist, and code conventions. The pages under
[`docs/`](docs/index.md) are the authoritative deep dive:
[`docs/architecture/overview.md`](docs/architecture/overview.md) for the
system, [`docs/rust-port/DESIGN.md`](docs/rust-port/DESIGN.md) for the
implementation's module map, cross-cutting conventions, and decision log.

## Project specifics

- This is a **Cargo workspace** (edition 2024, stable toolchain pinned by
  `rust-toolchain.toml`). No Python — the prototype was retired after the
  Rust port reached parity (July 2026).
- Integration surfaces are **feature-gated**: `discord`, `discord-voice`,
  `stt-deepgram`, `local-turn`, `local-embed`, `twitch`, `azure-tts`,
  `audio-resample`. Defaults (`store`, `net`, `images`) cover everything
  unit tests need. Feature-gated code must keep default-feature gates green.

## After every change

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
```

Touched feature-gated code? Also gate the combo you touched, e.g.
`cargo clippy --features discord,discord-voice --all-targets -- -D warnings`.
The pre-commit hook (`git config core.hooksPath .githooks`) runs these gates.

Red / green TDD. One project nuance: **compile errors don't count as red** —
a test failing because the symbol doesn't exist is not a valid red; the
item must exist (stub is fine) before the test can fail for the right reason.

If the change touched **env vars / config keys**, **on-disk layout under
the familiars root**, or **architecture** (providers, processors, pipeline,
memory, history), update the matching page under `docs/` **in the same
commit**.

Behavioral contracts to respect (see `docs/rust-port/DESIGN.md` §4 for the
full list): exact error-message strings are test contracts; timestamps go
through `support::time::iso_utc` (lexicographic ordering is load-bearing);
Python-`round()` call sites use `support::round::half_even`; log lines are
wire formats parsed by `diagnose`; truncation counts Unicode scalars via
`support::text`.

## Conventions — technical writing

Comments and documentation:

- Be concise. Prefer **telegraphic** style.
    - Omit: articles ("the", "a"), auxiliary verbs, unnecessary prepositions, filler.
    - Keep: nouns, verbs, adjectives, key modifiers.
- Avoid restating the obvious — don't restate types already in signatures; don't
  summarize a function when its name says it.
- Document what's close and stable; avoid "far away" references likely to change
  (exception: ok if lints/tests/jobs catch the breakage).
- Capitalize the first word of a comment, unless it's an identifier that begins
  lowercase (`std::env`, `self.x`, backticked code); continuation lines of a
  multi-line sentence stay lowercase. Periods only for full sentences. Full
  sentences only when needed; lean on context.

**Scope:** telegraphic style applies strictly to doc comments and inline
comments. Wiki pages (`docs/*.md`) keep full sentences for readability but stay
concise — trim wordiness, filler, restating.

## Conventions — logging

Adding a log call. Match existing style — don't invent a new one.

- Use `tracing` with `target: "familiar_connect.<module>"` (targets mirror the
  historical logger names so `diagnose` and log filters keep working).
- Compose with `crate::log_style as ls`:
    - `ls::tag(label, color)` — leading `[label]`
    - `ls::kv(key, val, ...)` — `key=value` chunk
    - `ls::trunc(text, limit)` — ellipsis-truncate payloads
- Layout: one line, leading `ls::tag(...)` then space-separated `ls::kv(...)`
  pairs. The formatter repaints the leading tag for `WARN`/`ERROR` — keep the
  tag first. **Log lines are wire formats**: `diagnose` regex-parses
  `span=/ms=/status=` lines, and single-parameter ANSI SGR codes only.
- Emoji: reserve for notable transitions (✨ summon, 🎙️ stream).
