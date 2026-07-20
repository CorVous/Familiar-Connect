# Contributing

Dev workflow and expectations.

## Environment setup

This is a Cargo workspace (edition 2024). Install a Rust toolchain via
[rustup](https://rustup.rs/); the pinned stable version in
`rust-toolchain.toml` is fetched automatically on first build. No Python —
the prototype was retired after the Rust port reached parity (July 2026).

Integration surfaces are feature-gated (`discord`, `discord-voice`,
`stt-deepgram`, `local-turn`, `local-embed`, `twitch`, `azure-tts`,
`audio-resample`); the defaults (`store`, `net`, `images`) cover everything
unit tests need. Build a combo with `--features`:

```bash
cargo build --features discord,discord-voice,stt-deepgram
```

See [Installation](getting-started/installation.md) for runtime prerequisites (Discord token, OpenRouter key, Cartesia key, CMake for voice builds, etc.).

## TDD workflow

Red / green TDD:

1. Failing test first (red).
2. Minimum code to pass (green).
3. Refactor if needed.

**Compile errors don't count as red.** A test failing because the symbol doesn't exist is not a valid red — the item must exist (a stub is fine) before the test can fail for the right reason.

## After every code change

Run the same four checks CI runs:

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings   # lint (pedantic; warnings are errors)
cargo fmt
```

Touched feature-gated code? Also gate the combo you touched, e.g.
`cargo clippy --features discord,discord-voice --all-targets -- -D warnings`.

Cheap on a clean tree. Local failures fail CI the same way — fix root cause before pushing. The pre-commit hook at `.githooks/pre-commit` runs the same gates (`git config core.hooksPath .githooks` to enable).

## Docs build & preview

Built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/), a standalone Python tool independent of the Rust workspace. Install it with [`uvx`](https://docs.astral.sh/uv/) or `pipx`, then preview locally:

```bash
mkdocs serve
```

Strict build (fails on broken internal links — what CI runs):

```bash
mkdocs build --strict
```

## Commit style

- Conventional prefix (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) + imperative description.
- Body explains *why*, not *what* — diff shows the what.
- Keep commits focused. Bug fix shouldn't drag a refactor; refactor shouldn't drag a bug fix.

## Scope discipline

- No features, refactors, or "improvements" beyond the task. Bug fixes don't need surrounding cleanup. Simple features don't need extra configurability.
- No error handling, fallbacks, or validation for scenarios that can't happen. Trust internal guarantees. Validate only at system boundaries (user input, external APIs).
- No helpers / abstractions for one-time ops. Three similar lines beats a premature abstraction.
- No design for hypothetical future requirements. [Design decisions](architecture/decisions.md) holds rejected ideas — check there first.

## Where things live

- [Architecture overview](architecture/overview.md) — big picture, component map.
- [Configuration model](architecture/configuration-model.md) — two-level config split, on-disk layout.
- [Security](architecture/security.md) — credential storage, logging rules.
- [Design decisions](architecture/decisions.md) — rejected ideas.
