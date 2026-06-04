#!/usr/bin/env bash
# Run the CI `lint-and-test` job locally, byte-for-byte.
#
# Two things make a bare `uv run ruff check` diverge from CI on some dev
# machines, and this script neutralizes both:
#   1. uv installs to ~/.local/bin, which isn't always on PATH.
#   2. Network mounts / umask can leave working-tree files executable.
#      git tracks them 100644, so CI sees no exec bit — but ruff stats the
#      real file and fires EXE002. We re-sync disk exec bits to git's modes
#      before linting.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"

# normalize working-tree exec bits to match git's tracked modes
git ls-files --stage | while read -r mode _ _ path; do
  if [ "$mode" = "100755" ]; then chmod +x "$path"; else chmod a-x "$path"; fi
done

uv sync --dev --extra local-turn   # local-turn pulls numpy + hf_hub, imported at test collection
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
