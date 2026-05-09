#!/bin/bash
# Dev environment bootstrap. Run once after clone or when the lock file drifts.
# Installs uv, syncs all dependency groups + local-turn extra, and verifies
# the toolchain (ruff, ty, pytest) can be invoked before any code work begins.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Install / upgrade uv
# ---------------------------------------------------------------------------
echo "==> Installing uv..."
pip install --quiet --upgrade uv

# uv self update pulls the latest binary; failures are non-fatal (rate limits,
# air-gapped machines) — pip already gave us a working version above.
echo "==> Updating uv binary (non-fatal if rate-limited)..."
uv self update || echo "    uv self update skipped (non-fatal)"

# ---------------------------------------------------------------------------
# 2. Regenerate the lock file then sync everything
#    --upgrade rebuilds lock from scratch, fixing stale or corrupt entries.
#    --extra local-turn is required for test collection (huggingface_hub,
#    numpy) — without it, import errors mask real test failures.
# ---------------------------------------------------------------------------
echo "==> Locking dependencies (regenerating uv.lock)..."
uv lock --upgrade

echo "==> Syncing dev deps + local-turn extra..."
uv sync --dev --extra local-turn

# ---------------------------------------------------------------------------
# 3. Smoke-test the toolchain — fail loudly here rather than mid-task
# ---------------------------------------------------------------------------
echo "==> Verifying toolchain..."

uv run ruff --version    || { echo "ERROR: ruff not available"; exit 1; }
uv run ty    --version   || { echo "ERROR: ty not available"; exit 1; }
uv run pytest --version  || { echo "ERROR: pytest not available"; exit 1; }
uv run familiar-connect --version || { echo "ERROR: familiar-connect CLI not importable"; exit 1; }

# ---------------------------------------------------------------------------
# 4. Run the full validation suite (mirrors AGENTS.md "After Every Code
#    Assignment" checklist)
# ---------------------------------------------------------------------------
echo "==> Running linter..."
uv run ruff check

echo "==> Running formatter check..."
uv run ruff format --check

echo "==> Running type checker..."
uv run ty check

echo "==> Running tests..."
uv run pytest

echo ""
echo "==> Dev environment ready."
