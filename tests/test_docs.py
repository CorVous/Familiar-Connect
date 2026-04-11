"""Automated checks that docs haven't drifted from code.

If a test here fails, documentation and implementation have diverged.
Fix the mismatch in the same PR — don't ignore the failure. See
``CLAUDE.md`` for the manual workflow these checks enforce.

The checks are deliberately one-directional: anything a doc names must
exist in source, but the reverse is not enforced (too many internal
tuning knobs to reasonably document). The one exception is the
canonical slash-commands reference page, which is checked bidirectionally.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
SRC_ROOT = REPO_ROOT / "src" / "familiar_connect"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-cd.yml"
SLASH_COMMANDS_DOC = DOCS_ROOT / "getting-started" / "slash-commands.md"
BOT_MODULE = SRC_ROOT / "bot.py"

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------
#
# If a documented artifact genuinely has no source-side counterpart,
# allowlist it here with a comment explaining why. Prefer fixing the
# drift over growing these lists.

#: Env vars mentioned in docs that aren't read directly by our code.
#: Typically third-party vars we document for operator convenience,
#: or planned-feature configuration where the read path doesn't exist yet.
#:
#: NOTE: ``TWITCH_CLIENT_ID`` and ``TWITCH_ACCESS_TOKEN`` are documented
#: in ``docs/guides/twitch.md`` for the planned ``/twitch connect``
#: slash command group, but the read path isn't wired up yet. Remove
#: these entries once ``familiar_connect.twitch`` starts reading them
#: from the environment.
ENV_VAR_ALLOWLIST: frozenset[str] = frozenset({
    "TWITCH_CLIENT_ID",
    "TWITCH_ACCESS_TOKEN",
})

#: Markdown files under ``docs/`` that intentionally live outside the
#: published ``mkdocs.yml`` nav (e.g. shared includes).
DOC_ORPHAN_ALLOWLIST: frozenset[str] = frozenset()

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
#
# ``mkdocs.yml`` uses ``!!python/name:`` tags from pymdownx. Plain
# ``yaml.safe_load`` chokes on these, so define a loader that ignores
# the tags without importing the referenced Python objects.


class _MkdocsLoader(yaml.SafeLoader):
    """A SafeLoader that tolerates mkdocs-material's ``!!python/name:`` tags."""


def _ignore_python_name(
    loader: yaml.Loader,  # noqa: ARG001
    suffix: str,  # noqa: ARG001
    node: yaml.Node,  # noqa: ARG001
) -> None:
    return None


_MkdocsLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    _ignore_python_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_nav(nav: Iterable[object]) -> Iterator[str]:
    """Walk the mkdocs ``nav:`` tree.

    Yields:
        Each relative markdown file path referenced by the tree.

    """
    for item in nav:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list):
                    yield from _flatten_nav(value)


def _read_mkdocs_nav() -> set[str]:
    cfg = yaml.load(
        MKDOCS_YML.read_text(encoding="utf-8"),
        Loader=_MkdocsLoader,  # noqa: S506
    )
    return set(_flatten_nav(cfg["nav"]))


def _docs_markdown_files() -> list[Path]:
    return sorted(DOCS_ROOT.rglob("*.md"))


def _is_os_environ(node: ast.AST) -> bool:
    """Return ``True`` if ``node`` is the expression ``os.environ``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_os_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "os"


def _extract_env_var_from_call(call: ast.Call) -> str | None:
    """Return the env var name if ``call`` reads one, else ``None``.

    Recognises ``os.environ.get(...)`` and ``os.getenv(...)`` where the
    first argument is a plain string literal.
    """
    if not call.args:
        return None
    first = call.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == "get" and _is_os_environ(func.value):
        return first.value
    if func.attr == "getenv" and _is_os_name(func.value):
        return first.value
    return None


def _extract_env_var_from_subscript(sub: ast.Subscript) -> str | None:
    """Return the env var name if ``sub`` is ``os.environ["..."]``."""
    if not _is_os_environ(sub.value):
        return None
    key = sub.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _collect_env_vars_read_in_src() -> set[str]:
    """Walk ``src/familiar_connect`` for env var names read at runtime."""
    names: set[str] = set()
    for py_file in SRC_ROOT.rglob("*.py"):
        if py_file.name == "_version.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _extract_env_var_from_call(node)
                if name is not None:
                    names.add(name)
            elif isinstance(node, ast.Subscript):
                name = _extract_env_var_from_subscript(node)
                if name is not None:
                    names.add(name)
    return names


# Inline backtick tokens in SHOUT_CASE with at least one underscore.
# The underscore requirement filters out bare constant-case words like
# IDLE / SPEAKING / TID251 that happen to look env-var-shaped.
_INLINE_ENV_VAR_RE = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")
_BASH_ASSIGNMENT_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)=")
_BASH_BLOCK_LANGUAGES = frozenset({"bash", "sh", "shell", "env", "dotenv"})
_ENV_CONTEXT_RE = re.compile(
    r"\benv\b|\.env\b|environment variable|os\.environ|getenv|dotenv",
    re.IGNORECASE,
)


def _extract_bash_env_vars(text: str) -> set[str]:
    """Extract NAME=... assignments from fenced bash/shell/env code blocks."""
    found: set[str] = set()
    in_env_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            if in_env_block:
                in_env_block = False
            elif lang in _BASH_BLOCK_LANGUAGES:
                in_env_block = True
            continue
        if in_env_block:
            match = _BASH_ASSIGNMENT_RE.match(line)
            if match is not None:
                found.add(match.group(1))
    return found


def _collect_env_vars_mentioned_in_docs() -> set[str]:
    """Scan docs for env var mentions with two complementary signals.

    Strong signal: ``NAME=value`` lines inside fenced bash/shell/env code
    blocks — always counted.

    Weaker signal: inline backticked SHOUT_CASE tokens containing an
    underscore — counted only in docs that also mention env-context
    cues (``.env``, "environment variable", "os.environ", etc.). This
    avoids matching Discord permission names, enum values, and lint-rule
    codes that happen to look env-var-shaped but live in docs that have
    nothing to do with configuration.
    """
    found: set[str] = set()
    for md_file in _docs_markdown_files():
        text = md_file.read_text(encoding="utf-8")
        found.update(_extract_bash_env_vars(text))
        if _ENV_CONTEXT_RE.search(text):
            found.update(match.group(1) for match in _INLINE_ENV_VAR_RE.finditer(text))
    return found


def _collect_registered_slash_commands() -> set[str]:
    """AST-parse ``bot.py`` for ``bot.slash_command(name=...)`` calls."""
    tree = ast.parse(BOT_MODULE.read_text(encoding="utf-8"))
    registered: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "slash_command"):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "name"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                registered.add(kw.value.value)
    return registered


_INLINE_SLASH_COMMAND_RE = re.compile(r"`/([a-z][a-z0-9-]*)`")


def _collect_documented_slash_commands(doc_path: Path) -> set[str]:
    """Scrape inline /command backtick tokens from a canonical reference doc."""
    return set(
        _INLINE_SLASH_COMMAND_RE.findall(doc_path.read_text(encoding="utf-8")),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mkdocs_strict_step_still_present() -> None:
    """CI must still run ``mkdocs build --strict`` (internal link check)."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["docs-build"]["steps"]
    assert any("mkdocs build --strict" in step.get("run", "") for step in steps), (
        "ci-cd.yml docs-build job must include `uv run mkdocs build --strict` — "
        "it's the canary for broken internal links across the site."
    )


def test_no_orphaned_docs() -> None:
    """Every markdown file under ``docs/`` must appear in ``mkdocs.yml`` nav."""
    nav_files = _read_mkdocs_nav()
    on_disk = {p.relative_to(DOCS_ROOT).as_posix() for p in _docs_markdown_files()}
    orphans = on_disk - nav_files - DOC_ORPHAN_ALLOWLIST
    assert not orphans, (
        f"Found markdown files under docs/ that aren't referenced by "
        f"mkdocs.yml nav: {sorted(orphans)}. Either add them to nav, "
        f"delete them, or (if they're intentional shared includes) add "
        f"them to DOC_ORPHAN_ALLOWLIST."
    )


def test_env_vars_documented_exist_in_src() -> None:
    """Every env var mentioned in docs must actually be read by src/.

    One-directional: undocumented internal tuning knobs are fine, but
    documenting an env var that no code reads is unambiguous drift.
    """
    documented = _collect_env_vars_mentioned_in_docs()
    src_read = _collect_env_vars_read_in_src()
    missing = documented - src_read - ENV_VAR_ALLOWLIST
    assert not missing, (
        f"Docs mention env vars that aren't read anywhere in "
        f"src/familiar_connect: {sorted(missing)}. Either remove them "
        f"from docs, implement the read path, or (if they're third-party "
        f"vars we deliberately document but don't consume ourselves) add "
        f"them to ENV_VAR_ALLOWLIST in tests/test_docs.py."
    )


def test_slash_commands_documented_match_code() -> None:
    """Slash commands in the canonical reference doc must match code exactly.

    Scoped to ``docs/getting-started/slash-commands.md``. This is the one
    place we enforce bidirectional parity — commands documented elsewhere
    (e.g. the Twitch guide's command groups) live under different
    registration patterns and are out of scope.
    """
    documented = _collect_documented_slash_commands(SLASH_COMMANDS_DOC)
    registered = _collect_registered_slash_commands()
    unimplemented = documented - registered
    undocumented = registered - documented
    assert not unimplemented, (
        f"slash-commands.md mentions commands that aren't registered in "
        f"bot.py: {sorted(unimplemented)}. Either wire them up or remove "
        f"them from the doc."
    )
    assert not undocumented, (
        f"bot.py registers slash commands that aren't in "
        f"docs/getting-started/slash-commands.md: {sorted(undocumented)}. "
        f"Add a row to the command table."
    )
