"""``familiar-connect init`` — bootstrap a familiar from narrative markdown.

Reads a folder of ``*.md`` files describing a character, drives the
``background`` LLM slot through generating three authored-canon
artifacts (``character.md``, ``character.toml``, ``lorebook.toml``),
and writes them under ``data/familiars/<id>/``.

By design touches only the authored-canon side of the memory trust
boundary documented in
``docs/architecture/memory-strategies.md``: no ``history.db``, no
``facts``, no ``people_dossiers``. Experiential memory accrues from
real conversations.

Usage::

    familiar-connect init <id> --from path/to/narrative-dir
    familiar-connect init <id> --from path/to/narrative-dir --dry-run
    familiar-connect init <id> --from path/to/narrative-dir --force
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w

from familiar_connect import log_style as ls
from familiar_connect.config import load_character_config
from familiar_connect.llm import LLMClient, Message

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

_logger = logging.getLogger(__name__)

_DEFAULT_FAMILIARS_ROOT = Path("data") / "familiars"
_DEFAULTS_DIR_NAME = "_default"

# files we'll write; bare filenames so dry-run reporting reads cleanly
_OUT_CHARACTER_MD = "character.md"
_OUT_CHARACTER_TOML = "character.toml"
_OUT_LOREBOOK_TOML = "lorebook.toml"


class InitError(Exception):
    """Raised on operator-recoverable init failures.

    Distinct from upstream LLM / IO errors so the CLI can map this
    class to exit-1 with a one-line message; everything else propagates
    with a stack trace.
    """


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the ``init`` subcommand."""
    parser = subparsers.add_parser(
        "init",
        parents=[common_parser],
        help="Bootstrap a familiar folder from narrative markdown",
        description=(
            "Generate character.md / character.toml / lorebook.toml for "
            "a new familiar from a folder of narrative markdown files. "
            "Writes only authored canon — no facts, no dossiers, no "
            "history.db. Reads OPENROUTER_API_KEY for the background "
            "LLM slot configured in data/familiars/_default/character.toml."
        ),
    )
    parser.add_argument(
        "familiar_id",
        metavar="ID",
        help="Folder name to create under data/familiars/.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory of narrative *.md files describing the character.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing familiar folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes; do not touch disk.",
    )
    parser.set_defaults(func=init)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def init(args: argparse.Namespace) -> int:
    """Subcommand entry. Returns 0 on success, 1 on operator error."""
    try:
        return _run(args)
    except InitError as exc:
        _logger.error(f"{ls.tag('Init', ls.R)} {ls.kv('error', str(exc), vc=ls.R)}")
        return 1


def _run(args: argparse.Namespace) -> int:
    target = _DEFAULT_FAMILIARS_ROOT / args.familiar_id
    source: Path = args.source

    if args.familiar_id == _DEFAULTS_DIR_NAME:
        msg = (
            f"refusing to write to reserved folder '{_DEFAULTS_DIR_NAME}' — "
            "pick a different familiar id."
        )
        raise InitError(msg)

    if not source.exists() or not source.is_dir():
        msg = f"narrative source dir not found: {source}"
        raise InitError(msg)

    narrative = _read_narrative(source)
    if not narrative.strip():
        msg = f"no markdown content found under {source}"
        raise InitError(msg)

    if target.exists() and not args.force:
        msg = f"familiar folder already exists: {target}. Pass --force to overwrite."
        raise InitError(msg)

    llm = _load_background_llm(_default_character_toml_path())
    plan = asyncio.run(_drive_llm(llm, narrative))

    if args.dry_run:
        _print_plan(target, plan)
        return 0

    _write_plan(target, plan)
    _logger.info(
        f"{ls.tag('Init', ls.LG)} "
        f"{ls.kv('familiar_id', args.familiar_id, vc=ls.LY)} "
        f"{ls.kv('wrote', ','.join(sorted(plan)), vc=ls.LW)}"
    )
    return 0


# ---------------------------------------------------------------------------
# Narrative loading
# ---------------------------------------------------------------------------


def _read_narrative(source: Path) -> str:
    """Concatenate every ``*.md`` file under *source*, header per file.

    Sort by relative path so the LLM sees a stable input across runs.
    """
    chunks: list[str] = []
    for md in sorted(source.rglob("*.md")):
        rel = md.relative_to(source)
        body = md.read_text(encoding="utf-8").rstrip()
        chunks.append(f"## {rel.as_posix()}\n\n{body}")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# LLM plan
# ---------------------------------------------------------------------------


async def _drive_llm(llm: LLMClient, narrative: str) -> dict[str, str]:
    """Async wrapper: build the plan and close the client in one loop."""
    try:
        return await _build_plan(llm, narrative)
    finally:
        await llm.close()


async def _build_plan(llm: LLMClient, narrative: str) -> dict[str, str]:
    """Drive the LLM through three artifact generations.

    Sequential — each later call sees prior outputs, so the persona
    voice in ``character.md`` colours the lorebook tone, and config
    aliases align with the character description.
    """
    character_md = await _gen_character_md(llm, narrative)
    character_toml = await _gen_character_toml(llm, narrative, character_md)
    lorebook_toml = await _gen_lorebook_toml(llm, narrative, character_md)
    return {
        _OUT_CHARACTER_MD: character_md,
        _OUT_CHARACTER_TOML: character_toml,
        _OUT_LOREBOOK_TOML: lorebook_toml,
    }


async def _gen_character_md(llm: LLMClient, narrative: str) -> str:
    """Produce the persona system-prompt sidecar."""
    prompt = [
        Message(
            role="system",
            content=(
                "You read narrative-style markdown about one fictional "
                "character and produce a persona description suitable "
                "as a system-prompt sidecar for that character to "
                "roleplay as themselves. Reply with markdown only — no "
                "preamble, no JSON, no code fences. 200-600 words. "
                "Cover: who they are, voice and manner, recurring "
                "concerns, how they treat others. Do not invent facts "
                "the source doesn't support."
            ),
        ),
        Message(role="user", content=narrative),
    ]
    reply = await llm.chat(prompt)
    text = reply.content.strip()
    if not text:
        msg = "character.md generation returned an empty reply"
        raise InitError(msg)
    return text + "\n"


async def _gen_character_toml(llm: LLMClient, narrative: str, character_md: str) -> str:
    """Produce a minimal ``character.toml`` overlay.

    Only emits keys the narrative actually motivates; deep-merge over
    ``_default/character.toml`` fills the rest.
    """
    prompt = [
        Message(
            role="system",
            content=(
                "Given a character description and source narrative, "
                "propose a minimal character.toml overlay for "
                "Familiar-Connect. Reply with a JSON object only "
                "(no prose, no code fences). Supported keys:\n"
                "  display_tz: IANA timezone string (default 'UTC')\n"
                "  aliases: list of additional names the character "
                "answers to (the bot's primary id is set elsewhere)\n"
                "Omit keys you can't justify from the source. Empty "
                "JSON object {} is acceptable."
            ),
        ),
        Message(
            role="user",
            content=f"Description:\n{character_md}\n\nNarrative:\n{narrative}",
        ),
    ]
    reply = await llm.chat(prompt)
    data = _parse_json_object(reply.content, artifact="character.toml")
    overlay = _coerce_character_overlay(data)
    return tomli_w.dumps(overlay)


async def _gen_lorebook_toml(llm: LLMClient, narrative: str, character_md: str) -> str:
    """Produce ``lorebook.toml`` — keyword-activated authored canon."""
    prompt = [
        Message(
            role="system",
            content=(
                "Build a Familiar-Connect lorebook from the narrative. "
                "Lorebook entries are keyword-activated: when any key "
                "appears in recent chat, the entry's content is "
                "injected into the prompt. Good entries: places, "
                "factions, recurring objects, world rules, named NPCs "
                "other than the focal character. Reply with a JSON "
                "object only:\n"
                '{"entries": [{"keys": ["..."], "content": "1-3 '
                'sentences", "priority": 0-100, "selective": false}]}\n'
                "5-20 entries. Omit entries you can't justify. "
                "'selective': true means ALL keys must match (AND); "
                "default false means ANY key matches (OR)."
            ),
        ),
        Message(
            role="user",
            content=f"Description:\n{character_md}\n\nNarrative:\n{narrative}",
        ),
    ]
    reply = await llm.chat(prompt)
    data = _parse_json_object(reply.content, artifact="lorebook.toml")
    entries = _coerce_lorebook_entries(data)
    if not entries:
        return ""
    return tomli_w.dumps({"entries": entries})


# ---------------------------------------------------------------------------
# JSON / TOML coercion
# ---------------------------------------------------------------------------


def _parse_json_object(content: str, *, artifact: str) -> dict[str, Any]:
    """Tolerant JSON extraction from an LLM reply.

    Strips ```json fences if present, then parses. Raises
    :class:`InitError` with the artifact name on failure so operators
    can tell which call went sideways.
    """
    text = content.strip()
    if text.startswith("```"):
        # drop opening fence (with optional language) + closing fence
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"{artifact}: LLM reply was not valid JSON ({exc.msg})"
        raise InitError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{artifact}: expected JSON object, got {type(data).__name__}"
        raise InitError(msg)
    return data


def _coerce_character_overlay(data: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only keys we understand; drop the rest silently.

    Lets the LLM be loose without polluting the written TOML with
    fields the config loader would reject.
    """
    out: dict[str, Any] = {}
    tz = data.get("display_tz")
    if isinstance(tz, str) and tz.strip():
        out["display_tz"] = tz.strip()
    aliases = data.get("aliases")
    if isinstance(aliases, list):
        cleaned = [str(a).strip() for a in aliases if str(a).strip()]
        if cleaned:
            out["aliases"] = cleaned
    return out


def _coerce_lorebook_entries(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate + normalise the LLM's lorebook proposal.

    Drops entries missing required fields; keeps the rest with
    defaults applied. Empty list when nothing usable came back.
    """
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        keys = raw.get("keys")
        content = raw.get("content")
        if not isinstance(keys, list) or not keys:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        keys_clean = [str(k).strip() for k in keys if str(k).strip()]
        if not keys_clean:
            continue
        entry: dict[str, Any] = {
            "keys": keys_clean,
            "content": content.strip(),
        }
        priority = raw.get("priority")
        if isinstance(priority, int):
            entry["priority"] = max(0, min(100, priority))
        selective = raw.get("selective")
        if isinstance(selective, bool):
            entry["selective"] = selective
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


def _write_plan(target: Path, plan: Mapping[str, str]) -> None:
    """Atomic per-file write into *target*.

    Tempfile + rename so a mid-write crash never leaves a partial
    file. Empty content skips that artifact — keeps the folder tidy
    when the LLM produced nothing meaningful.
    """
    target.mkdir(parents=True, exist_ok=True)
    for name, content in plan.items():
        if not content:
            continue
        _atomic_write(target / name, content)


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _print_plan(target: Path, plan: Mapping[str, str]) -> None:
    """Stdout summary for ``--dry-run``."""
    lines = [f"would write under {target}:"]
    for name in sorted(plan):
        body = plan[name]
        if not body:
            lines.append(f"  - {name}: (empty — skipped)")
            continue
        lines.append(f"  - {name}: {len(body)} chars")
    print("\n".join(lines))  # noqa: T201 — CLI surface


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------


def _default_character_toml_path() -> Path:
    return _DEFAULT_FAMILIARS_ROOT / _DEFAULTS_DIR_NAME / "character.toml"


def _load_background_llm(defaults_path: Path) -> LLMClient:
    """Build a single ``background``-slot LLM from the default profile.

    Skips the full ``create_llm_clients`` factory: we only need one
    slot, and init runs before a familiar folder exists so there's no
    per-familiar overlay to merge.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        msg = (
            "OPENROUTER_API_KEY not set — needed for the background "
            "LLM slot that powers init."
        )
        raise InitError(msg)
    config = load_character_config(defaults_path, defaults_path=defaults_path)
    slot = config.llm["background"]
    return LLMClient(
        api_key=api_key,
        model=slot.model,
        temperature=slot.temperature,
        slot="background",
        provider_order=slot.provider_order,
        provider_allow_fallbacks=slot.provider_allow_fallbacks,
        reasoning=slot.reasoning,
    )


__all__ = ["InitError", "add_parser", "init"]
