"""Seed authored memories (turns + stance facts) into a familiar's store.

Authored ``seed_turns.toml`` carries first-person "journal" turns plus
hand-written facts citing them. Direct insertion — no LLM extraction:
``FactExtractor`` is tuned to skip feelings/stances, exactly the content
seeding exists for. Provenance preserved via ``source_turn_ids``.

Idempotent: each entry keyed by ``platform_message_id = "seed:<id>"``;
existing entries skipped on re-run.

Watermark: advanced past seed turns only when store had no unprocessed
backlog before seeding — otherwise left untouched (extractor re-sweep of
seed turns is harmless; authored facts already present).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from familiar_connect import log_style as ls
from familiar_connect.config import ConfigError, load_character_config
from familiar_connect.embedding import create_embedder
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.processors.fact_embedding_worker import FactEmbeddingWorker

if TYPE_CHECKING:
    import argparse

    from familiar_connect.embedding.protocol import Embedder

_logger = logging.getLogger(__name__)

SEED_ID_PREFIX = "seed:"


@dataclass(frozen=True)
class SeedFact:
    """Authored fact attached to one seed turn."""

    text: str
    importance: int | None = None


@dataclass(frozen=True)
class SeedEntry:
    """One journal turn + its authored facts."""

    id: str
    turn: str
    facts: tuple[SeedFact, ...]


@dataclass(frozen=True)
class SeedFile:
    """Parsed ``seed_turns.toml``."""

    channel_id: int
    entries: tuple[SeedEntry, ...]


@dataclass
class SeedReport:
    """Outcome counts for one seeding run."""

    inserted_turns: int = 0
    skipped_turns: int = 0
    inserted_facts: int = 0
    embedded_facts: int = 0
    watermark_advanced: bool = False


def _parse_fact(raw: Any, entry_id: str) -> SeedFact:  # noqa: ANN401
    if not isinstance(raw, dict):
        msg = f"entry {entry_id!r}: each fact must be a table"
        raise TypeError(msg)
    text = str(raw.get("text", "")).strip()
    if not text:
        msg = f"entry {entry_id!r}: fact missing non-empty 'text'"
        raise ValueError(msg)
    importance = raw.get("importance")
    if importance is not None and not isinstance(importance, int):
        msg = f"entry {entry_id!r}: 'importance' must be an integer"
        raise ValueError(msg)
    return SeedFact(text=text, importance=importance)


def load_seed_file(path: Path) -> SeedFile:
    """Parse and validate seed TOML; raise ValueError on bad shape."""
    with path.open("rb") as f:
        raw = tomllib.load(f)

    channel_id = raw.get("channel_id")
    if channel_id is None:
        msg = f"{path}: top-level 'channel_id' required"
        raise ValueError(msg)
    if not isinstance(channel_id, int):
        msg = f"{path}: 'channel_id' must be an integer"
        raise TypeError(msg)

    entries: list[SeedEntry] = []
    seen_ids: set[str] = set()
    for item in raw.get("entries", []):
        entry_id = str(item.get("id", "")).strip()
        if not entry_id:
            msg = f"{path}: entry missing non-empty 'id'"
            raise ValueError(msg)
        if entry_id in seen_ids:
            msg = f"{path}: duplicate entry id {entry_id!r}"
            raise ValueError(msg)
        seen_ids.add(entry_id)
        turn = str(item.get("turn", "")).strip()
        if not turn:
            msg = f"{path}: entry {entry_id!r} missing non-empty 'turn'"
            raise ValueError(msg)
        facts = tuple(_parse_fact(f, entry_id) for f in item.get("facts", []))
        if not facts:
            msg = f"{path}: entry {entry_id!r} must carry at least one entry in 'facts'"
            raise ValueError(msg)
        entries.append(SeedEntry(id=entry_id, turn=turn, facts=facts))

    if not entries:
        msg = f"{path}: no [[entries]] found"
        raise ValueError(msg)
    return SeedFile(channel_id=channel_id, entries=tuple(entries))


async def seed_memory(
    *,
    store: AsyncHistoryStore,
    familiar_id: str,
    seed: SeedFile,
    embedder: Embedder | None = None,
) -> SeedReport:
    """Insert seed turns + facts; embed; manage watermark. Idempotent."""
    report = SeedReport()

    backlog = await store.turns_since_watermark(familiar_id=familiar_id, limit=1)
    had_backlog = bool(backlog)

    last_turn_id: int | None = None
    seen_fact_ids: set[int] = set()
    for entry in seed.entries:
        pmid = SEED_ID_PREFIX + entry.id
        existing = await store.lookup_turn_by_platform_message_id(
            familiar_id=familiar_id, platform_message_id=pmid
        )
        if existing is not None:
            report.skipped_turns += 1
            continue
        turn = await store.append_turn(
            familiar_id=familiar_id,
            channel_id=seed.channel_id,
            role="assistant",
            content=entry.turn,
            author=None,
            platform_message_id=pmid,
        )
        report.inserted_turns += 1
        last_turn_id = turn.id
        for fact in entry.facts:
            # channel_id=None — stances are global, not channel-bound
            inserted = await store.append_fact(
                familiar_id=familiar_id,
                channel_id=None,
                text=fact.text,
                source_turn_ids=[turn.id],
                importance=fact.importance,
            )
            # append_fact returns the existing Fact on dedup-skip; count
            # only ids not seen before this run.
            if inserted.id not in seen_fact_ids:
                seen_fact_ids.add(inserted.id)
                report.inserted_facts += 1

    if not had_backlog and last_turn_id is not None:
        await store.put_writer_watermark(
            familiar_id=familiar_id, last_written_id=last_turn_id
        )
        report.watermark_advanced = True
    elif had_backlog and report.inserted_turns:
        _logger.warning(
            f"{ls.tag('Seed', ls.Y)} "
            f"{ls.kv('watermark', 'untouched — extractor backlog', vc=ls.LY)} "
            f"{ls.kv('note', 'extractor will re-sweep seed turns', vc=ls.LW)}"
        )

    if embedder is not None:
        worker = FactEmbeddingWorker(
            store=store, embedder=embedder, familiar_id=familiar_id
        )
        while n := await worker.tick():
            report.embedded_facts += n

    _logger.info(
        f"{ls.tag('Seed', ls.LC)} "
        f"{ls.kv('inserted_turns', str(report.inserted_turns), vc=ls.LC)} "
        f"{ls.kv('skipped_turns', str(report.skipped_turns), vc=ls.LY)} "
        f"{ls.kv('inserted_facts', str(report.inserted_facts), vc=ls.LC)} "
        f"{ls.kv('embedded', str(report.embedded_facts), vc=ls.LW)} "
        f"{ls.kv('watermark_advanced', str(report.watermark_advanced), vc=ls.LW)}"
    )
    return report


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the seed-memory subcommand."""
    parser = subparsers.add_parser(
        "seed-memory",
        parents=[common_parser],
        help="Seed authored memories into a familiar's store",
        description=(
            "Insert authored journal turns + stance facts from "
            "seed_turns.toml into history.db; embed facts. Idempotent — "
            "re-runs skip entries already present."
        ),
    )
    parser.add_argument(
        "--familiar",
        metavar="ID",
        default=None,
        help=(
            "Folder name of the character to seed "
            "(under data/familiars/). Overrides FAMILIAR_ID."
        ),
    )
    parser.add_argument(
        "--seed-file",
        metavar="PATH",
        default=None,
        help="seed TOML path; default <familiar>/seed_turns.toml",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        default=False,
        help="skip fact embedding (RAG embedding signal stays cold until bot runs)",
    )
    parser.set_defaults(func=seed)
    return parser


def seed(args: argparse.Namespace) -> int:
    """Entry point for ``familiar-connect seed-memory``."""
    # lazy — run.py pulls heavy deps (discord, voice stack)
    from familiar_connect.commands.run import _resolve_familiar_root  # noqa: PLC0415

    try:
        familiar_root = _resolve_familiar_root(args)
    except ValueError as exc:
        _logger.error("%s", exc)
        return 1
    familiar_id = familiar_root.name

    seed_path = (
        Path(args.seed_file) if args.seed_file else familiar_root / "seed_turns.toml"
    )
    if not seed_path.exists():
        _logger.error("Seed file not found: %s", seed_path)
        return 1
    try:
        seed_file = load_seed_file(seed_path)
    except ValueError as exc:
        _logger.error("%s", exc)
        return 1

    embedder: Embedder | None = None
    if not args.no_embed:
        defaults_path = familiar_root.parent / "_default" / "character.toml"
        try:
            character_config = load_character_config(
                familiar_root / "character.toml",
                defaults_path=defaults_path,
            )
            embedder = create_embedder(character_config.embedding)
        except (ConfigError, ValueError, RuntimeError) as exc:
            _logger.error("Embedder unavailable: %s", exc)
            return 1

    store = HistoryStore(familiar_root / "history.db")
    try:
        report = asyncio.run(
            seed_memory(
                store=AsyncHistoryStore(store),
                familiar_id=familiar_id,
                seed=seed_file,
                embedder=embedder,
            )
        )
    finally:
        store.close()

    sys.stdout.write(
        f"Seeded {report.inserted_turns} turns / {report.inserted_facts} facts "
        f"({report.skipped_turns} entries already present, "
        f"{report.embedded_facts} facts embedded).\n"
    )
    return 0
