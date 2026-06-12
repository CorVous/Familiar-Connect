"""``familiar-connect sleep`` — nightly memory-hygiene consolidation pass.

Dry-run by DEFAULT: plans fact retirements/rewrites over the day's
window and writes an audit artifact, touching no rows. ``--apply``
executes the plan against the live store. Build memory-hygiene trust
on dry-run audits before ever wiring this to a clock.

See ``docs/architecture/sleep.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect.sleep.dream import DEFAULT_OPINION_CAP
from familiar_connect.sleep.hygiene import (
    DEFAULT_FACTS_MAX,
    DEFAULT_RETIRE_CAP,
    DEFAULT_TURNS_MAX,
)

# orchestration lives in sleep.passes (shared with the activity
# engine's lifecycle-coupled passes); re-exported here for callers
# of the CLI module
from familiar_connect.sleep.passes import (
    AUDIT_DIRNAME,
    execute_dream,
    execute_sleep,
    hygiene_denylist_ids,
)

if TYPE_CHECKING:
    import argparse

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the sleep subcommand."""
    parser = subparsers.add_parser(
        "sleep",
        parents=[common_parser],
        help="Run the memory-hygiene consolidation pass (dry-run by default)",
        description=(
            "Consolidate a familiar's facts: retire noise, merge "
            "near-duplicates, re-attribute misfiled claims. Dry-run by "
            "default — proposals are written to a sleep_audits/ artifact "
            "and NOTHING is changed. Pass --apply to write the plan to the "
            "live store. Pinned (seed/authored) facts are never touched; a "
            "per-run retirement cap bounds blast radius."
        ),
    )
    parser.add_argument(
        "--familiar",
        metavar="ID",
        default=None,
        help="Folder name under data/familiars/. Overrides FAMILIAR_ID.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="execute the plan against the live store (default: dry-run only)",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "hygiene", "dream"),
        default="all",
        help="which sleep stage(s) to run (default all: hygiene then dream)",
    )
    parser.add_argument(
        "--opinion-cap",
        type=int,
        default=DEFAULT_OPINION_CAP,
        metavar="N",
        help=f"max opinions minted per dream run (default {DEFAULT_OPINION_CAP})",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_RETIRE_CAP,
        metavar="N",
        help=f"max facts mutated per run (default {DEFAULT_RETIRE_CAP})",
    )
    parser.add_argument(
        "--facts-max",
        type=int,
        default=DEFAULT_FACTS_MAX,
        metavar="N",
        help=f"max current facts considered (default {DEFAULT_FACTS_MAX})",
    )
    parser.add_argument(
        "--turns-max",
        type=int,
        default=DEFAULT_TURNS_MAX,
        metavar="N",
        help=f"max recent turns shown for attribution (default {DEFAULT_TURNS_MAX})",
    )
    parser.add_argument(
        "--audit-dir",
        metavar="PATH",
        default=None,
        help="audit artifact directory; default <familiar>/sleep_audits/",
    )
    parser.set_defaults(func=run)
    return parser


async def _run_sleep(
    *,
    store: AsyncHistoryStore,
    llm: LLMClient,
    familiar_id: str,
    display_name: str,
    display_tz: str,
    audit_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Run requested stage(s): hygiene then dream. Return summary lines.

    Within one ``all`` run, hygiene's retired-fact texts feed the dream
    stage as a known-bits deny-list.
    """
    mode = "APPLIED" if args.apply else "DRY-RUN"
    lines: list[str] = []
    denylist: tuple[str, ...] = ()

    if args.stage in {"all", "hygiene"}:
        plan, path = await execute_sleep(
            store=store,
            llm=llm,
            familiar_id=familiar_id,
            familiar_display_name=display_name,
            audit_dir=audit_dir,
            apply=args.apply,
            facts_max=args.facts_max,
            turns_max=args.turns_max,
            cap=args.cap,
        )
        lines.append(
            f"[{mode}] hygiene: {len(plan.retire)} retire, "
            f"{len(plan.rewrite)} rewrite, {len(plan.rejected)} rejected "
            f"({plan.mutated_count} facts). Audit: {path}"
        )
        deny_ids = hygiene_denylist_ids(plan)
        if deny_ids:
            facts = await store.facts_by_ids(familiar_id=familiar_id, ids=deny_ids)
            denylist = tuple(f.text for f in facts)

    if args.stage in {"all", "dream"}:
        plan_d, path_d = await execute_dream(
            store=store,
            llm=llm,
            familiar_id=familiar_id,
            familiar_display_name=display_name,
            display_tz=display_tz,
            audit_dir=audit_dir,
            apply=args.apply,
            denylist=denylist,
            cap=args.opinion_cap,
        )
        flagged = len(plan_d.flags)
        lines.append(
            f"[{mode}] dream: {len(plan_d.opinions)} opinions over "
            f"{plan_d.days_considered} days, {len(plan_d.rejected)} rejected, "
            f"{flagged} flagged. Audit: {path_d}"
        )

    return lines


def run(args: argparse.Namespace) -> int:
    """Entry point for ``familiar-connect sleep``."""
    # lazy — run.py pulls heavy deps (discord, voice stack)
    from familiar_connect.commands.run import _resolve_familiar_root  # noqa: PLC0415
    from familiar_connect.config import (  # noqa: PLC0415
        ConfigError,
        load_character_config,
    )
    from familiar_connect.history.async_store import AsyncHistoryStore  # noqa: PLC0415
    from familiar_connect.history.store import HistoryStore  # noqa: PLC0415
    from familiar_connect.llm import create_llm_clients  # noqa: PLC0415

    try:
        familiar_root = _resolve_familiar_root(args)
    except ValueError as exc:
        _logger.error("%s", exc)
        return 1
    familiar_id = familiar_root.name

    defaults_path = familiar_root.parent / "_default" / "character.toml"
    try:
        character_config = load_character_config(
            familiar_root / "character.toml", defaults_path=defaults_path
        )
    except (ConfigError, ValueError) as exc:
        _logger.error("Config error: %s", exc)
        return 1

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        _logger.error("OPENROUTER_API_KEY environment variable is required")
        return 1
    llm_clients = create_llm_clients(api_key, character_config)
    llm = llm_clients["background"]

    display_name = (
        character_config.aliases[0] if character_config.aliases else familiar_id.title()
    )
    audit_dir = (
        Path(args.audit_dir) if args.audit_dir else familiar_root / AUDIT_DIRNAME
    )

    store = HistoryStore(familiar_root / "history.db")
    try:
        lines = asyncio.run(
            _run_sleep(
                store=AsyncHistoryStore(store),
                llm=llm,
                familiar_id=familiar_id,
                display_name=display_name,
                display_tz=character_config.display_tz,
                audit_dir=audit_dir,
                args=args,
            )
        )
    finally:
        store.close()

    for line in lines:
        sys.stdout.write(line + "\n")
    return 0
