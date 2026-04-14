"""Tests for the `metrics` CLI subcommand."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from familiar_connect.commands import metrics as metrics_cmd
from familiar_connect.metrics.sqlite_collector import SQLiteCollector
from familiar_connect.metrics.types import StageSpan, TurnTrace

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _populate(db_path: Path, n: int = 3) -> None:
    collector = SQLiteCollector(db_path, flush_interval=1)
    for i in range(n):
        trace = TurnTrace(
            trace_id=f"t{i}",
            familiar_id="demo",
            channel_id=100,
            guild_id=None,
            modality="text",
            timestamp=f"2026-04-14T12:00:{i:02d}+00:00",
            total_duration_s=float(i + 1),
            stages=(
                StageSpan(name="llm_call", start_s=0.0, duration_s=float(i + 1) * 0.5),
            ),
            tags={"channel_mode": "full_rp"},
        )
        collector.record(trace)
    collector.close()


def _build_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "familiar": "demo",
        "last": 100,
        "stage": None,
        "tag": None,
        "compare": None,
        "plot": False,
        "familiars_root": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_add_parser_registers_subcommand() -> None:
    parent = argparse.ArgumentParser()
    subs = parent.add_subparsers()
    common = argparse.ArgumentParser(add_help=False)
    sub = metrics_cmd.add_parser(subs, common)
    # basic sanity: parser has our expected options
    actions = {a.dest for a in sub._actions}
    for name in ("familiar", "last", "stage", "tag", "compare", "plot"):
        assert name in actions


def test_run_missing_familiar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _build_args(familiar="does-not-exist", familiars_root=str(tmp_path))
    exit_code = metrics_cmd.run(args)
    assert exit_code == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "metrics.db" in out or "not found" in out.lower()


def test_run_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fam_root = tmp_path / "demo"
    fam_root.mkdir()
    _populate(fam_root / "metrics.db", n=5)

    args = _build_args(familiars_root=str(tmp_path))
    exit_code = metrics_cmd.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "traces:" in out.lower() or "5" in out
    assert "llm_call" in out


def test_run_with_compare_tag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fam_root = tmp_path / "demo"
    fam_root.mkdir()
    _populate(fam_root / "metrics.db", n=3)

    args = _build_args(compare="channel_mode", familiars_root=str(tmp_path))
    exit_code = metrics_cmd.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "channel_mode" in out
